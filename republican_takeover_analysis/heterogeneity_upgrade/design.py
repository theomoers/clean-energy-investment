"""Outcome-free subgroup-aware propensity and convex calibration candidates."""
import numpy as np
import pandas as pd
from scipy.optimize import minimize, linprog, Bounds, LinearConstraint
from scipy.special import expit

def ess(w):
    w=np.asarray(w,float)
    return w.sum()**2/(w@w) if w.sum()>0 else 0.

def pooled_propensity(x,state,t,group,penalty):
    states=pd.get_dummies(state,dtype=float).to_numpy()
    group=np.asarray(group,float)
    z=np.column_stack([states,x,group,group[:,None]*x,group[:,None]*states])
    k=states.shape[1];p=x.shape[1];n=len(t)
    ridge=np.r_[np.zeros(k),np.full(p,.001),0.,np.full(p+k,penalty)]
    def objective(b):
        eta=z@b
        value=np.mean(np.logaddexp(0,eta)-t*eta)+.5*np.sum(ridge*b*b)
        gradient=z.T@(expit(eta)-t)/n+ridge*b
        return value,gradient
    fit=minimize(objective,np.zeros(z.shape[1]),jac=True,method="BFGS",
                 options={"gtol":1e-9,"maxiter":1000})
    e=np.clip(expit(z@fit.x),1e-10,1-1e-10)
    w=np.where(t==1,1-e,e)
    return w,e,float(np.max(abs(objective(fit.x)[1])))

def diagnostic(x,state,t,group,w,names,state_sd):
    rows=[];cells=[];maxsmd=0.;maxcov=0.;maxstate=0.;unsupported=0.
    states=sorted(set(state));state=np.asarray(state);t=np.asarray(t);group=np.asarray(group)
    for g in [0,1]:
        gm=group==g;armweights={}
        for arm in [0,1]:
            m=gm&(t==arm);raw=w[m]
            if raw.sum()<=0:return dict(supported=False,max_smd=np.inf),[],[]
            norm=raw/raw.sum();armweights[arm]=(m,norm)
            cells.append(dict(group=g,arm=arm,n=int(m.sum()),nonzero=int((raw>1e-12).sum()),
                              ess=ess(raw),max_normalized_weight=float(norm.max()),
                              states=len(set(state[m]))))
        for j,name in enumerate(names):
            v=sum((1 if a else -1)*np.sum(ww*x[m,j]) for a,(m,ww) in armweights.items())
            rows.append(dict(group=g,feature=name,smd=v))
            maxcov=max(maxcov,abs(v));maxsmd=max(maxsmd,abs(v))
        no_both=[]
        for s in states:
            v=sum((1 if a else -1)*ww[state[m]==s].sum() for a,(m,ww) in armweights.items())/state_sd[s]
            rows.append(dict(group=g,feature="state_"+s,smd=v))
            maxstate=max(maxstate,abs(v));maxsmd=max(maxsmd,abs(v))
            if len(np.unique(t[gm&(state==s)]))<2:no_both.append(s)
        mass=sum(ww[np.isin(state[m],no_both)].sum() for m,ww in armweights.values())/2
        unsupported=max(unsupported,mass)
    miness=min(c["ess"] for c in cells);maxw=max(c["max_normalized_weight"] for c in cells)
    supported=miness>=5-1e-7 and min(c["states"] for c in cells)>=3
    out=dict(max_smd=maxsmd,max_covariate_smd=maxcov,max_state_smd=maxstate,
             min_cell_ess=miness,max_normalized_weight=maxw,
             max_one_party_state_mass=unsupported,supported=supported,
             passes_design=bool(supported and maxsmd<=.100001 and maxw<=.250001 and unsupported<=.100001))
    return out,rows,cells

def calibrate(x,state,t,group,state_sd,tolerance):
    state=np.asarray(state);t=np.asarray(t);group=np.asarray(group);n=len(t)
    equalities=[];a=[];lower=np.empty(n);upper=np.empty(n)
    for g in [0,1]:
        for arm in [0,1]:
            m=(group==g)&(t==arm);count=m.sum()
            if count<5:return None,"cell_has_fewer_than_five_areas"
            equalities.append(m.astype(float));lower[m]=.01/count;upper[m]=min(3/count,.20)
        sign=np.where(t==1,1.,-1.)*(group==g)
        a.extend((x*sign[:,None]).T)
        for s in sorted(set(state)):
            a.append(sign*(state==s)/state_sd[s])
    eq=np.array(equalities);a=np.asarray(a)
    feasible=linprog(np.zeros(n),A_ub=np.r_[a,-a],b_ub=np.full(2*len(a),tolerance),
                     A_eq=eq,b_eq=np.ones(4),bounds=list(zip(lower,upper)),method="highs")
    if not feasible.success:return None,"infeasible"
    opt=minimize(lambda w:(float(w@w),2*w),feasible.x,jac=True,method="SLSQP",
                 constraints=[LinearConstraint(eq,1.,1.),LinearConstraint(a,-tolerance,tolerance)],
                 bounds=Bounds(lower,upper),options={"ftol":1e-12,"maxiter":1000})
    w=opt.x
    valid=np.max(abs(eq@w-1))<1e-6 and np.max(abs(a@w))<=tolerance+1e-6 and np.min(w-lower)>-1e-7 and np.min(upper-w)>-1e-7
    if not valid:return None,"quadratic_solver_failed"
    return w,"optimal" if opt.success else "feasible_not_certified_optimal"
