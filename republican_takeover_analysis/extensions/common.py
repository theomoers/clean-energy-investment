"""Design and inference helpers; no outcomes enter the matching optimizer."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import milp, Bounds, LinearConstraint, minimize, brentq
from scipy.special import expit, roots_genlaguerre
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT/'republican_takeover_analysis/results'
AUDIT = ROOT/'majority_alignment_audit/verified_run'
MAIN = ['clean_announcements','manufacturing_announcements','wind_solar_net_mw']
FEATURES = ['log_clean_capacity','log_fossil_capacity','log_prior_announcements','log_core_area']
BASE = FEATURES + ['prior_announcement_slope']
EXTRA = ['log_prior_manufacturing','prior_manufacturing_slope','prior_net_mw','prior_net_mw_slope']


def mean_ci(y, states):
    y=np.asarray(y,float); states=np.asarray(states); g=len(np.unique(states));n=len(y)
    if g<3 or n<3:return dict(estimate=float(y.mean()),se=np.nan,lo=np.nan,hi=np.nan,p=np.nan,states=g,n=n)
    b=y.mean();scores=pd.Series(y-b).groupby(states).sum().to_numpy()
    se=np.sqrt(g/(g-1)*sum(scores**2)/n**2)
    t=stats.t.ppf(.975,g-1)
    return dict(estimate=b,se=se,lo=b-t*se,hi=b+t*se,
                p=2*stats.t.sf(abs(b/se),g-1) if se else float(b==0),states=g,n=n)


def cardinality(units, columns, caliper_columns, out, label, tol=.10):
    sd=units[columns].std().replace(0,1)
    z=units[columns]/sd
    pairs=[]
    for st,g in units.groupby('state'):
        for i in g.index[g.gainer.eq(1)]:
            for j in g.index[g.gainer.eq(0)]:
                diff=(z.loc[i]-z.loc[j]).to_numpy()
                distance=np.sqrt(np.mean((z.loc[i,caliper_columns]-z.loc[j,caliper_columns])**2))
                if distance<=1.5:pairs.append((i,j,st,distance,diff))
    n=len(pairs);assert n>0
    incidence=np.zeros((len(units),n))
    lookup={v:i for i,v in enumerate(units.index)}
    for k,(i,j,_,__,___) in enumerate(pairs):incidence[lookup[i],k]=incidence[lookup[j],k]=1
    diffs=np.column_stack([p[4] for p in pairs])
    mat=np.vstack([incidence,diffs-tol,-diffs-tol])
    constraint=LinearConstraint(mat,-np.inf,np.r_[np.ones(len(units)),np.zeros(2*len(columns))])
    args=dict(integrality=np.ones(n),bounds=Bounds(0,1),constraints=constraint,
              options={'time_limit':120,'mip_rel_gap':0})
    fit=milp(-np.ones(n),**args)
    assert fit.x is not None,fit.message
    count=int(np.rint(fit.x).sum());assert count>0
    args['constraints']=[constraint,LinearConstraint(np.ones((1,n)),count,count)]
    second=milp(np.array([p[3] for p in pairs]),**args)
    chosen=second.x if second.x is not None else fit.x
    selected=np.rint(chosen).astype(bool)
    assert np.max(mat@selected-np.r_[np.ones(len(units)),np.zeros(2*len(columns))])<1e-6
    records=[]
    for k,(i,j,st,distance,_) in enumerate(pairs):
        records.append(dict(config=label,R=units.loc[i,'unit'],D=units.loc[j,'unit'],state=st,
                            distance=distance,selected=selected[k]))
    ledger=pd.DataFrame(records);ledger.to_csv(out/f'{label}_matching_ledger.csv',index=False)
    selected=ledger[ledger.selected].copy().reset_index(drop=True)
    selected['pair']=[f'{label}_{i:03}' for i in range(len(selected))]
    assert selected.R.is_unique and selected.D.is_unique
    u=units.set_index('unit')
    for col in columns:selected[col]=(selected.R.map(u[col])+selected.D.map(u[col]))/2
    additions={}
    for col in units.columns:
        if any(col.endswith('_'+suffix) for suffix in ['pre','post','delta','placebo','trend_adjusted','exclude_last_pre']):
            additions[col+'_R']=selected.R.map(u[col]);additions[col+'_D']=selected.D.map(u[col])
            additions[col]=additions[col+'_R']-additions[col+'_D']
    selected=pd.concat([selected,pd.DataFrame(additions)],axis=1)
    return selected,dict(config=label,eligible=len(units),candidate_pairs=n,pairs=count,
        states=selected.state.nunique(),max_count_optimal=fit.status==0,cost_optimal=second.status==0,
        count_mip_gap=float(getattr(fit,'mip_gap',np.nan)),tolerance=tol)


def overlap_weights(units, columns, frequency=None):
    # Unpenalized logistic overlap weights: exact mean balance if score converges.
    x=units[columns].to_numpy(float);x=(x-x.mean(0))/np.where(x.std(0)>0,x.std(0),1)
    x=np.column_stack([pd.get_dummies(units.state,dtype=float).to_numpy(),x])
    w=units.gainer.to_numpy(float);f=np.ones(len(w)) if frequency is None else frequency
    def objective(b):
        a=x@b;p=expit(a)
        return np.sum(f*(np.logaddexp(0,a)-w*a)),x.T@(f*(p-w))
    fit=minimize(objective,np.zeros(x.shape[1]),jac=True,method='BFGS',
                 options={'gtol':1e-8,'maxiter':500})
    p=expit(x@fit.x);weight=f*np.where(w==1,1-p,p)
    score=np.max(np.abs(objective(fit.x)[1]))
    return weight,p,score


def contrast(y,t,w):
    return np.average(np.asarray(y)[t==1],weights=w[t==1])-np.average(np.asarray(y)[t==0],weights=w[t==0])


def add_outcomes(units, annual, preyears, postyears):
    units=units.copy()
    for outcome,t in annual.groupby('outcome'):
        wide=t.pivot(index='unit',columns='year',values='y').reindex(units.unit)
        assert wide[preyears+postyears].notna().all().all()
        pre=wide[preyears].mean(axis=1);post=wide[postyears].mean(axis=1)
        xx=np.array(preyears)-preyears[-1]
        slopes=wide[preyears].to_numpy()@(xx-xx.mean())/sum((xx-xx.mean())**2)
        prediction=pre.to_numpy()+slopes*(np.mean(postyears)-np.mean(preyears))
        for suffix,v in dict(pre=pre,post=post,delta=post-pre,placebo=wide[preyears[-1]]-wide[preyears[-2]],
                             trend_adjusted=post.to_numpy()-prediction,
                             exclude_last_pre=post-wide[preyears[:-1]].mean(axis=1)).items():
            units[f'{outcome}_{suffix}']=np.asarray(v)
    return units


def holm(p):
    a=np.asarray(p,float);out=np.full(len(a),np.nan);valid=np.flatnonzero(np.isfinite(a))
    ix=valid[np.argsort(a[valid])];out[ix]=np.minimum(1,np.maximum.accumulate(a[ix]*(len(valid)-np.arange(len(valid)))))
    return out


def power_equivalence(estimates, baseline, out):
    rows=[];design=[]
    for r in estimates.itertuples():
        if not np.isfinite(r.se) or r.se<=0:continue
        df=int(r.states)-1;b=r.estimate;se=r.se;crit=stats.t.ppf(.95,df)
        minimum=abs(b)+crit*se
        c=stats.t.ppf(.975,df)
        # Evaluate noncentral-t power via its normal/chi-square mixture. The
        # installed SciPy nct CDF occasionally returns NaN at finite noncentrality.
        nodes,weights=roots_genlaguerre(128,df/2-1);weights=weights/weights.sum()
        scale=np.sqrt(2*nodes/df)
        def power(effect):return weights@(stats.norm.cdf(effect/se-c*scale)+stats.norm.cdf(-effect/se-c*scale))
        assert abs(power(0)-.05)<1e-6
        mde=brentq(lambda v:power(v)-.8,0,10*se)
        base=baseline[(r.config,r.outcome)]
        absolute=([5,10,25,50] if 'mw' in r.outcome else
                  [.025,.05,.1,.2] if 'manufacturing' in r.outcome else [.1,.2,.5,1])
        for kind,thresholds in [('absolute',absolute),('baseline_fraction',[.10,.25,.50])]:
            for bound_index,value in enumerate(thresholds,1):
                bound=value if kind=='absolute' else value*abs(base)
                if bound<=0:continue
                p1=stats.t.sf((b+bound)/se,df);p2=stats.t.cdf((b-bound)/se,df)
                pe=max(p1,p2)
                lo=b-crit*se;hi=b+crit*se
                assert (pe<.05)==(lo>-bound and hi<bound)
                # Fixed-SE planning approximation: no power calculation at observed b.
                power_zero=max(0.,2*stats.norm.cdf(bound/se-crit)-1)
                required_se=bound/(crit+stats.norm.ppf(.9))
                factor=max(1.,(se/required_se)**2)
                rows.append(dict(config=r.config,outcome=r.outcome,version=r.version,estimate=b,se=se,states=r.states,
                    bound_type=kind,bound_index=bound_index,bound_value=value,bound=bound,pre_baseline=base,lo90=lo,hi90=hi,
                    lo95=b-c*se,hi95=b+c*se,significance_p=r.p,equivalence_p=pe,equivalent=pe<.05,
                    minimal_symmetric_bound=minimum,mde80=mde,power_equivalence_true_zero_approx=power_zero,
                    se_required_80pct_equivalence_approx=required_se,independent_information_multiplier_approx=factor))
        design.append(dict(config=r.config,outcome=r.outcome,version=r.version,pre_baseline=base,
                           mde80=mde,minimal_symmetric_equivalence_bound=minimum,states=r.states,se=se))
    result=pd.DataFrame(rows)
    # Absolute bounds differ in units across outcomes: adjust each predeclared
    # grid-position family together, not coincidentally equal numeric bounds.
    result['equivalence_p_holm']=result.groupby(['config','version','bound_type','bound_index']).equivalence_p.transform(holm)
    result['significance_p_holm']=result.groupby(['config','version','bound_type','bound_index']).significance_p.transform(holm)
    result.to_csv(out/'equivalence.csv',index=False)
    pd.DataFrame(design).to_csv(out/'power_summary.csv',index=False)
    return result
