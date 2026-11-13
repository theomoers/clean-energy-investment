"""Held-out augmented-score ranking validation and subgroup balancing audit."""
from pathlib import Path
import argparse,hashlib,json,os,subprocess,sys,shutil
os.environ.setdefault("MPLCONFIGDIR","/private/tmp/majority-mpl")
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/"republican_takeover_analysis/extensions"))
from common import overlap_weights,BASE,EXTRA,MAIN
from design import pooled_propensity,diagnostic,calibrate,ess
SOURCE=ROOT/"republican_takeover_analysis/overlap_forest/results_current"
FEATURES=BASE+EXTRA
VERSIONS=["delta","placebo","trend_adjusted"]

def ridge(x,y,a):
    penalty=np.eye(x.shape[1])*a;penalty[0,0]=0
    return np.linalg.solve(x.T@x+penalty,x.T@y)

def outcome_models(u,z,out):
    tr=u.fold.eq("train").to_numpy();te=~tr
    state=pd.get_dummies(u.state,dtype=float).to_numpy()
    x=np.column_stack([np.ones(len(u)),z,state])
    predictions={};cv=[];params=[]
    for o in MAIN:
        for v in VERSIONS:
            y=u[o+"_"+v].to_numpy()
            p=[]
            for arm in [0,1]:
                m=tr & u.gainer.eq(arm).to_numpy();losses={}
                for alpha in [.1,1.,10.,100.]:
                    losses[alpha]=[]
                    for s in sorted(u.loc[m,"state"].unique()):
                        train=m & u.state.ne(s).to_numpy();valid=m & u.state.eq(s).to_numpy()
                        b=ridge(x[train],y[train],alpha)
                        losses[alpha].append(np.mean((y[valid]-x[valid]@b)**2))
                    cv.append(dict(outcome=o,version=v,arm=arm,alpha=alpha,
                                   mse=float(np.mean(losses[alpha]))))
                alpha=min(losses,key=lambda a:(np.mean(losses[a]),a))
                b=ridge(x[m],y[m],alpha)
                p.append(x@b)
                observed=te & u.gainer.eq(arm).to_numpy()
                params.append(dict(outcome=o,version=v,arm=arm,alpha=alpha,
                                   training_areas=int(m.sum()),
                                   heldout_observed_arm_mse=float(np.mean((y[observed]-p[-1][observed])**2))))
            predictions[o,v]=np.column_stack(p)
    pd.DataFrame(cv).to_csv(out/"nuisance_training_cv.csv",index=False)
    pd.DataFrame(params).to_csv(out/"nuisance_models.csv",index=False)
    rows=[]
    for (o,v),m in predictions.items():
        for i in range(len(u)):
            rows.append(dict(unit=u.unit.iloc[i],fold=u.fold.iloc[i],outcome=o,version=v,m0=m[i,0],m1=m[i,1]))
    pd.DataFrame(rows).to_csv(out/"outcome_predictions.csv",index=False)
    return predictions

def bounded_score(y,t,e,m):
    h=e*(1-e);tau=m[:,1]-m[:,0]
    residual=y-np.where(t==1,m[:,1],m[:,0])
    num=h*tau+(t-e)*residual
    # Independently verify AIPW representation for positive propensities.
    direct=tau+t/e*(y-m[:,1])-(1-t)/(1-e)*(y-m[:,0])
    assert np.allclose(num,h*direct,rtol=1e-8,atol=1e-8)
    return h,num

def qini(priority,h,num,check=False):
    valid=h>1e-14
    priority=priority[valid];h=h[valid];num=num[valid]
    groups=pd.DataFrame(dict(f=priority,h=h,num=num)).groupby("f",sort=True).sum()
    mass=groups.h.to_numpy();n=groups.num.to_numpy();total=mass.sum()
    rank=(np.cumsum(mass)-mass/2)/total
    value=float(np.sum(n*(rank-.5))/total)
    if check:
        # Integrate q*TOC at every weighted tie-block boundary.
        mass=mass[::-1];n=n[::-1];q=np.r_[0,np.cumsum(mass)/total]
        curve=np.r_[0,np.cumsum(n)/total]-q*np.sum(n)/total
        assert np.isclose(value,np.trapezoid(curve,q),rtol=1e-8,atol=1e-10)
    return value

def weighted_median(f,h):
    order=np.argsort(f,kind="stable")
    return f[order][np.searchsorted(np.cumsum(h[order]),h.sum()/2)]

def statistics(u,idx,z,priorities,groups,e,predictions,frequency,check=False):
    t=u.gainer.to_numpy();freq=frequency[idx];ee=e[idx];tt=t[idx]
    hd=ee*(1-ee)*freq
    states=pd.get_dummies(u.loc[idx,"state"],dtype=float).to_numpy()
    design=np.column_stack([states,z[idx,FEATURES.index("log_core_area")],
                            z[idx,FEATURES.index("log_clean_capacity")]])
    results={};scores=[]
    for o in MAIN:
        for v in VERSIONS:
            m=predictions[o,v][idx];y=u.loc[idx,o+"_"+v].to_numpy()
            h,num=bounded_score(y,tt,ee,m);h*=freq;num*=freq
            if h.sum()<1e-12:continue
            results[o,v,"average"]=num.sum()/h.sum()
            results[o,v,"qini"]=qini(priorities[o][idx],h,num,check=check)
            g=groups[o]["original"][idx]
            if all(h[g==a].sum()>1e-12 for a in [0,1]):
                results[o,v,"original_gap"]=num[g==1].sum()/h[g==1].sum()-num[g==0].sum()/h[g==0].sum()
            else:results[o,v,"original_gap"]=np.nan
            coef=np.linalg.pinv(design.T@(h[:,None]*design))@(design.T@num)
            results[o,v,"blp_area"]=coef[-2];results[o,v,"blp_clean_capacity"]=coef[-1]
            if check:
                import statsmodels.api as sm
                model=sm.WLS(num/h,design,weights=h).fit()
                assert np.allclose(coef[-2:],model.params[-2:],rtol=1e-7,atol=1e-8)
            if v=="delta":
                h0,n0=bounded_score(y,tt,ee,np.zeros_like(m))
                results[o,v,"qini_ipw"]=qini(priorities[o][idx],h0*freq,n0*freq)
            if check:
                for j,i in enumerate(idx):
                    scores.append(dict(unit=u.unit.iloc[i],outcome=o,version=v,priority=priorities[o][i],
                                       propensity=ee[j],tilt=h[j],score=num[j]/h[j],bounded_score=num[j]))
    return results,scores

def design_audit(u,z,priorities,groups,out):
    te=u.fold.eq("test").to_numpy();t=u.loc[te,"gainer"].to_numpy();state=u.loc[te,"state"].to_numpy()
    states=pd.get_dummies(u.state,dtype=float)
    sd={s:float(states[s].std()) for s in states}
    candidates=[];balance=[];cells=[];weightrows=[];winners=[]
    for o in MAIN:
        for definition,g0 in groups[o].items():
            group=g0[te];orders=[]
            for alpha in [.001,.01,.1,1.]:
                w,e,gradient=pooled_propensity(z[te],state,t,group,alpha)
                name=f"pooled_ridge_{alpha:g}"
                diag,rows,cc=diagnostic(z[te],state,t,group,w,FEATURES,sd)
                diag.update(outcome=o,definition=definition,candidate=name,
                            family="pooled_propensity",status="converged" if gradient<1e-6 else "not_converged",
                            gradient=gradient,tolerance=np.nan)
                if gradient>=1e-6:diag["passes_design"]=False
                candidates.append(diag);orders.append((diag,w,e))
                for r in rows:balance.append(dict(outcome=o,definition=definition,candidate=name,**r))
                for r in cc:cells.append(dict(outcome=o,definition=definition,candidate=name,**r))
                for i,wi in zip(u.index[te],w):weightrows.append(dict(unit=u.unit.iloc[i],outcome=o,definition=definition,candidate=name,weight=wi))
            for tol in [.1,.2,.3]:
                w,status=calibrate(z[te],state,t,group,sd,tol)
                name=f"calibration_{tol:g}"
                if w is None:
                    candidates.append(dict(outcome=o,definition=definition,candidate=name,
                        family="calibration",status=status,tolerance=tol,passes_design=False))
                    continue
                diag,rows,cc=diagnostic(z[te],state,t,group,w,FEATURES,sd)
                diag.update(outcome=o,definition=definition,candidate=name,
                            family="calibration",status=status,tolerance=tol,gradient=np.nan)
                candidates.append(diag);orders.append((diag,w,None))
                for r in rows:balance.append(dict(outcome=o,definition=definition,candidate=name,**r))
                for r in cc:cells.append(dict(outcome=o,definition=definition,candidate=name,**r))
                for i,wi in zip(u.index[te],w):weightrows.append(dict(unit=u.unit.iloc[i],outcome=o,definition=definition,candidate=name,weight=wi))
            eligible=[a for a in orders if a[0].get("passes_design",False)]
            selected=sorted(eligible,key=lambda a:(-a[0]["min_cell_ess"],a[0]["max_smd"],a[0]["candidate"]))[0] if eligible else None
            winners.append(dict(outcome=o,definition=definition,selected_candidate=selected[0]["candidate"] if selected else "",
                                passes_design=selected is not None))
    pd.DataFrame(candidates).to_csv(out/"design_frontier.csv",index=False)
    pd.DataFrame(balance).to_csv(out/"subgroup_balance.csv",index=False)
    pd.DataFrame(cells).to_csv(out/"subgroup_support.csv",index=False)
    pd.DataFrame(weightrows).to_csv(out/"subgroup_weights.csv",index=False)
    pd.DataFrame(winners).to_csv(out/"selected_designs.csv",index=False)
    return winners

def infer(out,point,draws,reps,states):
    rows=[]
    for key,value in point.items():
        b=np.asarray(draws[key]);b=b[np.isfinite(b)]
        reliable=len(b)>=.95*reps
        se=b.std(ddof=1) if reliable else np.nan
        c=stats.t.ppf(.975,states-1)
        p=2*stats.t.sf(abs(value/se),states-1) if se>0 else (1. if se==0 else np.nan)
        rows.append(dict(outcome=key[0],version=key[1],statistic=key[2],estimate=value,se=se,
                         lo=value-c*se,hi=value+c*se,p=p,states=states,
                         successful_draws=len(b),reliable_bootstrap=reliable))
    r=pd.DataFrame(rows)
    for (v,s),g in r.groupby(["version","statistic"]):
        r.loc[g.index,"p_holm_outcomes"]=multipletests(g.p.fillna(1),method="holm")[1]
    ix=r.statistic.eq("qini")
    r.loc[ix,"p_holm_all_qini_versions"]=multipletests(r.loc[ix,"p"].fillna(1),method="holm")[1]
    ix=r.version.eq("delta")&r.statistic.isin(["blp_area","blp_clean_capacity"])
    r.loc[ix,"p_holm_six_modifiers"]=multipletests(r.loc[ix,"p"].fillna(1),method="holm")[1]
    r.to_csv(out/"adjusted_heterogeneity.csv",index=False)
    joint=[]
    for o in MAIN:
        keys=[(o,"delta","blp_area"),(o,"delta","blp_clean_capacity")]
        b=np.column_stack([draws[k] for k in keys])
        b=b[np.isfinite(b).all(axis=1)]
        covariance=np.cov(b,rowvar=False)
        beta=np.array([point[k] for k in keys])
        rank=np.linalg.matrix_rank(covariance)
        f=beta@np.linalg.pinv(covariance)@beta/rank
        joint.append(dict(outcome=o,F=f,restrictions=rank,p=stats.f.sf(f,rank,states-1)))
    j=pd.DataFrame(joint);j["p_holm"]=multipletests(j.p,method="holm")[1]
    j.to_csv(out/"joint_modifier_tests.csv",index=False)
    pd.DataFrame({"|".join(k):pd.Series(v) for k,v in draws.items()}).to_csv(out/"bootstrap_statistics.csv",index=False)
    return r,j

def finish(out,u,r,j,selected,reps):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels={"delta":"Main","placebo":"Pre-period placebo","trend_adjusted":"Trend adjusted"}
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout="constrained")
    for ax,o in zip(axes,MAIN):
        g=r[r.outcome.eq(o)&r.statistic.eq("qini")]
        for k,row in enumerate(g.itertuples()):
            ax.errorbar(row.estimate,k,xerr=[[row.estimate-row.lo],[row.hi-row.estimate]],fmt="o",capsize=3)
        ax.set_yticks(range(len(g)),[labels[v] for v in g.version]);ax.axvline(0,color="gray")
        ax.set_title(o.replace("_"," "));ax.set_xlabel("Adjusted QINI integral")
    fig.suptitle("Frozen forest ranking: held-out adjusted-score validation (95% intervals)")
    fig.savefig(out/"adjusted_ranking.png",dpi=180);plt.close(fig)
    frontier=pd.read_csv(out/"design_frontier.csv")
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout="constrained")
    colors=dict(zip(MAIN,["#2878b5","#d17d24","#398552"]))
    names=dict(zip(MAIN,["Clean power","Manufacturing","Net capacity"]))
    for ax,measure,title in zip(axes,["max_covariate_smd","max_state_smd"],
                                ["Nine baseline covariates","State composition"]):
        for o in MAIN:
            for definition,marker in [("original","o"),("rank_half","^")]:
                g=frontier[frontier.outcome.eq(o)&frontier.definition.eq(definition)&frontier.status.ne("infeasible")]
                ax.scatter(g.min_cell_ess,g[measure],color=colors[o],marker=marker,
                           label=names[o]+(" / original cutoff" if definition=="original" else " / rank halves"),s=35)
        ax.axhline(.1,ls="--",color="gray");ax.axvline(5,ls=":",color="gray")
        ax.set(title=title,xlabel="Smallest group/party effective sample",ylabel="Largest absolute subgroup SMD")
    axes[1].legend(fontsize=7)
    fig.suptitle("Subgroup weighting frontier: every feasible candidate retains all 101 evaluation areas")
    fig.savefig(out/"balance_frontier.png",dpi=180);plt.close(fig)
    summary=dict(date="2026-11-07",areas=196,training_areas=95,evaluation_areas=101,
                 training_states=15,evaluation_states=10,bootstrap_reps=reps,
                 ranking_reproduced=True,outcome_nuisances_training_only=True,
                 qini_verified_by_exact_numeric_integration=True,
                 projections_verified_against_independent_WLS=True,
                 outcome_free_weight_selection=True,passing_designs=int(sum(s["passes_design"] for s in selected)),
                 manuscript_changed=False)
    (out/"verification.json").write_text(json.dumps(summary,indent=2)+"\n")
    sources=[SOURCE/"areas.csv",SOURCE/"forest_predictions.csv",
             ROOT/"republican_takeover_analysis/extensions/common.py"]+sorted(HERE.glob("*.py"))+sorted(HERE.glob("*.R"))+[HERE/"PROTOCOL.txt"]
    pd.DataFrame([dict(file=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sources]).to_csv(out/"input_code_hashes.csv",index=False)
    print(r[r.version.eq("delta")&r.statistic.isin(["qini","average","original_gap"])].to_string(index=False),flush=True)
    print(j.to_string(index=False),flush=True)
    print(json.dumps(summary,indent=2),flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--bootstrap-reps",type=int,default=999);args=parser.parse_args()
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    u=pd.read_csv(SOURCE/"areas.csv")
    # Preserve CSV bytes: tiny Python/R float-parser roundtrips can alter
    # near-tied forest split decisions despite identical economic inputs.
    shutil.copy2(SOURCE/"areas.csv",out/"areas.csv")
    assert len(u)==196 and u.fold.eq("test").sum()==101
    subprocess.run(["/usr/local/bin/Rscript","--vanilla",str(HERE/"ranking.R"),str(out)],check=True)
    prior=pd.read_csv(SOURCE/"forest_predictions.csv")
    current=pd.read_csv(out/"priorities.csv")
    prior=prior[prior.fold.eq("test")].merge(current,on=["unit","outcome","fold"],validate="1:1")
    assert np.allclose(prior.heldout_score,prior.priority,rtol=1e-10,atol=1e-10)
    z=((u[FEATURES]-u[FEATURES].mean())/u[FEATURES].std()).to_numpy()
    e=u.propensity.to_numpy();priorities={};groups={};group_rows=[]
    old=pd.read_csv(SOURCE/"forest_predictions.csv")
    for o in MAIN:
        f=current[current.outcome.eq(o)].set_index("unit").loc[u.unit,"priority"].to_numpy()
        cutoff=float(old[old.outcome.eq(o)].training_cutoff.iloc[0])
        train=u.fold.eq("train").to_numpy();te=~train
        assert np.isclose(weighted_median(f[train],u.loc[train,"training_propensity"].to_numpy()*(1-u.loc[train,"training_propensity"].to_numpy())),cutoff)
        half=weighted_median(f[te],e[te]*(1-e[te]))
        priorities[o]=f;groups[o]={"original":f>cutoff,"rank_half":f>half}
        for definition,g in groups[o].items():
            for i in np.flatnonzero(te):
                group_rows.append(dict(unit=u.unit.iloc[i],outcome=o,definition=definition,high=int(g[i]),
                                       cutoff=cutoff if definition=="original" else half))
    pd.DataFrame(group_rows).to_csv(out/"group_definitions.csv",index=False)
    # Complete design audit before fitting any new outcome model.
    selected=design_audit(u,z,priorities,groups,out)
    print("Design-only weight audit complete:",selected,flush=True)
    predictions=outcome_models(u,z,out)
    idx=np.flatnonzero(u.fold.eq("test"));t=u.gainer.to_numpy()
    point,scores=statistics(u,idx,z,priorities,groups,e,predictions,np.ones(len(u)),check=True)
    pd.DataFrame(scores).to_csv(out/"evaluation_scores.csv",index=False)
    rng=np.random.default_rng(22481);states=sorted(u.loc[idx,"state"].unique())
    draws={k:[] for k in point};failed=0
    for b in range(args.bootstrap_reps):
        counts=pd.Series(rng.choice(states,len(states),replace=True)).value_counts()
        frequency=np.where(u.fold.eq("test"),u.state.map(counts).fillna(0),1).astype(float)
        _,eb,score=overlap_weights(u,FEATURES,frequency)
        if score<1e-5:
            ss,_=statistics(u,idx,z,priorities,groups,eb,predictions,frequency)
            for k in draws:draws[k].append(ss.get(k,np.nan))
        else:
            failed+=1
            for k in draws:draws[k].append(np.nan)
        if (b+1)%250==0:print("Refitted-score bootstrap",b+1,flush=True)
    assert failed<.05*args.bootstrap_reps
    r,j=infer(out,point,draws,args.bootstrap_reps,len(states))
    finish(out,u,r,j,selected,args.bootstrap_reps)
    subprocess.run(["/usr/local/bin/Rscript","--vanilla",str(HERE/"rate_check.R"),str(out)],check=True)

if __name__=="__main__":main()
