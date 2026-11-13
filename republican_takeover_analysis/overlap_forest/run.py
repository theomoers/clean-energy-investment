"""Rebuild separate overlap-targeted forest audit from frozen prior inputs."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
import sys
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/majority-mpl")
import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "republican_takeover_analysis/extensions"))
from common import overlap_weights, BASE, EXTRA, MAIN, contrast
from analyze import balance
FEATURES = BASE + EXTRA
MODIFIERS = ["log_clean_capacity", "log_fossil_capacity", "log_prior_announcements", "log_core_area"]
SOURCE = ROOT / "republican_takeover_analysis/extensions/results_current"
VERSIONS = ["delta", "placebo", "trend_adjusted", "exclude_last_pre"]

def ess(w):
    w = np.asarray(w)
    return w.sum()**2/(w@w) if w.sum()>0 else 0.

def weights(u, frequency=None):
    w, p, score = overlap_weights(u, FEATURES, frequency)
    return w, p, bool(score<1e-5 and np.min([w[u.gainer.eq(t)].sum() for t in [0,1]])>1e-10)

def ci(point, draws, df):
    b = np.asarray(draws)
    b = b[np.isfinite(b)]
    if len(b)<2:
        return dict(estimate=point,se=np.nan,lo=np.nan,hi=np.nan,p=np.nan,successful_draws=len(b))
    se = b.std(ddof=1); t = stats.t.ppf(.975,df)
    return dict(estimate=point,se=se,lo=point-t*se,hi=point+t*se,
                p=2*stats.t.sf(abs(point/se),df) if se else float(point==0),successful_draws=len(b))

def subgroup_support(u, mask, w, outcome, label):
    g = u.loc[mask]; wg = w[mask]; t = g.gainer.to_numpy()
    result = dict(outcome=outcome,group=label,areas=len(g),states=g.state.nunique())
    for arm, name in [(0,"D"),(1,"R")]:
        a = t==arm; ww=wg[a]
        result.update({f"areas_{name}":int(a.sum()),f"ess_{name}":ess(ww),
                       f"states_{name}":g.loc[a,"state"].nunique(),
                       f"post_total_{name}":float((g.loc[a,outcome+"_post"]*2).sum())})
    result["post_units"]="net MW" if outcome=="wind_solar_net_mw" else "announcements"
    result["supported"] = min(result["areas_R"],result["areas_D"])>=4 and min(result["ess_R"],result["ess_D"])>=5 and min(result["states_R"],result["states_D"])>=3
    return result

def group_balance(u, mask, w, outcome, label, version):
    g=u.loc[mask]; t=g.gainer.to_numpy(); wg=w[mask]
    if len(np.unique(t))<2:return []
    rows=[]
    z=pd.concat([u[FEATURES],pd.get_dummies(u.state,prefix="state",dtype=float)],axis=1)
    for col in z:
        sd=z[col].std()
        value=contrast(z.loc[mask,col],t,wg)/sd if sd else 0
        rows.append(dict(outcome=outcome,group=label,version=version,feature=col,smd=value))
    return rows

def prepare(out):
    u=pd.read_csv(SOURCE/"units2023.csv")
    assert len(u)==196 and u.unit.is_unique and u.state.nunique()==25
    assert np.isfinite(u[FEATURES]).all().all()
    w,p,ok=weights(u);assert ok
    old=pd.read_csv(SOURCE/"overlap_weights.csv").set_index("unit").loc[u.unit]
    assert np.allclose(w,old.weight,rtol=1e-6,atol=1e-8)
    states=np.random.default_rng(48123).permutation(sorted(u.state.unique()))
    test=set(states[:len(states)*2//5]);u["fold"]=np.where(u.state.isin(test),"test","train")
    u["propensity"]=p;u["weight"]=w
    train=u.fold.eq("train")
    _,pt,ok=weights(u.loc[train]);assert ok
    u["training_propensity"]=np.nan;u.loc[train,"training_propensity"]=pt
    u.to_csv(out/"areas.csv",index=False)
    pd.DataFrame({"state":states,"fold":["test" if s in test else "train" for s in states]}).to_csv(out/"folds.csv",index=False)
    pd.DataFrame(balance(u,None,FEATURES,"overlap",w)).to_csv(out/"overall_balance.csv",index=False)
    return u

def full_diagnostics(out,u,reps):
    w=u.weight.to_numpy();t=u.gainer.to_numpy()
    annual=pd.read_csv(SOURCE/"annual2023.csv")
    arrays=[];metadata=[]
    for o in MAIN:
        for v in VERSIONS:
            arrays.append(u[o+"_"+v].to_numpy());metadata.append(dict(outcome=o,version=v,year=np.nan))
        a=annual[annual.outcome.eq(o)].pivot(index="unit",columns="year",values="y").reindex(u.unit)
        assert a.notna().all().all()
        for year in a:
            arrays.append((a[year]-a[2022]).to_numpy());metadata.append(dict(outcome=o,version="annual_vs2022",year=int(year)))
    y=np.column_stack(arrays)
    point=np.array([contrast(y[:,j],t,w) for j in range(y.shape[1])])
    rng=np.random.default_rng(11481);states=sorted(u.state.unique());draws=[]
    for b in range(reps):
        counts=pd.Series(rng.choice(states,len(states),replace=True)).value_counts()
        f=u.state.map(counts).fillna(0).to_numpy()
        wb,_,ok=weights(u,f)
        if ok:draws.append([contrast(y[:,j],t,wb) for j in range(y.shape[1])])
        if (b+1)%250==0:print("Average/pretrend bootstrap",b+1,flush=True)
    boot=np.array(draws);assert len(boot)>=.95*reps
    rows=[dict(**m,**ci(p,boot[:,j],len(states)-1)) for j,(m,p) in enumerate(zip(metadata,point))]
    results=pd.DataFrame(rows)
    for v in VERSIONS:
        mask=results.version.eq(v)
        from statsmodels.stats.multitest import multipletests
        results.loc[mask,"p_holm"]=multipletests(results.loc[mask,"p"],method="holm")[1]
    results.to_csv(out/"weighted_effects_and_annual.csv",index=False)
    prior=pd.read_csv(SOURCE/"estimates.csv")
    for o in MAIN:
        for v in VERSIONS:
            a=results[results.outcome.eq(o)&results.version.eq(v)].iloc[0]
            b=prior[prior.config.eq("overlap2023")&prior.outcome.eq(o)&prior.version.eq(v)].iloc[0]
            assert np.allclose(a[["estimate","se","lo","hi"]].astype(float),
                               b[["estimate","se","lo","hi"]].astype(float),rtol=1e-7,atol=1e-8)
    np.save(out/"average_bootstrap.npy",boot)
    joint=[]
    for o in MAIN:
        ix=[j for j,m in enumerate(metadata) if m["outcome"]==o and m["version"]=="annual_vs2022" and m["year"]<2022]
        covariance=np.cov(boot[:,ix],rowvar=False)
        rank=np.linalg.matrix_rank(covariance)
        f=point[ix]@np.linalg.pinv(covariance)@point[ix]/rank
        joint.append(dict(outcome=o,restrictions=rank,states=25,F=f,p=stats.f.sf(f,rank,24)))
    pd.DataFrame(joint).to_csv(out/"weighted_joint_pretests.csv",index=False)
    return results

def evaluate(out,u,reps):
    predictions=pd.read_csv(out/"forest_predictions.csv")
    w=u.weight.to_numpy();t=u.gainer.to_numpy();test=u.fold.eq("test").to_numpy()
    support=[];balances=[];settings={};repair_support=[]
    for o in MAIN:
        pred=predictions[predictions.outcome.eq(o)].set_index("unit").reindex(u.unit)
        high=pred.high.eq(1).to_numpy() & test
        low=pred.high.eq(0).to_numpy() & test
        masks=[low,high]
        groups=[]
        for label,mask in zip(["low","high"],masks):
            s=subgroup_support(u,mask,w,o,label);support.append(s)
            balances+=group_balance(u,mask,w,o,label,"full_design_weights")
            groups.append(s)
        supported=all(s["supported"] for s in groups) and u.loc[test,"state"].nunique()>=4
        point={}
        for v in VERSIONS:
            vals=u[o+"_"+v].to_numpy()
            rates=[contrast(vals[m],t[m],w[m]) if len(np.unique(t[m]))==2 else np.nan for m in masks]
            point[v]=dict(low=rates[0],high=rates[1],gap=rates[1]-rates[0])
        # Repair uses identical states for both groups and all nine covariates.
        common=set(u.loc[test,"state"])
        for m in masks:
            both=u.loc[m].groupby("state").gainer.nunique()
            common &= set(both[both.eq(2)].index)
        repair_masks=[m & u.state.isin(common).to_numpy() for m in masks]
        repair_w=[];repair_ok=len(common)>=4
        for label,m in zip(["low","high"],repair_masks):
            if m.sum():
                rw,_,ok=weights(u.loc[m])
                expanded=np.zeros(len(u));expanded[m]=rw
                s=subgroup_support(u,m,expanded,o,label);s.update(version="recalibrated_common_states",converged=ok)
                repair_support.append(s)
                if ok:balances+=group_balance(u,m,expanded,o,label,"recalibrated_common_states")
                repair_ok &= ok and s["supported"]
                repair_w.append(expanded)
            else:
                repair_ok=False;repair_w.append(np.zeros(len(u)))
        settings[o]=dict(masks=masks,point=point,supported=supported,common=sorted(common),
                         repair_masks=repair_masks,repair_weights=repair_w,repair_ok=repair_ok)
    pd.DataFrame(support).to_csv(out/"heldout_support.csv",index=False)
    pd.DataFrame(repair_support).to_csv(out/"recalibrated_support.csv",index=False)
    pd.DataFrame(balances).to_csv(out/"heldout_balance.csv",index=False)
    rng=np.random.default_rng(22481);teststates=sorted(u.loc[test,"state"].unique())
    boots={(o,v):[] for o in MAIN for v in VERSIONS}
    repairs={o:[] for o in MAIN}
    for b in range(reps):
        counts=pd.Series(rng.choice(teststates,len(teststates),replace=True)).value_counts()
        frequency=np.where(test,u.state.map(counts).fillna(0),1).astype(float)
        wb,_,ok=weights(u,frequency)
        for o,s in settings.items():
            if ok:
                mm=s["masks"]
                if all(min(wb[m & (t==a)].sum() for a in [0,1])>1e-10 for m in mm):
                    for v in VERSIONS:
                        y=u[o+"_"+v].to_numpy()
                        val=[contrast(y[m],t[m],wb[m]) for m in mm]
                        boots[o,v].append(val[1]-val[0])
            if s["repair_ok"]:
                vals=[];valid=True
                for m in s["repair_masks"]:
                    f=frequency[m]
                    rw,_,success=weights(u.loc[m],f)
                    if not success:valid=False;break
                    vals.append(contrast(u.loc[m,o+"_delta"],t[m],rw))
                if valid:repairs[o].append(vals[1]-vals[0])
        if (b+1)%250==0:print("Held-out bootstrap",b+1,flush=True)
    rows=[]
    for o,s in settings.items():
        for v,p in s["point"].items():
            r=ci(p["gap"],boots[o,v],len(teststates)-1)
            r.update(outcome=o,version=v,low=p["low"],high=p["high"],supported=s["supported"],
                     bootstrap_reliable=len(boots[o,v])>=.95*reps,method="full_design_weights")
            rows.append(r)
        if s["repair_ok"]:
            y=u[o+"_delta"].to_numpy()
            vv=[contrast(y[m],t[m],rw[m]) for m,rw in zip(s["repair_masks"],s["repair_weights"])]
            r=ci(vv[1]-vv[0],repairs[o],len(s["common"])-1)
            reliable=len(repairs[o])>=.95*reps
            if not reliable:
                for k in ["se","lo","hi","p"]:r[k]=np.nan
            r.update(outcome=o,version="delta",low=vv[0],high=vv[1],supported=True,
                     bootstrap_reliable=reliable,method="recalibrated_common_states")
            rows.append(r)
    r=pd.DataFrame(rows)
    from statsmodels.stats.multitest import multipletests
    for (method,version),g in r.groupby(["method","version"]):
        # Keep the three-outcome family even when some comparisons are unavailable.
        pp=g.p.fillna(1).to_numpy()
        pp=np.r_[pp,np.ones(max(0,3-len(pp)))]
        r.loc[g.index,"p_holm"]=multipletests(pp,method="holm")[1][:len(g)]
    r.to_csv(out/"heldout_validation.csv",index=False)
    pd.DataFrame([dict(outcome=o,common_states=",".join(s["common"]),recalibration_supported=s["repair_ok"])
                  for o,s in settings.items()]).to_csv(out/"recalibration_status.csv",index=False)
    pd.DataFrame({o+"_"+v:pd.Series(b) for (o,v),b in boots.items()}).to_csv(out/"heldout_bootstrap.csv",index=False)
    return r

def finish(out,u,results,validation):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout="constrained")
    for ax,o in zip(axes,MAIN):
        a=results[results.outcome.eq(o)&results.version.eq("annual_vs2022")].sort_values("year")
        ax.errorbar(a.year,a.estimate,yerr=[a.estimate-a.lo,a.hi-a.estimate],fmt="o-",capsize=3)
        ax.axvline(2022.5,ls="--",color="gray");ax.axhline(0,color="gray")
        ax.set_title(o.replace("_","\n"));ax.set_xlabel("Year")
    fig.suptitle("Overlap-weighted R-D annual contrasts relative to 2022; 95% intervals")
    fig.savefig(out/"weighted_pretrends.png",dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout="constrained")
    for ax,o in zip(axes,MAIN):
        a=validation[validation.outcome.eq(o)&validation.method.eq("full_design_weights")&validation.version.isin(["delta","placebo","trend_adjusted"])]
        for j,row in enumerate(a.itertuples()):
            if np.isfinite(row.lo):ax.errorbar(row.estimate,j,xerr=[[row.estimate-row.lo],[row.hi-row.estimate]],fmt="o",capsize=3)
        ax.set_yticks(range(len(a)),a.version);ax.axvline(0,color="gray")
        bb=pd.read_csv(out/"heldout_balance.csv")
        bb=bb[bb.outcome.eq(o)&bb.version.eq("full_design_weights")]
        maxsmd=bb.smd.abs().max()
        ax.set_title(o.replace("_","\n") + f"\nMax subgroup SMD: {maxsmd:.2f}")
    fig.suptitle("Held-out high-minus-low contrasts: support/balance flags must also pass")
    fig.savefig(out/"heldout_comparisons.png",dpi=180);plt.close(fig)
    paths=[SOURCE/"units2023.csv",SOURCE/"annual2023.csv",SOURCE/"overlap_weights.csv",
           SOURCE/"overlap_support.csv",SOURCE/"estimates.csv",
           ROOT/"republican_takeover_analysis/extensions/common.py",
           ROOT/"republican_takeover_analysis/extensions/analyze.py"]+sorted(HERE.glob("*.py"))+sorted(HERE.glob("*.R"))+[HERE/"PROTOCOL.txt"]
    pd.DataFrame([dict(file=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths]).to_csv(out/"input_code_hashes.csv",index=False)
    assert set(pd.read_csv(out/"forest_predictions.csv").unit)==set(u.unit)
    # Independently reproduce native GRF overlap score by residual OLS.
    import statsmodels.api as sm
    from statsmodels.stats.sandwich_covariance import cov_cluster
    ns=pd.read_csv(out/"nuisance_predictions.csv")
    native=pd.read_csv(out/"native_overlap_score.csv")
    for o in MAIN:
        g=ns[ns.outcome.eq(o)].merge(u[["unit","state"]],on="unit",validate="1:1")
        model=sm.OLS(g.residual_y,sm.add_constant(g.residual_w)).fit()
        row=native[native.outcome.eq(o)].iloc[0]
        assert np.isclose(model.params.iloc[1],row.estimate)
        assert np.isclose(np.sqrt(cov_cluster(model,g.state)[1,1]),row.se,rtol=1e-6)
    bal=pd.read_csv(out/"heldout_balance.csv")
    audits=[]
    for o in MAIN:
        b=bal[bal.outcome.eq(o)&bal.version.eq("full_design_weights")]
        v=validation[validation.outcome.eq(o)&validation.version.eq("delta")&validation.method.eq("full_design_weights")].iloc[0]
        audits.append(dict(outcome=o,max_covariate_smd=float(b.loc[b.feature.isin(FEATURES),"smd"].abs().max()),
                          max_state_smd=float(b.loc[b.feature.str.startswith("state_"),"smd"].abs().max()),
                          subgroup_sample_support=bool(v.supported),
                          passes_support_and_balance=bool(v.supported and b.smd.abs().max()<=.1)))
    pd.DataFrame(audits).to_csv(out/"heterogeneity_credibility.csv",index=False)
    summary=dict(areas=len(u),states=u.state.nunique(),train_areas=int(u.fold.eq("train").sum()),
                 test_areas=int(u.fold.eq("test").sum()),train_states=u.loc[u.fold.eq("train"),"state"].nunique(),
                 test_states=u.loc[u.fold.eq("test"),"state"].nunique(),
                 ess_R=ess(u.loc[u.gainer.eq(1),"weight"]),ess_D=ess(u.loc[u.gainer.eq(0),"weight"]),
                 original_weights_reproduced=True,training_outcomes_state_disjoint=True,
                 no_manual_training_overlap_weights=True,
                 prior_average_effects_and_uncertainty_reproduced=True,
                 native_overlap_score_and_cluster_se_independently_verified=True,
                 subgroup_credibility=audits,
                 weighted_pretrend_tests=pd.read_csv(out/"weighted_joint_pretests.csv").to_dict("records"))
    (out/"verification.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2),flush=True)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",required=True,type=Path)
    parser.add_argument("--bootstrap-reps",type=int,default=999)
    args=parser.parse_args();out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    u=prepare(out)
    subprocess.run(["/usr/local/bin/Rscript","--vanilla",str(HERE/"forest.R"),str(out)],check=True)
    results=full_diagnostics(out,u,args.bootstrap_reps)
    validation=evaluate(out,u,args.bootstrap_reps)
    finish(out,u,results,validation)
if __name__=="__main__":main()
