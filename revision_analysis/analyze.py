"""Reproduce the manufacturing analysis into a new or empty directory."""
import argparse
import hashlib
import os
import tempfile
from importlib.metadata import version
from pathlib import Path

os.environ.setdefault("NUMBA_DISABLE_JIT","1")
os.environ.setdefault("MPLCONFIGDIR",str(Path(tempfile.gettempdir())/"mfg-mpl"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from rdrobust import rdrobust
from rddensity import rddensity
from panel import DATA, HERE, build, lookup, terms
from verify import verify


def rd(d,outcome,spec="primary",h=None,scale=1,b=None):
    s=d.dropna(subset=[outcome]).copy()
    cov=pd.get_dummies(s.year.astype(str),prefix="cycle",drop_first=True,dtype=float)
    fit=rdrobust(s[outcome]*scale,s.running,covs=cov if cov.shape[1] else None,
        cluster=pd.factorize(s.state)[0],vce="cr1",p=1,q=2,kernel="tri",bwselect="mserd",
        h=h,b=b if b is not None else (None if h is None else 2*h))
    hl,hr=fit.bws.iloc[0]
    left=s[(s.running<0)&(s.running>-hl)]
    right=s[(s.running>=0)&(s.running<hr)]
    row=dict(outcome=outcome,spec=spec,n=len(s),estimate=fit.coef.iloc[2,0],se=fit.se.iloc[2,0],
        low=fit.ci.iloc[2,0],high=fit.ci.iloc[2,1],p=fit.pv.iloc[2,0],
        h_left=hl,h_right=hr,b_left=fit.bws.iloc[1,0],b_right=fit.bws.iloc[1,1],
        states=pd.concat([left,right]).state.nunique(),scale=scale)
    for side,subset in [("D",left),("R",right)]:
        row.update({side+"_races":len(subset),side+"_states":subset.state.nunique(),
            side+"_districts_with_projects":subset.any_projects.sum(),
            side+"_mean":subset[outcome].mean()*scale})
    assert [len(left),len(right)]==list(fit.N_h)
    return row


def associations(f,out):
    sample=f[f.historical_party.isin(["Republican","Democrat"]) &
             f.legacy_party.isin(["Republican","Democratic","Democrat"]) & f.capex.notna()].copy()
    sample["geo_cluster"]=sample.state+"_"+sample.congress.astype(str)+"_"+sample.district.astype(str)
    sample["legacy_R"]=sample.legacy_party.eq("Republican").astype(int)
    sample["historical_R"]=sample.historical_party.eq("Republican").astype(int)
    sample["log_capex"]=np.log1p(sample.capex)
    sample[["unique_id","legacy_R","historical_R","log_capex","geo_cluster"]].to_csv(out/"common_projects.csv",index=False)
    results=[]
    for term in ["legacy_R","historical_R"]:
        fit=smf.ols(f"log_capex ~ {term} + C(state) + C(quarter) + C(Technology)",sample).fit(
            cov_type="cluster",cov_kwds={"groups":sample.geo_cluster},use_t=True)
        results.append(dict(party_measure=term,n=int(fit.nobs),clusters=sample.geo_cluster.nunique(),
            relabeled=(sample.legacy_R!=sample.historical_R).sum(),estimate=fit.params[term],
            low=fit.conf_int().loc[term,0],high=fit.conf_int().loc[term,1],p=fit.pvalues[term]))
    pd.DataFrame(results).to_csv(out/"project_size_associations.csv",index=False)
    # Bridge sector composition at a fixed estimator, period, coding and inclusion rule.
    raw=pd.read_csv(DATA/"manufacturing_energy_and_industry_facility_metadata.csv",skiprows=5,low_memory=False)
    raw=raw.rename(columns={"Estimated_Total_Facility_CAPEX":"capex","US Representative Party":"legacy_party","State":"state"})
    date=pd.to_datetime(raw.Announcement_Date)
    raw["quarter"]=date.dt.to_period("Q").astype(str)
    raw=raw[date.dt.year.between(2018,2024)&raw.LatLon_Valid&raw.capex.notna()&
            raw.legacy_party.isin(["Democrat","Democratic","Republican"])].copy()
    raw["legacy_R"]=raw.legacy_party.eq("Republican").astype(int)
    bridge=[]
    for label,s in [("pooled",raw),("manufacturing",raw[raw.Segment=="Manufacturing"])]:
        fit=smf.ols("np.log1p(capex) ~ legacy_R + C(state) + C(quarter) + C(Technology)",s).fit(
            cov_type="cluster",cov_kwds={"groups":s.CD119_2024_Name},use_t=True)
        bridge.append(dict(sample=label,n=len(s),estimate=fit.params.legacy_R,
            low=fit.conf_int().loc["legacy_R",0],high=fit.conf_int().loc["legacy_R",1]))
    pd.DataFrame(bridge).to_csv(out/"sector_bridge.csv",index=False)


def figures(d,sens,primary,out):
    fig,axes=plt.subplots(1,2,figsize=(9,3.6),sharex=True)
    bins=[]
    for ax,(outcome,title) in zip(axes,[("pre_log_capex","Before the election · Jan–Sep"),("log_capex","Following term")]):
        h=primary["h_left"]
        local=d[d.running.abs()<h].copy()
        for sign,color in [(-1,"#355d77"),(1,"#b05844")]:
            sub=local[local.running*sign>0].copy()
            sub["bin"]=pd.cut(sub.running.abs(),np.linspace(0,h,7),include_lowest=True)
            g=sub.groupby("bin",observed=True).agg(margin=("running","mean"),y=(outcome,"mean"),n=(outcome,"size"))
            bins.append(g.assign(outcome=outcome,side=sign))
            ax.scatter(100*g.margin,g.y,s=18+g.n,color=color)
            co=np.polyfit(sub.running,sub[outcome],1,w=np.sqrt(1-sub.running.abs()/h))
            x=np.array([0,h*sign]);ax.plot(100*x,np.polyval(co,x),color=color)
        ax.axvline(0,color=".6",ls=":");ax.set(title=title,xlabel="Republican minus Democratic margin (pp)")
    axes[0].set_ylabel("Mean log(1 + documented $ millions)")
    fig.suptitle("Manufacturing announcements near tied House elections")
    fig.tight_layout();fig.savefig(out/"investment_cutoff.png",dpi=180);plt.close(fig)
    pd.concat(bins).to_csv(out/"cutoff_plot_bins.csv",index=False)
    fig,ax=plt.subplots(figsize=(7,3.8))
    ax.errorbar(sens.estimate,range(len(sens)),xerr=np.array([sens.estimate-sens.low,sens.high-sens.estimate]),
        fmt="o",color="#355d77",capsize=3)
    ax.set_yticks(range(len(sens)),sens.spec);ax.invert_yaxis()
    ax.axvline(0,color=".55",ls=":");ax.set_xlabel("Republican victory effect on log(1 + documented $ millions)")
    ax.set_title("Close-election estimates with 95% intervals")
    fig.tight_layout();fig.savefig(out/"sensitivity.png",dpi=180);plt.close(fig)


def analyze(out):
    paths=[DATA/"manufacturing_energy_and_industry_facility_metadata.csv",DATA/"MIT_election_lab/1976-2024-house.tab",
        DATA/"posteda/acs_data.csv",HERE/"nc_117.zip",*sorted((DATA/"redesign_raw/boundaries").glob("*.zip")),
        *sorted((DATA/"redesign_raw/legislators").glob("*.json")),
        *[DATA/f"election_stats/{y}election.pdf" for y in [2018,2020,2022]],
        DATA/"posteda/X_trim.csv",DATA/"results/ate_base.csv",DATA/"results/perm_test.csv"]
    hashes=[hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
    assert hashes[0]=="9ef8d6ae43465d8097998e55293055e96c3a6e84a6dbd4c397c45d8b3c4add8d"
    pd.DataFrame({"file":[p.relative_to(DATA.parent) for p in paths],"sha256":hashes}).to_csv(out/"inputs.csv",index=False)
    pd.Series({k:version(k) for k in ["pandas","numpy","geopandas","rdrobust","rddensity","statsmodels","matplotlib"]},
        name="version").to_csv(out/"environment.csv")
    f,d=build(out);associations(f,out)
    main=pd.DataFrame([rd(d,y,scale=100 if y=="any_projects" else 1)
        for y in ["log_capex","dollars","projects","any_projects"]])
    main.to_csv(out/"main_rd.csv",index=False)
    primary=main.iloc[0].to_dict()
    effective=d[d.running.abs()<primary["h_left"]].copy()
    term_records=terms()
    effective["member_party_jan3"]=[lookup(term_records,r.state,r.district,pd.Timestamp(r.year+1,1,3)) for r in effective.itertuples()]
    mapped=effective.member_party_jan3.map({"Democrat":"D","Republican":"R"})
    assert mapped.dropna().eq(effective.loc[mapped.notna(),"winner"]).all()
    effective["member_crosscheck"]=np.where(mapped.isna(),"vacancy/unseated on Jan 3","agrees")
    effective[["year","state","district","D_name","R_name","D_votes","R_votes","running","winner","audit",
        "member_party_jan3","member_crosscheck"]].to_csv(out/"effective_winner_audit.csv",index=False)
    old=pd.read_csv(DATA/"posteda/X_trim.csv",usecols=["Segment_Manufacturing","Segment_Energy and Industry"])
    legacy=pd.concat([pd.read_csv(DATA/"results/ate_base.csv"),pd.read_csv(DATA/"results/perm_test.csv")],axis=1)
    legacy.assign(n=len(old),manufacturing=old.Segment_Manufacturing.sum(),
        energy_industry=old["Segment_Energy and Industry"].sum()).to_csv(out/"legacy_trace.csv",index=False)
    sens=pd.DataFrame([primary,*[rd(d,"log_capex",f"±{h*100:g} pp margin",h) for h in [.05,.10]],
        *[rd(d[d.year!=year],"log_capex",f"Omit {year}") for year in [2018,2020,2022]],
        rd(d,"log_minus_largest","Omit largest announcement")])
    sens.to_csv(out/"sensitivity.csv",index=False)
    # Same primary estimation & bias windows for predetermined variables and disclosure.
    checks=[rd(d,y,"primary estimation window",primary["h_left"],scale,primary["b_left"]) for y,scale in [
        ("pre_log_capex",1),("pre_count",1),("incumbent_R",100),("med_household_income",1),
        ("unemployed",1),("total_pop",1),("gini",1),("disclosure",100),("any_unknown",100)]]
    pd.DataFrame(checks).to_csv(out/"diagnostics.csv",index=False)
    den=rddensity(X=d.running)
    est,se=den.hat["diff"],den.sd_jk["diff"]
    pd.DataFrame([dict(estimate=est,se=se,low=est-1.96*se,high=est+1.96*se,p=den.test.p_jk,
        h_left=den.h.left,h_right=den.h.right)]).to_csv(out/"density.csv",index=False)
    policy=d[d.year==2022]; rows=[];counts=[]
    for category in ["clear","broad"]:
        for label,h in [("MSE",None),("±10 pp margin",.1)]:
            row=rd(policy,"any_"+category,label,h,100)
            for side,sign,width in [("D",-1,row["h_left"]),("R",1,row["h_right"])]:
                local=policy[(policy.running*sign>0)&(policy.running.abs()<width)]
                counts.append(dict(definition=category,window=label,side=side,n=len(local),states=local.state.nunique(),
                    event_districts=(local[category]>0).sum(),projects=local[category].sum(),
                    rate_pct=100*(local[category]>0).mean(),bandwidth=width))
            row["sparse"]=min(c["event_districts"] for c in counts[-2:])<5
            rows.append(row)
    pd.DataFrame(rows).to_csv(out/"policy_rd.csv",index=False)
    pd.DataFrame(counts).to_csv(out/"policy_counts.csv",index=False)
    figures(d,sens,primary,out)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,default=HERE/"results")
    out=parser.parse_args().out.resolve()
    out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists(): raise SystemExit("Choose a fresh --out path. Existing outputs are preserved.")
    with tempfile.TemporaryDirectory(dir=out.parent) as temp:
        stage=Path(temp)/"result";stage.mkdir()
        analyze(stage)
        verify(stage)
        stage.rename(out)
    print(f"Verified and wrote results to {out}")
