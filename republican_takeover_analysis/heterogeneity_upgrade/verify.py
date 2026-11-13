"""Independent output checks and deterministic fresh-run comparison."""
from pathlib import Path
import argparse,hashlib,json,tempfile
import numpy as np
import pandas as pd
from scipy import stats
from run import ROOT,SOURCE,FEATURES,MAIN,outcome_models

parser=argparse.ArgumentParser()
parser.add_argument("--current",type=Path,required=True)
parser.add_argument("--replication",type=Path,required=True)
args=parser.parse_args();a=args.current.resolve();b=args.replication.resolve()
checks=[]
def check(name,value):
    assert value,name
    checks.append(name)
def manifest(d):
    return {str(p.relative_to(d)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(d.rglob("*")) if p.is_file()}
ma,mb=manifest(a),manifest(b)
check("fresh_outputs_byte_identical",ma==mb)
check("source_CSV_preserved_byte_for_byte",(a/"areas.csv").read_bytes()==(SOURCE/"areas.csv").read_bytes())
u=pd.read_csv(a/"areas.csv");test=u.fold.eq("test").to_numpy()
check("correct_196_area_roster",len(u)==196 and u.unit.is_unique)
check("state_disjoint_training_evaluation",not set(u.loc[test,"state"])&set(u.loc[~test,"state"]))
check("evaluation_101_areas_10_states",test.sum()==101 and u.loc[test,"state"].nunique()==10)
check("training_95_areas_15_states",(~test).sum()==95 and u.loc[~test,"state"].nunique()==15)
prior=pd.read_csv(SOURCE/"forest_predictions.csv")
new=pd.read_csv(a/"priorities.csv")
x=prior[prior.fold.eq("test")].merge(new,on=["unit","outcome","fold"],validate="1:1")
check("original_rankings_reproduced",np.allclose(x.heldout_score,x.priority,atol=1e-10,rtol=1e-10))
for row in pd.read_csv(a/"input_code_hashes.csv").itertuples():
    check("source_hash_"+row.file,hashlib.sha256((ROOT/row.file).read_bytes()).hexdigest()==row.sha256)
z=((u[FEATURES]-u[FEATURES].mean())/u[FEATURES].std()).to_numpy()
perturbed=u.copy()
for o in MAIN:
    for v in ["delta","placebo","trend_adjusted"]:
        perturbed.loc[test,o+"_"+v]+=1000000.
with tempfile.TemporaryDirectory(prefix="heterogeneity-nuisance-audit-") as td:
    outcome_models(perturbed,z,Path(td))
    p1=pd.read_csv(a/"outcome_predictions.csv")
    p2=pd.read_csv(Path(td)/"outcome_predictions.csv")
    check("evaluation_outcome_perturbation_leaves_nuisance_predictions_unchanged",
          np.array_equal(p1[["m0","m1"]].to_numpy(),p2[["m0","m1"]].to_numpy()))
    check("evaluation_outcome_perturbation_leaves_training_CV_unchanged",
          (a/"nuisance_training_cv.csv").read_bytes()==(Path(td)/"nuisance_training_cv.csv").read_bytes())
r=pd.read_csv(a/"adjusted_heterogeneity.csv")
boot=pd.read_csv(a/"bootstrap_statistics.csv")
for row in r[r.statistic.eq("qini")].itertuples():
    key="|".join([row.outcome,row.version,row.statistic])
    values=boot[key].dropna()
    check("bootstrap_SE_"+key,np.isclose(values.std(ddof=1),row.se))
    check("t9_interval_"+key,np.isclose(row.lo,row.estimate-stats.t.ppf(.975,9)*row.se))
    check("enough_bootstrap_draws_"+key,len(values)>=950)
frontier=pd.read_csv(a/"design_frontier.csv")
check("all_42_design_candidates_reported",len(frontier)==42)
weights=pd.read_csv(a/"subgroup_weights.csv")
definitions=pd.read_csv(a/"group_definitions.csv")
for (o,definition,candidate),g in weights.groupby(["outcome","definition","candidate"]):
    check("all_101_areas_retained_"+o+"_"+definition+"_"+candidate,len(g)==101 and g.weight.gt(0).all())
    if candidate.startswith("calibration"):
        tol=float(candidate.split("_")[1])
        meta=definitions[definitions.outcome.eq(o)&definitions.definition.eq(definition)][["unit","high"]]
        gg=g.merge(meta,on="unit",validate="1:1").merge(u[["unit","gainer"]],on="unit",validate="1:1")
        for _,cell in gg.groupby(["high","gainer"]):
            w=cell.weight.to_numpy();n=len(w)
            check("calibration_mass_and_caps_"+o+"_"+definition+"_"+candidate,
                  np.isclose(w.sum(),1) and w.min()>=.01/n-1e-7 and w.max()<=min(3/n,.20)+1e-7)
        row=frontier[frontier.outcome.eq(o)&frontier.definition.eq(definition)&frontier.candidate.eq(candidate)].iloc[0]
        check("calibration_SMD_"+o+"_"+definition+"_"+candidate,row.max_smd<=tol+1e-6)
report=dict(date="2026-11-07",canonical_outputs=str(a.relative_to(ROOT)),
            fresh_rebuild=str(b.relative_to(ROOT)),files_identical_byte_for_byte=len(ma),
            manifest_digest_sha256=hashlib.sha256(json.dumps(ma,sort_keys=True).encode()).hexdigest(),
            checks_passed=len(checks),checks=checks,
            manuscript_changed=False,earlier_analyses_changed=False)
(Path(__file__).resolve().parent/"REPLICATION.json").write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps({k:v for k,v in report.items() if k!="checks"},indent=2))
