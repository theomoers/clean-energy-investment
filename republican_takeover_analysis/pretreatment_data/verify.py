"""Independent numerical/source checks, and fresh-run comparisons."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
import pandas as pd
from scipy import stats
HERE=Path(__file__).resolve().parent

def main():
 p=argparse.ArgumentParser();p.add_argument('results',type=Path);p.add_argument('--compare',type=Path)
 p.add_argument('--skip-raw',action='store_true',help='Verify derived analysis only; raw-download and source-geography hashes are not checked.')
 p.add_argument('--no-write',action='store_true',help='Run checks without changing saved verification reports or hash ledgers.')
 a=p.parse_args()
 checks=[]
 def check(name,value):
  assert bool(value),name
  checks.append(name)
 manifest=json.loads((HERE/'download_manifest.json').read_text())
 if not a.skip_raw:
  for r in manifest:
   check('source_hash_'+r['file'],hashlib.sha256((HERE/'raw'/r['file']).read_bytes()).hexdigest()==r['sha256'])
 c=pd.read_csv(HERE/'covariates.csv');features=json.loads((HERE/'features.json').read_text())
 check('196_unique_core_covariates',len(c)==196 and c.unit.is_unique)
 check('no_missing_new_features',np.isfinite(c[features]).all().all())
 for col in ['poverty_share','college_share','unemployment_share','ira_coal_area_share','ira_ffe_area_share','area_within25km_hv_node']:
  check('valid_rate_'+col,c[col].between(0,1.00001).all())
 check('acs_geographic_coverage',c.acs_area_coverage.gt(.999).all())
 for name in ['solar_kwh_m2_day','wind50m_ms']:
  check('complete_resource_grid_'+name,c[name+'_coverage'].between(.99999,1.00001).all())
 r=pd.read_csv(a.results/'comparison.csv')
 sources=list(HERE.glob('*.py'))+list(HERE.glob('*.R'))+[HERE/'PROTOCOL.txt',HERE/'covariates.csv',HERE/'features.json',HERE/'download_manifest.json',
  HERE.parent/'overlap_forest/results_current/areas.csv',HERE.parent/'heterogeneity_upgrade/results_current/priorities.csv',
  HERE.parent/'heterogeneity_upgrade/run.py',HERE.parent/'heterogeneity_upgrade/design.py',HERE.parent/'extensions/common.py',
  HERE.parents[1]/'energy_community_audit/inputs/tracts.zip',
  HERE.parents[1]/'energy_community_audit/results/dated_closures.csv']
 if a.skip_raw:sources=[s for s in sources if 'energy_community_audit' not in s.parts]
 if not a.no_write:
  hashes=[dict(file=str(s.relative_to(HERE.parents[1])),sha256=hashlib.sha256(s.read_bytes()).hexdigest()) for s in sorted(sources)]
  pd.DataFrame(hashes).to_csv(a.results/'input_hashes.csv',index=False)
 for model in ['baseline','expanded']:
  d=pd.read_csv(a.results/model/'areas.csv');t=d.gainer.to_numpy();w=d.weight.to_numpy()
  check(model+'_fixed_sample',len(d)==196 and d.unit.is_unique and d.fold.eq('test').sum()==101)
  check(model+'_state_disjoint',not set(d.loc[d.fold.eq('train'),'state'])&set(d.loc[d.fold.eq('test'),'state']))
  check(model+'_positive_weights',(w>0).all())
  check(model+'_overlap_formula',np.allclose(w,np.where(t==1,1-d.propensity,d.propensity)))
  rr=r[r.model.eq(model)]
  check(model+'_interval_calculation',np.allclose(rr.hi-rr.estimate,stats.t.ppf(.975,9)*rr.se))
  check(model+'_inference_draw_coverage',rr.successful_draws.ge(.95*rr.successful_draws.max()).all())
  for rank in ['old']+(['expanded'] if model=='expanded' else []):
   designs=pd.read_csv(a.results/model/rank/'design_frontier.csv')
   check(model+'_'+rank+'_all_candidates',len(designs)==42)
  m=pd.read_csv(a.results/model/'outcome_predictions.csv')
  for o in rr.outcome.unique():
   for v in ['delta','placebo','trend_adjusted']:
    pred=m[m.outcome.eq(o)&m.version.eq(v)].set_index('unit').reindex(d.unit)
    idx=d.fold.eq('test').to_numpy();e=d.propensity.to_numpy()[idx];tt=t[idx]
    y=d.loc[idx,o+'_'+v].to_numpy();m0=pred.m0.to_numpy()[idx];m1=pred.m1.to_numpy()[idx]
    gamma=m1-m0+tt/e*(y-m1)-(1-tt)/(1-e)*(y-m0)
    avg=np.average(gamma,weights=e*(1-e))
    point=rr[rr.outcome.eq(o)&rr.version.eq(v)&rr.statistic.eq('average')].estimate.item()
    check(model+'_'+o+'_'+v+'_independent_AIPW',np.isclose(avg,point,rtol=1e-8,atol=1e-8))
 if a.compare:
  # Compare every generated table; text session metadata may differ by path.
  tables=list(a.results.rglob('*.csv'))
  for path in tables:
   rel=path.relative_to(a.results)
   if rel.name=='input_hashes.csv':continue # Code/path provenance changes do not constitute numerical nonreplication.
   other=a.compare/rel
   check('replication_exists_'+str(rel),other.exists())
   left=pd.read_csv(path);right=pd.read_csv(other)
   pd.testing.assert_frame_equal(left,right,check_exact=False,rtol=1e-9,atol=1e-9)
   checks.append('replication_values_'+str(rel))
 result=dict(passed=len(checks),checks=checks,manuscript_changed=False,raw_sources_checked=not a.skip_raw)
 if not a.no_write:(a.results/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
 print('Passed',len(checks),'checks')

if __name__=='__main__':main()
