"""Prediction, support and full-panel pretrend comparisons. No spec selection."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
from scipy import stats
from analyze import propensity,BASE,MAIN,HERE,SOURCE,ess

def contrast(y,t,w):
 return np.average(y[t==1],axis=0,weights=w[t==1])-np.average(y[t==0],axis=0,weights=w[t==0])

def main():
 p=argparse.ArgumentParser();p.add_argument('results',type=Path);p.add_argument('--bootstrap-reps',type=int,default=999);a=p.parse_args()
 out=a.results;support=[];prediction=[];trends=[];joints=[];groups=[]
 annual=pd.read_csv(HERE.parent/'extensions/results_current/annual2023.csv')
 for model in ['baseline','expanded']:
  path=out/model;u=pd.read_csv(path/'areas.csv').copy();t=u.gainer.to_numpy();w=u.weight.to_numpy()
  features=BASE+(json.loads((HERE/'features.json').read_text()) if model=='expanded' else [])
  e=u.propensity.to_numpy();te=u.fold.eq('test').to_numpy()
  for sample,m in [('full',np.ones(len(u),bool)),('train',~te),('test',te)]:
   state_mass=u.loc[m].assign(h=e[m]*(1-e[m])).groupby('state').h.sum().to_numpy()
   support.append(dict(model=model,sample=sample,areas=int(m.sum()),states=u.loc[m,'state'].nunique(),
     ess_R=ess(w[m&(t==1)]),ess_D=ess(w[m&(t==0)]),effective_states=ess(state_mass),
     extreme_propensities_01=int(((e[m]<.01)|(e[m]>.99)).sum()),
     largest_arm_weight_R=float((w[m&(t==1)]/w[m&(t==1)].sum()).max()),
     largest_arm_weight_D=float((w[m&(t==0)]/w[m&(t==0)].sum()).max())))
  pred=pd.read_csv(path/'outcome_predictions.csv')
  for o in MAIN:
   d=pred[pred.outcome.eq(o)&pred.version.eq('delta')].set_index('unit').reindex(u.unit)
   m=np.where(t==1,d.m1,d.m0);y=u[o+'_delta'].to_numpy()
   prediction.append(dict(model=model,outcome=o,heldout_mse=float(np.mean((y[te]-m[te])**2)),
       heldout_overlap_mse=float(np.average((y[te]-m[te])**2,weights=e[te]*(1-e[te]))),
       heldout_mae=float(np.mean(abs(y[te]-m[te])))))
  for rank in ['old']+(['expanded'] if model=='expanded' else []):
   f=pd.read_csv(path/rank/'design_frontier.csv')
   for (o,definition),g in f.groupby(['outcome','definition']):
    # Report the same fixed illustrative candidate in every model, never best p.
    d=g[g.candidate.eq('pooled_ridge_0.001')].iloc[0].to_dict()
    d.update(model=model,ranking=rank,passing_candidates=int(g.passes_design.sum()))
    groups.append(d)
  arrays=[];metadata=[]
  for o in MAIN:
   yy=annual[annual.outcome.eq(o)].pivot(index='unit',columns='year',values='y').reindex(u.unit)
   assert yy[[2019,2020,2021,2022]].notna().all().all()
   for year in [2019,2020,2021]:
    arrays.append((yy[year]-yy[2022]).to_numpy());metadata.append((o,year))
  y=np.column_stack(arrays);point=contrast(y,t,w)
  rng=np.random.default_rng(11481);states=sorted(u.state.unique());draws=[]
  penalty=json.loads((path/'propensity_rule.json').read_text())['penalty']
  for b in range(a.bootstrap_reps):
   count=pd.Series(rng.choice(states,len(states),replace=True)).value_counts()
   freq=u.state.map(count).fillna(0).to_numpy();ee,g,_=propensity(u,features,freq,penalty)
   ww=freq*np.where(t==1,1-ee,ee)
   if g<1e-6:draws.append(contrast(y,t,ww))
  boot=np.array(draws);assert len(boot)>=a.bootstrap_reps*.95
  for j,(o,year) in enumerate(metadata):
   se=boot[:,j].std(ddof=1);c=stats.t.ppf(.975,24)
   trends.append(dict(model=model,outcome=o,year=year,reference_year=2022,estimate=point[j],
                      se=se,lo=point[j]-c*se,hi=point[j]+c*se,p=2*stats.t.sf(abs(point[j]/se),24)))
  for o in MAIN:
   ix=[j for j,(oo,_) in enumerate(metadata) if o==oo];v=point[ix];cov=np.cov(boot[:,ix],rowvar=False)
   rank=np.linalg.matrix_rank(cov);F=v@np.linalg.pinv(cov)@v/rank
   joints.append(dict(model=model,outcome=o,F=F,restrictions=rank,df_states=24,p=stats.f.sf(F,rank,24),
                       successful_draws=len(boot)))
 pd.DataFrame(support).to_csv(out/'support_comparison.csv',index=False)
 pd.DataFrame(prediction).to_csv(out/'prediction_metrics.csv',index=False)
 pd.DataFrame(groups).to_csv(out/'subgroup_comparison.csv',index=False)
 pd.DataFrame(trends).to_csv(out/'pretrend_estimates.csv',index=False)
 pd.DataFrame(joints).to_csv(out/'pretrend_joint_tests.csv',index=False)
 print(pd.DataFrame(support).to_string(index=False),flush=True)
 print(pd.DataFrame(prediction).to_string(index=False),flush=True)
 print(pd.DataFrame(joints).to_string(index=False),flush=True)

if __name__=='__main__':main()
