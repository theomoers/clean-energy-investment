"""Frozen-split comparison of additional pre-treatment controls."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,os,shutil,subprocess,sys
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/majority-mpl')
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from scipy.special import expit
from statsmodels.stats.multitest import multipletests
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE.parent/'heterogeneity_upgrade'))
spec=importlib.util.spec_from_file_location('upgrade',HERE.parent/'heterogeneity_upgrade/run.py')
up=importlib.util.module_from_spec(spec);spec.loader.exec_module(up)
from design import diagnostic,ess
BASE=list(up.FEATURES);MAIN=list(up.MAIN)
SOURCE=HERE.parent/'overlap_forest/results_current'

def propensity(u,features,frequency=None,penalty=0.):
 x=u[features].to_numpy(float);x=(x-x.mean(0))/x.std(0)
 state=pd.get_dummies(u.state,dtype=float).to_numpy();x=np.c_[state,x]
 t=u.gainer.to_numpy(float);f=np.ones(len(t)) if frequency is None else np.asarray(frequency)
 ridge=np.r_[np.zeros(state.shape[1]),np.full(len(features),penalty)]
 def obj(b):
  a=x@b;p=expit(a)
  return np.sum(f*(np.logaddexp(0,a)-t*a))/f.sum()+.5*np.sum(ridge*b*b),x.T@(f*(p-t))/f.sum()+ridge*b
 fit=minimize(obj,np.zeros(x.shape[1]),jac=True,method='BFGS',options={'gtol':1e-9,'maxiter':1000})
 p=expit(x@fit.x);score=float(max(abs(obj(fit.x)[1])))
 return p,score,float(max(abs(fit.x)))

def score_stats(u,idx,priorities,e,pred,frequency):
 t=u.gainer.to_numpy();rows={}
 for outcome in MAIN:
  for version in ['delta','placebo','trend_adjusted']:
   h,num=up.bounded_score(u.loc[idx,outcome+'_'+version].to_numpy(),t[idx],e[idx],pred[outcome,version][idx])
   h=h*frequency[idx];num=num*frequency[idx]
   rows[outcome,version,'average']=num.sum()/h.sum()
   for rank,f in priorities.items():
    rows[outcome,version,'qini_'+rank]=up.qini(f[outcome][idx],h,num,check=True)
 return rows

def run_model(u,features,model,out,reps):
 out.mkdir(parents=True,exist_ok=False);te=u.fold.eq('test').to_numpy();tr=~te
 audit=[]
 for sample,m in [('full',np.ones(len(u),bool)),('train',tr)]:
  for lam in [0.,.01]:
   p,g,b=propensity(u.loc[m],features,penalty=lam)
   audit.append(dict(model=model,sample=sample,penalty=lam,gradient=g,max_abs_coef=b,
      min_propensity=p.min(),max_propensity=p.max(),ess_R=ess((1-p)[u.loc[m,'gainer'].eq(1)]),
      ess_D=ess(p[u.loc[m,'gainer'].eq(0)]),extreme_01=int(((p<.01)|(p>.99)).sum())))
 pd.DataFrame(audit).to_csv(out/'propensity_audit.csv',index=False)
 # Rule is outcome-free, fixed before fitting any new forest/result:
 # any near-separation in full or training sample triggers same .01 penalty.
 mle=[r for r in audit if r['penalty']==0]
 lam=.01 if any(r['extreme_01']>0 or r['gradient']>1e-6 for r in mle) else 0.
 e,g,_=propensity(u,features,penalty=lam);pt,gt,_=propensity(u.loc[tr],features,penalty=lam)
 assert max(g,gt)<1e-6 and min(e)>0 and max(e)<1
 u=u.copy();u['propensity']=e;u['weight']=np.where(u.gainer.eq(1),1-e,e)
 u['training_propensity']=np.nan;u.loc[tr,'training_propensity']=pt
 u.to_csv(out/'areas.csv',index=False);(out/'features.txt').write_text('\n'.join(features)+'\n')
 (out/'propensity_rule.json').write_text(json.dumps(dict(penalty=lam,reason='outcome-free near-separation rule'),indent=2)+'\n')
 t=u.gainer.to_numpy();w=u.weight.to_numpy()
 full=[];z=((u[features]-u[features].mean())/u[features].std()).to_numpy()
 for j,col in enumerate(features):
  value=np.average(z[t==1,j],weights=w[t==1])-np.average(z[t==0,j],weights=w[t==0])
  full.append(dict(feature=col,smd=value))
 for s in sorted(u.state.unique()):
  v=u.state.eq(s).to_numpy(float);sd=v.std(ddof=1)
  full.append(dict(feature='state_'+s,smd=(np.average(v[t==1],weights=w[t==1])-np.average(v[t==0],weights=w[t==0]))/sd))
 pd.DataFrame(full).to_csv(out/'overall_balance.csv',index=False)
 # Audit all added controls even when baseline model does not adjust for them.
 allfeatures=BASE+json.loads((HERE/'features.json').read_text())
 allbal=[]
 for col in allfeatures:
  zz=(u[col]-u[col].mean())/u[col].std()
  allbal.append(dict(feature=col,smd=np.average(zz[t==1],weights=w[t==1])-np.average(zz[t==0],weights=w[t==0])))
 pd.DataFrame(allbal).to_csv(out/'all_control_balance.csv',index=False)
 if model=='expanded':
  rscript=os.environ.get('RSCRIPT') or shutil.which('Rscript')
  if not rscript:raise RuntimeError('Install R and grf, or set RSCRIPT to the Rscript executable.')
  subprocess.run([rscript,'--vanilla',str(HERE/'forest.R'),str(out)],check=True)
  new=pd.read_csv(out/'priorities.csv')
 else:
  new=pd.read_csv(HERE.parent/'heterogeneity_upgrade/results_current/priorities.csv')
  new.to_csv(out/'priorities.csv',index=False)
 old=pd.read_csv(HERE.parent/'heterogeneity_upgrade/results_current/priorities.csv')
 priorities={rank:{o:d[d.outcome.eq(o)].set_index('unit').loc[u.unit,'priority'].to_numpy() for o in MAIN}
             for rank,d in [('old',old),('expanded',new)]}
 if model=='baseline':priorities={'old':priorities['old']}
 # Test outcome-free subgroup balancing under both independent rankings.
 up.FEATURES=features
 groups={};frozen={}
 for rank,ranking in priorities.items():
  sub=out/rank;sub.mkdir()
  gr={}
  for o,f in ranking.items():
   cutoff=up.weighted_median(f[tr],pt*(1-pt));half=up.weighted_median(f[te],e[te]*(1-e[te]))
   gr[o]={'original':f>cutoff,'rank_half':f>half}
  up.design_audit(u,z,ranking,gr,sub)
  groups[rank]=gr
 # Auxiliary regressions use only the training states, including tuning.
 pred=up.outcome_models(u,z,out)
 idx=np.flatnonzero(te);point=score_stats(u,idx,priorities,e,pred,np.ones(len(u)))
 rng=np.random.default_rng(22481);states=sorted(u.loc[te,'state'].unique());draws={k:[] for k in point}
 bad=0
 for b in range(reps):
  count=pd.Series(rng.choice(states,len(states),replace=True)).value_counts()
  frequency=np.where(te,u.state.map(count).fillna(0),1).astype(float)
  eb,grad,_=propensity(u,features,frequency,lam)
  if grad>1e-6 or eb.min()<=0 or eb.max()>=1:
   bad+=1;ss={}
  else:ss=score_stats(u,idx,priorities,eb,pred,frequency)
  for k in draws:draws[k].append(ss.get(k,np.nan))
  if (b+1)%250==0:print(model,'bootstrap',b+1,flush=True)
 assert bad<.05*reps,(model,bad,reps)
 result=[]
 for (o,v,s),value in point.items():
  b=np.array(draws[o,v,s]);b=b[np.isfinite(b)];se=b.std(ddof=1);crit=stats.t.ppf(.975,len(states)-1)
  result.append(dict(model=model,outcome=o,version=v,statistic=s,estimate=value,se=se,
                    lo=value-crit*se,hi=value+crit*se,p=2*stats.t.sf(abs(value/se),len(states)-1),
                    states=len(states),successful_draws=len(b),penalty=lam))
 r=pd.DataFrame(result);r.to_csv(out/'results.csv',index=False)
 pd.DataFrame({'|'.join(k):pd.Series(v) for k,v in draws.items()}).to_csv(out/'bootstrap.csv',index=False)
 return r

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--bootstrap-reps',type=int,default=999);a=p.parse_args()
 out=a.out.resolve();out.mkdir(parents=True,exist_ok=False)
 u=pd.read_csv(SOURCE/'areas.csv');d=pd.read_csv(HERE/'covariates.csv')
 u=u.merge(d,on='unit',validate='1:1',how='left');assert len(u)==196 and u.fold.eq('test').sum()==101
 extra=json.loads((HERE/'features.json').read_text());assert np.isfinite(u[BASE+extra]).all().all()
 results=[]
 for model,features in [('baseline',BASE),('expanded',BASE+extra)]:
  results.append(run_model(u,features,model,out/model,a.bootstrap_reps))
 r=pd.concat(results,ignore_index=True)
 ix=r.model.eq('expanded')&r.version.eq('delta')&r.statistic.str.startswith('qini')
 r.loc[ix,'p_holm_six_rankings']=multipletests(r.loc[ix,'p'],method='holm')[1]
 r.to_csv(out/'comparison.csv',index=False)
 # Compare actual heldout outcome prediction without using it to choose spec.
 ms=[]
 for model in ['baseline','expanded']:
  n=pd.read_csv(out/model/'nuisance_models.csv');n['model']=model;ms.append(n)
 pd.concat(ms).to_csv(out/'prediction_comparison.csv',index=False)
 hashes=[]
 sources=list(HERE.glob('*.py'))+list(HERE.glob('*.R'))+[HERE/'PROTOCOL.txt',HERE/'covariates.csv',SOURCE/'areas.csv']
 for path in sources:hashes.append(dict(file=str(path.relative_to(ROOT)),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
 pd.DataFrame(hashes).to_csv(out/'input_hashes.csv',index=False)
 print(r[r.version.eq('delta')].to_string(index=False),flush=True)

if __name__=='__main__':main()
