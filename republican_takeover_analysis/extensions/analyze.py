"""Transparent estimates, overlap-weight uncertainty, multiplicity and power."""
import json
import numpy as np
import pandas as pd
from scipy import stats
from common import OLD,MAIN,BASE,EXTRA,mean_ci,contrast,overlap_weights,holm,power_equivalence
from prepare2023 import CATEGORIES


def pair_columns(pairs,units,outcomes):
    p=pairs.copy();u=units.set_index('unit')
    additions={}
    for o in outcomes:
        for v in ['pre','post','delta','placebo','trend_adjusted','exclude_last_pre']:
            c=o+'_'+v;additions[c+'_R']=p.R.map(u[c]);additions[c+'_D']=p.D.map(u[c]);additions[c]=additions[c+'_R']-additions[c+'_D']
    return pd.concat([p.drop(columns=[c for c in additions if c in p]),pd.DataFrame(additions)],axis=1)


def balance(units,pairs,columns,config,weights=None):
    sd=units[columns].std().replace(0,1);t=units.gainer.to_numpy()
    if weights is None:weights=units.unit.isin(set(pairs.R)|set(pairs.D)).to_numpy(float)
    rows=[]
    for c in columns:
        before=units.groupby('gainer')[c].mean()
        x=units[c].to_numpy();grid=np.unique(x)
        variances=[];cdfs=[]
        for arm in [0,1]:
            values=x[t==arm];w=weights[t==arm]
            mu=np.average(values,weights=w)
            variances.append(np.average((values-mu)**2,weights=w))
            cdfs.append(np.array([w[values<=q].sum()/w.sum() for q in grid]))
        rows.append(dict(config=config,feature=c,before_smd=(before[1]-before[0])/sd[c],
            after_smd=contrast(units[c],t,weights)/sd[c],
            after_variance_ratio=variances[1]/variances[0] if variances[0]>0 else np.nan,
            after_weighted_ks_distance=np.max(abs(cdfs[1]-cdfs[0]))))
    return rows


def main(out,units,pairs,annual,u11,p11,a11,features11,bootstrap_reps=999):
    original=pd.read_csv(OLD/'pairs.csv');original=original[original.config.eq('primary')].copy()
    original['config']='original2023'
    original=pair_columns(original,units,MAIN+CATEGORIES)
    configs=[(original,units,annual,MAIN+CATEGORIES), (pairs,units,annual,MAIN+CATEGORIES)]
    for config,g in p11.groupby('config'):
        uu=u11[u11.config.eq(config)];configs.append((g,uu,a11,['wind_solar_net_mw']))
    estimates=[];balances=[];baseline={};states=[];events=[];support=[];joint=[]
    for pp,uu,aa,outcomes in configs:
        config=pp.config.iloc[0];columns=features11 if config.startswith('audit2011') else BASE+EXTRA
        balances.extend(balance(uu,pp,columns,config))
        selected=set(pp.R)|set(pp.D)
        for o in outcomes:
            baseline[(config,o)]=np.mean(np.r_[pp[o+'_pre_R'],pp[o+'_pre_D']])
            for version in ['delta','trend_adjusted','placebo','exclude_last_pre']:
                estimates.append(dict(config=config,outcome=o,version=version,**mean_ci(pp[o+'_'+version],pp.state)))
            for st,g in pp.groupby('state'):
                states.append(dict(config=config,outcome=o,state=st,pairs=len(g),
                    estimate=g[o+'_delta'].mean(),pre_R=g[o+'_pre_R'].mean(),pre_D=g[o+'_pre_D'].mean(),
                    post_R=g[o+'_post_R'].mean(),post_D=g[o+'_post_D'].mean(),
                    inferential_status='Descriptive only: one state cluster; no state-level causal significance claim'))
            support.append(dict(config=config,outcome=o,pairs=len(pp),states=pp.state.nunique(),
                post_total_R=2*pp[o+'_post_R'].sum(),post_total_D=2*pp[o+'_post_D'].sum(),
                post_positive_R=int(pp[o+'_post_R'].gt(0).sum()),post_positive_D=int(pp[o+'_post_D'].gt(0).sum()),
                baseline=baseline[(config,o)],support_warning=(
                    'No observed post-period events; no supported category-specific causal claim'
                    if o in CATEGORIES and (pp[o+'_post_R'].sum()+pp[o+'_post_D'].sum())==0 else
                    'Fewer than 20 post-period events; sparse descriptive support'
                    if o in CATEGORIES and 2*(pp[o+'_post_R'].sum()+pp[o+'_post_D'].sum())<20 else '')))
            av=aa[aa.outcome.eq(o)&aa.unit.isin(selected)]
            wide=av.pivot(index='unit',columns='year',values='y')
            ref=2010 if config.startswith('audit2011') else 2022
            for year in wide.columns:
                vals=wide[year]-wide[ref]
                events.append(dict(config=config,outcome=o,year=year,
                    **mean_ci(pp.R.map(vals)-pp.D.map(vals),pp.state)))
            preyears=[y for y in wide.columns if y<ref]
            differences=np.column_stack([pp.R.map(wide[y]-wide[ref])-pp.D.map(wide[y]-wide[ref]) for y in preyears])
            b=differences.mean(0);g=pp.state.nunique();n=len(pp)
            scores=pd.DataFrame(differences-b).groupby(pp.state.to_numpy()).sum().to_numpy()
            cov=scores.T@scores/n**2*g/(g-1);rank=np.linalg.matrix_rank(cov)
            f=float(b@np.linalg.pinv(cov)@b/rank) if rank else np.nan
            joint.append(dict(config=config,outcome=o,states=g,restrictions=rank,F=f,p=stats.f.sf(f,rank,g-1) if rank else np.nan,
                warning='Matching uses early pre-outcomes; nonrejection is not independent evidence of parallel trends'))
    # Logistic overlap weighting uses all eligible areas but changes the target population.
    weights,prop,score=overlap_weights(units,BASE+EXTRA)
    assert score<1e-5,score
    t=units.gainer.to_numpy();config='overlap2023'
    balances.extend(balance(units,None,BASE+EXTRA,config,weights))
    us=units[['unit','state','district','gainer']].copy();us['propensity']=prop;us['weight']=weights
    us.to_csv(out/'overlap_weights.csv',index=False)
    cols=[o+'_'+v for o in MAIN+CATEGORIES for v in ['delta','trend_adjusted','placebo','exclude_last_pre']]
    y=units[cols].to_numpy()
    def all_contrasts(w):return np.average(y[t==1],axis=0,weights=w[t==1])-np.average(y[t==0],axis=0,weights=w[t==0])
    point=all_contrasts(weights)
    rng=np.random.default_rng(11481);st=sorted(units.state.unique());G=len(st)
    boots=[];fail=0
    for b in range(bootstrap_reps):
        draws=rng.choice(st,len(st),replace=True);counts=pd.Series(draws).value_counts()
        frequency=units.state.map(counts).fillna(0).to_numpy()
        w,_,sc=overlap_weights(units,BASE+EXTRA,frequency)
        if sc>1e-5 or min(w[t==0].sum(),w[t==1].sum())<1e-8:
            fail+=1;continue
        boots.append(all_contrasts(w))
        if (b+1)%250==0:print('Overlap state bootstrap',b+1,flush=True)
    assert len(boots)>=.95*bootstrap_reps,(len(boots),bootstrap_reps)
    boot=np.array(boots);se=boot.std(axis=0,ddof=1);crit=stats.t.ppf(.975,G-1)
    pd.DataFrame(boot,columns=cols).to_csv(out/'overlap_bootstrap_draws.csv',index=False)
    summaries=[]
    for arm in [0,1]:
        w=weights[t==arm];s=units.loc[t==arm,'state']
        sw=pd.Series(w).groupby(np.asarray(s)).sum()
        summaries.append(dict(arm=arm,areas=len(w),kish_area_ess=w.sum()**2/(w@w),
            weighted_state_ess=sw.sum()**2/(sw@sw),max_normalized_weight=w.max()/w.sum(),
            min_propensity=prop[t==arm].min(),max_propensity=prop[t==arm].max(),score_residual=score,
            bootstrap_success=len(boots),bootstrap_failures=fail,bootstrap_requested=bootstrap_reps))
    pd.DataFrame(summaries).to_csv(out/'overlap_support.csv',index=False)
    for i,col in enumerate(cols):
        o,v=next((o,v) for o in MAIN+CATEGORIES for v in ['delta','trend_adjusted','placebo','exclude_last_pre'] if col==o+'_'+v)
        estimates.append(dict(config=config,outcome=o,version=v,estimate=point[i],se=se[i],lo=point[i]-crit*se[i],hi=point[i]+crit*se[i],
            p=2*stats.t.sf(abs(point[i]/se[i]),G-1) if se[i]>0 else np.nan,states=G,n=len(units),
            percentile_lo=np.quantile(boot[:,i],.025),percentile_hi=np.quantile(boot[:,i],.975)))
        baseline[(config,o)]=(np.average(units.loc[t==1,o+'_pre'],weights=weights[t==1])+
                               np.average(units.loc[t==0,o+'_pre'],weights=weights[t==0]))/2
    est=pd.DataFrame(estimates)
    est['family']=np.where(est.outcome.isin(MAIN),'primary_outcomes','technology_categories')
    est['p_holm']=est.groupby(['config','version','family']).p.transform(holm)
    est['p_holm_all_primary_sensitivities']=np.nan
    ix=est.family.eq('primary_outcomes')&est.version.isin(['delta','trend_adjusted','exclude_last_pre'])
    est.loc[ix,'p_holm_all_primary_sensitivities']=holm(est.loc[ix,'p'])
    est.to_csv(out/'estimates.csv',index=False)
    pd.DataFrame(balances).to_csv(out/'balance.csv',index=False)
    pd.DataFrame(states).to_csv(out/'state_descriptive.csv',index=False)
    pd.DataFrame(events).to_csv(out/'matched_event_study.csv',index=False)
    pd.DataFrame(support).to_csv(out/'matched_support.csv',index=False)
    pd.DataFrame(joint).to_csv(out/'matched_joint_pretests.csv',index=False)
    eq=power_equivalence(est[est.outcome.isin(MAIN)&est.version.isin(['delta','trend_adjusted'])],baseline,out)
    pd.DataFrame([dict(config=k[0],outcome=k[1],baseline=v) for k,v in baseline.items()]).to_csv(out/'baselines.csv',index=False)
    # Six-category heterogeneity: equal effects across categories, not differing p-values.
    gaps=[]
    for pp,_,__,outcomes in configs[:2]:
        for i,a in enumerate(CATEGORIES):
            for b in CATEGORIES[i+1:]:
                gaps.append(dict(config=pp.config.iloc[0],a=a,b=b,**mean_ci(pp[a+'_delta']-pp[b+'_delta'],pp.state)))
    g=pd.DataFrame(gaps);g['p_holm']=g.groupby('config').p.transform(holm);g.to_csv(out/'category_effect_differences.csv',index=False)
    allpairs=pd.concat([pairs,p11],ignore_index=True);allpairs.to_csv(out/'forest_pairs.csv',index=False)
    with open(out/'forest_configs.json','w') as f:
        json.dump({**{'balanced2023':dict(outcomes=MAIN,features=['log_clean_capacity','log_fossil_capacity','log_prior_announcements','log_core_area'])},
                   **{c:dict(outcomes=['wind_solar_net_mw'],features=features11) for c in p11.config.unique()}},f,indent=2)
    return est,eq
