"""Reconcile matched forest results, diagnostics, sensitivity, and figures."""
from pathlib import Path
import sys
import json
import hashlib
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUTCOMES=['clean_announcements','manufacturing_announcements','wind_solar_net_mw']
LABELS=['Clean-power announcements','Manufacturing announcements','Net wind/solar capacity (MW)']


def fit(y,state,treatment=None):
    y=np.asarray(y,float);state=np.asarray(state)
    x=np.ones((len(y),1))
    if treatment is not None:
        x=np.column_stack([x,treatment,pd.get_dummies(state,drop_first=True,dtype=float)])
    f=sm.OLS(y,x).fit(cov_type='cluster',cov_kwds={'groups':state},use_t=True)
    k=0 if treatment is None else 1;g=len(set(state));b=f.params[k];se=f.bse[k]
    return dict(estimate=b,se=se,lo=b-stats.t.ppf(.975,g-1)*se,hi=b+stats.t.ppf(.975,g-1)*se,p=f.pvalues[k],states=g,n=len(y))


def summarize(out):
    pairs=pd.read_csv(out/'pairs.csv');units=pd.read_csv(out/'units.csv');annual=pd.read_csv(out/'annual.csv')
    averages=pd.read_csv(out/'paired_estimates.csv');pred=pd.read_csv(out/'forest_predictions.csv')
    ledger=pd.read_csv(out/'matching_ledger.csv');folds=pd.read_csv(out/'state_folds.csv')
    assert not units.duplicated(['config','unit']).any()
    assert not pairs.duplicated('pair').any()
    supports=[];comparisons=[];events=[];joint=[];influence=[];calipers=[]
    for config,p in pairs.groupby('config'):
        u=units[units.config.eq(config)]
        assert p.R.is_unique and p.D.is_unique and not set(p.R)&set(p.D)
        for row in p.itertuples():
            r=u[u.unit.eq(row.R)].iloc[0];d=u[u.unit.eq(row.D)].iloc[0]
            assert r.gainer==1 and d.gainer==0 and r.state==d.state==row.state
            for outcome in OUTCOMES:
                assert np.isclose(getattr(row,outcome+'_delta'),r[outcome+'_delta']-d[outcome+'_delta'])
        fs=folds[folds.config.eq(config)]
        assert not set(fs[fs.fold.eq('train')].state)&set(fs[fs.fold.eq('test')].state)
        for outcome in OUTCOMES:
            for scope,z in [('all',u),('matched',u[u.matched])]:
                for arm,v in z.groupby('gainer'):
                    supports.append(dict(config=config,outcome=outcome,scope=scope,gainer=arm,districts=len(v),states=v.state.nunique(),
                        post_total=float(v[outcome+'_post'].sum()*2),post_active=int(v[outcome+'_post'].gt(0).sum())))
            for version in ['delta','trend_adjusted','exclude2022','placebo']:
                comparisons.append(dict(config=config,outcome=outcome,sample='all_eligible',version=version,
                                        **fit(u[outcome+'_'+version],u.state,u.gainer)))
                comparisons.append(dict(config=config,outcome=outcome,sample='matched',version=version,
                                        **fit(p[outcome+'_'+version],p.state)))
                r=averages[averages.config.eq(config)&averages.outcome.eq(outcome)&averages.version.eq(version)].iloc[0]
                check=fit(p[outcome+'_'+version],p.state)
                assert np.allclose([r.estimate,r.se,r.lo,r.hi],[check[k] for k in ['estimate','se','lo','hi']],rtol=1e-7,atol=1e-9)
            for c in [.75,1.,1.5]:
                sub=p[p.distance.le(c)]
                if len(sub)>2 and sub.state.nunique()>2:
                    calipers.append(dict(config=config,outcome=outcome,caliper=c,**fit(sub[outcome+'_delta'],sub.state)))
            for omitted in p.state.unique():
                sub=p[p.state.ne(omitted)]
                influence.append(dict(config=config,outcome=outcome,omitted_state=omitted,**fit(sub[outcome+'_delta'],sub.state)))
            a=annual[annual.config.eq(config)&annual.outcome.eq(outcome)]
            wide=a.pivot(index='unit',columns='year',values='y')
            gaps=np.stack([wide.loc[p.R,y].to_numpy()-wide.loc[p.D,y].to_numpy() for y in wide.columns],axis=1)
            years=wide.columns.to_numpy();baseline=gaps[:,years==2022].ravel()
            for k,y in enumerate(years):
                val=gaps[:,k]-baseline
                res=fit(val,p.state) if y!=2022 else dict(estimate=0.,se=0.,lo=0.,hi=0.,p=np.nan,states=p.state.nunique(),n=len(p))
                events.append(dict(config=config,outcome=outcome,year=y,**res))
            for start in [2019,2017]:
                if start<min(years):continue
                y=gaps[:,(years>=start)&(years<2022)]-baseline[:,None]
                means=y.mean(axis=0);res=y-means
                score=pd.DataFrame(res/len(p)).groupby(p.state.to_numpy()).sum().to_numpy()
                g=p.state.nunique();v=score.T@score*g/(g-1);rank=np.linalg.matrix_rank(v)
                statistic=means@np.linalg.pinv(v)@means/rank if rank else np.nan
                joint.append(dict(config=config,outcome=outcome,first_pre_year=start,pairs=len(p),states=g,
                    restrictions=rank,F=statistic,p=stats.f.sf(statistic,rank,g-1) if rank else np.nan))
    for name,rows in [('support',supports),('average_comparison',comparisons),('event_study',events),
                      ('joint_pretests',joint),('leave_state_out',influence),('caliper_sensitivity',calipers)]:
        pd.DataFrame(rows).to_csv(out/f'{name}.csv',index=False)
    # Stable disclosure of remaining imbalance, not a pass/fail based on outcome p-values.
    bal=pd.read_csv(out/'balance.csv')
    fig,axes=plt.subplots(1,3,figsize=(14,4.5),layout='constrained')
    e=pd.DataFrame(events)
    for ax,outcome,label in zip(axes,OUTCOMES,LABELS):
        z=e[e.config.eq('primary')&e.outcome.eq(outcome)]
        ax.errorbar(z.year,z.estimate,yerr=[z.estimate-z.lo,z.hi-z.estimate],fmt='o-',capsize=3)
        ax.axhline(0,color='gray',lw=1);ax.axvline(2022.5,color='gray',ls='--')
        ax.set(title=label,xlabel='Year',ylabel='Paired R-minus-D change\nrelative to 2022, per area')
    fig.suptitle('Republican House takeover — matched-area pre-trends and outcomes')
    fig.savefig(out/'event_study.png',dpi=160);plt.close(fig)
    v=pd.read_csv(out/'heldout_validation.csv')
    fig,axes=plt.subplots(1,3,figsize=(14,3.7),layout='constrained')
    for ax,outcome,label in zip(axes,OUTCOMES,LABELS):
        z=v[v.config.eq('primary')&v.outcome.eq(outcome)&v.test.isin(['delta','trend_adjusted','placebo'])]
        pos=np.arange(len(z));ax.errorbar(z.estimate,pos,xerr=[z.estimate-z.lo,z.hi-z.estimate],fmt='o',capsize=3)
        ax.axvline(0,color='gray',lw=1)
        names=z.test.map({'delta':'Post-period contrast','placebo':'Prior-year placebo','trend_adjusted':'Trend sensitivity'})
        ax.set_yticks(pos,names);ax.set(title=label,xlabel='Held-out high-minus-low contrast')
    fig.suptitle('Forest validation — 19 pairs in eight held-out states; 95% clustered intervals')
    fig.savefig(out/'forest_validation.png',dpi=160);plt.close(fig)
    p=pairs[pairs.config.eq('primary')]
    assert len(p)==49 and p.state.nunique()==21
    ptest=pred[pred.config.eq('primary')&pred.fold.eq('test')]
    assert ptest.groupby('outcome').size().eq(19).all()
    checks=dict(paired_identities_verified=True,state_disjoint_validation=True,independent_R_python_intervals_agree=True,
        primary_pairs=len(p),primary_states=p.state.nunique(),max_primary_abs_matching_smd=float(bal[bal.config.eq('primary')].after_smd.abs().max()),
        no_confirmatory_heterogeneity=bool(v[v.config.eq('primary')&v.test.eq('delta')].p_holm_primary_family.ge(.05).all()))
    (out/'verification.json').write_text(json.dumps(checks,indent=2))
    files=list(Path(__file__).parent.glob('*.py'))+list(Path(__file__).parent.glob('*.R'))
    files+=list((ROOT/'data/redesign_raw/boundaries').glob('*.zip'))+[ROOT/'revision_analysis/nc_117.zip',ROOT/'revision_analysis/panel.py',ROOT/'scripts/historical_party_panel.py']
    pd.DataFrame([dict(file=str(f.relative_to(ROOT)),sha256=hashlib.sha256(f.read_bytes()).hexdigest()) for f in files]).to_csv(out/'code_and_geometry_hashes.csv',index=False)
    print(json.dumps(checks,indent=2))
    print(pd.DataFrame(joint).query('config=="primary"').to_string(index=False))


if __name__=='__main__':summarize(Path(sys.argv[1]))
