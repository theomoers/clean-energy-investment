"""Independent numerical checks and compact scientific figures."""
import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import MAIN

LABELS={'clean_announcements':'Clean-power announcements / area-year',
        'manufacturing_announcements':'Manufacturing announcements / area-year',
        'wind_solar_net_mw':'Net wind/solar MW / area-year'}


def main(out):
    est=pd.read_csv(out/'estimates.csv');bal=pd.read_csv(out/'balance.csv');eq=pd.read_csv(out/'equivalence.csv')
    pairs=pd.read_csv(out/'forest_pairs.csv');units=pd.read_csv(out/'units2023.csv')
    folds=pd.read_csv(out/'forest_folds.csv');pred=pd.read_csv(out/'forest_predictions.csv')
    checks={}
    subgroup_balance=[]
    checks['unique_forest_pairs']=not pairs.pair.duplicated().any()
    for config,p in pairs.groupby('config'):
        checks[config+'_no_area_reuse']=len(set(p.R)|set(p.D))==2*len(p)
        checks[config+'_balance']=bool(bal[bal.config.eq(config)].after_smd.abs().max()<=.1000001)
        uu=units if config=='balanced2023' else pd.read_csv(out/'audit2011/units.csv').query('config==@config')
        keyed=uu.set_index('unit')
        checks[config+'_assignment']=bool(p.R.map(keyed.gainer).eq(1).all() and p.D.map(keyed.gainer).eq(0).all()
            and p.R.map(keyed.state).eq(p.state).all() and p.D.map(keyed.state).eq(p.state).all())
        outcomes=['wind_solar_net_mw'] if config.startswith('audit2011') else MAIN
        for o in outcomes:
            for v in ['delta','trend_adjusted','placebo','exclude_last_pre']:
                r=est[est.config.eq(config)&est.outcome.eq(o)&est.version.eq(v)].iloc[0]
                fit=sm.OLS(p[o+'_'+v],np.ones((len(p),1))).fit(cov_type='cluster',cov_kwds={'groups':p.state},use_t=True)
                assert np.allclose([r.estimate,r.se],[fit.params.iloc[0],fit.bse.iloc[0]],atol=1e-9)
        f=folds[folds.config.eq(config)]
        checks[config+'_state_disjoint']=not f.state.duplicated().any() and not (set(f[f.fold.eq('train')].state)&set(f[f.fold.eq('test')].state))
        for o,g in pred[pred.config.eq(config)].groupby('outcome'):
            checks[config+'_'+o+'_heldout_only']=bool(g.loc[g.fold.eq('train'),'heldout_score'].isna().all())
            cols=bal.loc[bal.config.eq(config),'feature'].tolist()
            for high,v in g[g.fold.eq('test')&g.high.notna()].groupby('high'):
                selected=p[p.pair.isin(v.pair)]
                for c in cols:
                    sd=uu[c].std()
                    diff=selected.R.map(keyed[c]).mean()-selected.D.map(keyed[c]).mean()
                    subgroup_balance.append(dict(config=config,outcome=o,high=high,feature=c,pairs=len(selected),
                        within_group_smd=diff/sd if sd else np.nan))
    checks['equivalence_90pct_CI_identity']=bool(((eq.lo90>-eq.bound)&(eq.hi90<eq.bound)).eq(eq.equivalent).all())
    checks['overlap_mean_balance']=bool(bal[bal.config.eq('overlap2023')].after_smd.abs().max()<1e-5)
    checks['baseline_not_post']=bool(np.isfinite(eq.pre_baseline).all())
    checks['positive_information_multipliers']=bool(eq.independent_information_multiplier_approx.ge(1).all())
    r=pd.read_csv(out/'audit2011/district_roster.csv').fillna({'exclusion':''})
    checks['historical_roster_435']=len(r)==435 and not r.duplicated(['state','district']).any()
    checks['2011_eligible_MIT_party_verification']=bool(r.loc[r.exclusion.eq(''),'mit_party_agrees'].all())
    checks['2011_no_member_changes_in_primary']=bool(r.loc[r.exclusion.eq(''),'member'].eq(r.loc[r.exclusion.eq(''),'prior_member']).all())
    assert all(checks.values()),checks
    (out/'verification.json').write_text(json.dumps(checks,indent=2))
    pd.DataFrame(subgroup_balance).to_csv(out/'forest_subgroup_balance.csv',index=False)
    # Descriptive precision planning for held-out forest high/low contrasts.
    validation=pd.read_csv(out/'forest_validation.csv')
    if len(validation):
        vals=validation[validation.version.eq('delta')].copy()
        vals['mde80_normal_approx']=(stats.t.ppf(.975,vals.states-1)+stats.norm.ppf(.8))*vals.se
        vals['minimal_symmetric_equivalence_bound']=abs(vals.estimate)+stats.t.ppf(.95,vals.states-1)*vals.se
        vals.to_csv(out/'forest_precision.csv',index=False)
        fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
        for ax,o in zip(axes,MAIN):
            g=vals[vals.outcome.eq(o)].reset_index(drop=True)
            if len(g):
                ax.errorbar(g.estimate,np.arange(len(g)),xerr=[g.estimate-g.lo,g.hi-g.estimate],fmt='o',capsize=4)
                names=g.config.replace({'audit2011_transition_stable':'2011 transition-stable',
                    'audit2011_continuous_member':'2011 continuous-member','balanced2023':'2023 balanced'})
                ax.set_yticks(np.arange(len(g)),names)
                ax.axvline(0,color='gray');ax.set_xlabel(LABELS[o]);ax.invert_yaxis()
            else:
                ax.text(.5,.5,'Manufacturing announcements\nInsufficient held-out high/low support',transform=ax.transAxes,ha='center')
                ax.set_axis_off()
            if o=='wind_solar_net_mw':ax.text(.02,.5,'2023: insufficient high/low support',transform=ax.transAxes,fontsize=9)
        fig.suptitle('Forest-ranked high-minus-low effects: held-out estimates and 95% intervals',fontsize=12)
        fig.savefig(out/'forest_validation.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    names=['original2023','balanced2023','overlap2023'];labels=['Original matching','Balance-constrained matching','Overlap weighting']
    for ax,o in zip(axes,MAIN):
        g=est[est.outcome.eq(o)&est.version.eq('delta')].set_index('config').loc[names]
        ax.errorbar(g.estimate,np.arange(3),xerr=[g.estimate-g.lo,g.hi-g.estimate],fmt='o',capsize=4)
        ax.axvline(0,color='gray',lw=1);ax.set_yticks(np.arange(3),labels if ax==axes[0] else ['']*3)
        ax.set_xlabel(LABELS[o]);ax.invert_yaxis()
    fig.suptitle('2023 Republican takeover: estimates and 95% intervals\nExploratory comparisons; target populations differ',fontsize=12)
    fig.savefig(out/'estimates2023.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,5),layout='constrained')
    for config,marker in [('original2023','o'),('balanced2023','s'),('overlap2023','^')]:
        g=bal[bal.config.eq(config)]
        ax.plot(g.after_smd.abs(),range(len(g)),marker,ls='',label=config)
    ax.set_yticks(range(len(g)),g.feature);ax.axvline(.1,color='gray',ls='--');ax.set_xlabel('Absolute standardized mean difference')
    ax.legend();ax.set_title('Balance on nine measured pre-treatment characteristics')
    fig.savefig(out/'balance.png',dpi=160);plt.close(fig)
    event=pd.read_csv(out/'audit2011/event_study.csv')
    matched=pd.read_csv(out/'matched_event_study.csv')
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for ax,source,title in zip(axes,[event,matched],['All eligible mixed-state districts','Balance-constrained pairs']):
        g=source[source.config.eq('audit2011_transition_stable')].sort_values('year')
        ax.errorbar(g.year,g.estimate,yerr=[g.estimate-g.lo,g.hi-g.estimate],fmt='o-',capsize=3)
        ax.axhline(0,color='gray');ax.axvline(2010.5,color='gray',ls='--')
        ax.set(title=title,xlabel='Year',ylabel='R–D relative net MW change\nversus 2010, per area-year')
    fig.suptitle('2011 takeover feasibility: annual diagnostics, 95% intervals')
    fig.savefig(out/'audit2011_pretrends.png',dpi=160);plt.close(fig)
    print('Verified',len(checks),'checks',flush=True)
