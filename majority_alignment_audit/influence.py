"""Capacity influence diagnostics and illustrative subgroup precision."""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from audit import diagnostic_fit
from supplement import HERE


def main():
    p=pd.read_csv(HERE/'power_plants/district_year_panel.csv')
    results=[];precision=[]
    for sample in ['continuous_member','transition_stable']:
        g=p[p.transition.eq(2023)&p.threshold.eq(.5)&p['sample'].eq(sample)&p.sector.eq('wind_solar')]
        pre=g[g.year<2023].groupby(['state','district']).net_mw.mean()
        post=g[g.year>=2023].groupby(['state','district']).net_mw.mean()
        d=g[['state','district','gainer']].drop_duplicates().set_index(['state','district']).assign(delta=post-pre).reset_index()
        both=d.groupby('state').gainer.nunique();d=d[d.state.isin(both[both.eq(2)].index)].reset_index(drop=True)
        trend=[]
        for (st,dist),v in g[g.year<2023].groupby(['state','district']):
            coef=np.polyfit(v.year-2022,v.net_mw,1)
            predicted_post=np.polyval(coef,np.array([1,2])).mean()
            trend.append([st,dist,predicted_post])
        adjusted=d.merge(pd.DataFrame(trend,columns=['state','district','predicted_post']),on=['state','district'],validate='1:1')
        adjusted['adjusted']=post.reindex(pd.MultiIndex.from_frame(adjusted[['state','district']])).to_numpy()-adjusted.predicted_post
        results.append(dict(sample=sample,omit_state='linear_pretrend_extrapolation',**diagnostic_fit(adjusted,'adjusted',True)))
        for state in ['none',*sorted(d.state.unique())]:
            a=d[d.state.ne(state)]
            results.append(dict(sample=sample,omit_state=state,**diagnostic_fit(a,'delta',True)))
        largest=d.delta.abs().idxmax()
        results.append(dict(sample=sample,omit_state=f'largest_change_district_{d.loc[largest,"state"]}_{d.loc[largest,"district"]}',
                            **diagnostic_fit(d.drop(index=largest),'delta',True)))
        states=np.array(sorted(d.state.unique()));rng=np.random.default_rng(48123+2023);rng.shuffle(states)
        for fold,ss in [('all',set(states)),('test',set(states[:len(states)//2]))]:
            a=d[d.state.isin(ss)].reset_index(drop=True);ses=[];rng=np.random.default_rng(88413)
            for _ in range(300):
                high=rng.permutation(np.arange(len(a)))<len(a)//2
                if min(pd.crosstab(high,a.gainer).to_numpy().ravel())<5:continue
                x=pd.DataFrame({'const':1.,'gainer':a.gainer,'high':high.astype(float),'interaction':high*a.gainer})
                x=pd.concat([x,pd.get_dummies(a.state,drop_first=True,dtype=float)],axis=1).astype(float)
                f=sm.OLS(a.delta,x).fit(cov_type='cluster',cov_kwds={'groups':a.state})
                ses.append(f.bse.interaction)
            se=float(np.median(ses))
            precision.append(dict(sample=sample,fold=fold,districts=len(a),states=a.state.nunique(),
                                  median_interaction_se=se,approximate_80pct_mde=(stats.t.ppf(.975,a.state.nunique()-1)+stats.norm.ppf(.8))*se))
    pd.DataFrame(results).to_csv(HERE/'power_plants/influence.csv',index=False)
    pd.DataFrame(precision).to_csv(HERE/'power_plants/heterogeneity_precision.csv',index=False)
    print(pd.DataFrame(results).query('sample=="continuous_member"').to_string(index=False))
    print(pd.DataFrame(precision).to_string(index=False))


if __name__=='__main__':main()
