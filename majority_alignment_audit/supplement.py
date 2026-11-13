"""Release bridge and cluster-aware support/illustrative heterogeneity precision."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from audit import ROOT

HERE=Path(__file__).resolve().parent


def main():
    out=HERE/'supplement';out.mkdir(exist_ok=False)
    old=pd.read_csv(ROOT/'data/manufacturing_energy_and_industry_facility_metadata.csv',skiprows=5,low_memory=False)
    new=pd.read_csv(ROOT/'data/CleanInvestmentMonitor_Download/manufacturing_energy_and_industry_facility_metadata.csv',skiprows=5,low_memory=False)
    for d in [old,new]:d['date']=pd.to_datetime(d.Announcement_Date,format='mixed')
    bridge=old[['unique_id','Segment','date','LatLon_Valid']].merge(new[['unique_id','Segment','date','LatLon_Valid']],on='unique_id',how='outer',suffixes=('_old','_new'),indicator=True)
    bridge['date_changed']=bridge.date_old.ne(bridge.date_new)&bridge['_merge'].eq('both')
    bridge.to_csv(out/'release_id_bridge.csv',index=False)
    bridge.groupby(['_merge','Segment_old','Segment_new'],observed=True,dropna=False).size().rename('facilities').to_csv(out/'release_bridge_summary.csv')
    pd.DataFrame([dict(release=label,facilities=len(d),manufacturing=d.Segment.eq('Manufacturing').sum(),valid=int(d.LatLon_Valid.sum()),
                       first=str(d.date.min().date()),last=str(d.date.max().date()),
                       dated_2018_24=int(d.date.dt.year.between(2018,2024).sum())) for label,d in [('2024Q4',old),('2026Q2',new)]]).to_csv(out/'release_coverage.csv',index=False)
    outputs=[];precision=[]
    for label in ['frozen','latest']:
        panel=pd.read_csv(HERE/label/'district_year_panel.csv')
        for trans,first,threshold in [(2019,2017,.99),(2023,2019,.5),(2023,2019,.8)]:
            for sample in ['transition_stable','continuous_member']:
                for sector in ['manufacturing','clean_power']:
                    d=panel[(panel.transition==trans)&panel.first_year.eq(first)&panel.threshold.eq(threshold)&panel['sample'].eq(sample)&panel.sector.eq(sector)].copy()
                    mixed=d.groupby('state').gainer.nunique();d=d[d.state.isin(mixed[mixed.eq(2)].index)]
                    units=d[['state','district','gainer']].drop_duplicates()
                    states=np.array(sorted(units.state.unique()));rng=np.random.default_rng(48123+trans);rng.shuffle(states)
                    teststates=set(states[:len(states)//2])
                    for fold,st in [('all',set(states)),('train',set(states)-teststates),('test',teststates)]:
                        a=d[d.state.isin(st)]
                        for arm,z in a.groupby('gainer'):
                            post=z[z.year>=trans];pre=z[z.year<trans]
                            outputs.append(dict(release=label,transition=trans,first_year=first,threshold=threshold,sample=sample,sector=sector,fold=fold,gainer=arm,
                                districts=z[['state','district']].drop_duplicates().shape[0],states=z.state.nunique(),pre_projects=pre.projects.sum(),post_projects=post.projects.sum(),
                                post_active=post.groupby(['state','district']).projects.sum().gt(0).sum()))
                    # Illustrative known random subgroup with no subgroup discovery cost.
                    # Analytic 80% MDE approximation, not a forest power guarantee.
                    for fold,st in [('all',set(states)),('test',teststates)]:
                        g=units[units.state.isin(st)].copy()
                        z=d[d.state.isin(st)]
                        pre=z[z.year<trans].groupby(['state','district']).projects.mean()
                        post=z[z.year>=trans].groupby(['state','district']).projects.mean()
                        g=g.set_index(['state','district']).assign(delta=post-pre).reset_index()
                        ses=[];rng=np.random.default_rng(88413)
                        for _ in range(300):
                            high=rng.permutation(np.arange(len(g)))<len(g)//2
                            if min(pd.crosstab(high,g.gainer).to_numpy().ravel())<5:continue
                            x=pd.DataFrame({'const':1.,'gainer':g.gainer,'high':high.astype(float),'interaction':high*g.gainer})
                            x=pd.concat([x,pd.get_dummies(g.state,drop_first=True,dtype=float)],axis=1).astype(float)
                            f=sm.OLS(g.delta,x).fit(cov_type='cluster',cov_kwds={'groups':g.state})
                            ses.append(float(f.bse.interaction))
                        se=float(np.median(ses)) if ses else np.nan
                        precision.append(dict(release=label,transition=trans,first_year=first,threshold=threshold,sample=sample,sector=sector,fold=fold,
                            districts=len(g),states=g.state.nunique(),replications=len(ses),median_interaction_se=se,
                            approximate_80pct_mde=(stats.t.ppf(.975,g.state.nunique()-1)+stats.norm.ppf(.8))*se,
                            post_mean_projects_per_district_year=float(post.mean())))
    pd.DataFrame(outputs).to_csv(out/'effective_support.csv',index=False)
    pd.DataFrame(precision).to_csv(out/'heterogeneity_precision.csv',index=False)
    # Cross-check district label claims. Current metadata labels are not historical assignment.
    loc=pd.read_csv(HERE/'latest/facility_geography.csv')
    z=new.merge(loc[loc.congress.eq(118)&loc.reason.eq('matched')],on='unique_id',validate='1:1')
    z['label_district']=pd.to_numeric(z.CD119_2024_Name.str.extract(r'-(\d+)$')[0],errors='coerce')
    z['district_label_disagrees_118']=z.label_district.ne(z.district)
    z.groupby('State').agg(mapped=('unique_id','size'),disagrees_118=('district_label_disagrees_118','sum')).to_csv(out/'current_label_geography_audit.csv')
    print(pd.DataFrame(outputs).query('release=="latest" and transition==2023 and threshold==.5 and sample=="continuous_member" and fold=="all"').to_string(index=False))
    print(pd.DataFrame(precision).query('release=="latest" and transition==2023 and threshold==.5 and sample=="continuous_member"').to_string(index=False))


if __name__=='__main__':main()
