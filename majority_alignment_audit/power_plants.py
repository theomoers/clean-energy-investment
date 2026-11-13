"""Historical RHG plant-capacity support and pre-trends, no forecast outcomes.

Rows do not have generator IDs. Identical rows within a scenario may be distinct
generators and MUST NOT be dropped. Earliest Operating Year is not used to date
new generators. Outcome is annual net capacity change, not gross investment.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from audit import ROOT,geographies,assignments,diagnostic_fit,joint_pre

HERE=Path(__file__).resolve().parent
OUT=HERE/'power_plants'
INP=ROOT/'data/RHG_PowerPlants_TS2026_20261106'


def main():
    OUT.mkdir(exist_ok=False)
    raw=pd.read_csv(INP/'plants_in_operation.csv')
    fullcols=[c for c in raw if c!='Scenario']
    # Compare multisets with multiplicities, not just national capacity totals.
    fingerprints={}
    for scenario,g in raw.groupby('Scenario'):
        h=np.sort(pd.util.hash_pandas_object(g[fullcols],index=False).to_numpy())
        fingerprints[scenario]=hashlib.sha256(h.tobytes()).hexdigest()
    assert len(set(fingerprints.values()))==1, fingerprints
    p=raw[raw.Scenario.eq('ts_mid')].copy()
    assert p.Year.max()<=2025
    retire=pd.read_csv(INP/'retirements.csv')
    retire.groupby(['Scenario','Retirement Type']).agg(rows=('Retirement Year','size'),first=('Retirement Year','min'),last=('Retirement Year','max')).to_csv(OUT/'retirement_coverage.csv')
    summary={'raw_rows':len(raw),'scenario_fingerprints':fingerprints,'selected_scenario':'ts_mid',
        'selected_rows':len(p),'first_year':int(p.Year.min()),'last_year':int(p.Year.max()),
        'plants':p['EIA Plant ID'].nunique(),'within_scenario_identical_rows_retained':int(p.duplicated().sum()),
        'warning':'Rows have no generator identifier. Repeated identical rows retained. Forecast retirements excluded. No gross additions inferred.'}
    (OUT/'source_audit.json').write_text(json.dumps(summary,indent=2))
    p.groupby(['Year','Fuel Category']).agg(rows=('EIA Plant ID','size'),plants=('EIA Plant ID','nunique'),mw=('Capacity (MW)','sum')).to_csv(OUT/'national_capacity.csv')
    # One spatial operation per distinct plant-coordinate-state location.
    cols=['EIA Plant ID','Latitude','Longitude','State Code']
    locations=p[cols].drop_duplicates().reset_index(drop=True)
    locations['unique_id']=np.arange(len(locations))
    locations['LatLon_Valid']=locations.Latitude.between(-90,90)&locations.Longitude.between(-180,180)&~(locations.Latitude.eq(0)&locations.Longitude.eq(0))
    locations=locations.rename(columns={'State Code':'State'})
    loc=assignments(locations,geographies(),OUT)
    p=p.merge(locations.rename(columns={'State':'State Code'})[cols+['unique_id']],on=cols,validate='m:1')
    rosters=pd.read_csv(HERE/'latest/district_roster.csv').fillna({'transition_exclusion':''})
    results=[];events=[];support=[];panels=[];joint=[]
    for transition,first,threshold in [(2019,2013,.99),(2023,2017,.5),(2023,2017,.8)]:
        new=116 if transition==2019 else 118;sel_first=2017 if transition==2019 else 2019
        rr=rosters[(rosters.transition==transition)&rosters.first_year.eq(sel_first)&rosters.threshold.eq(threshold)&rosters.transition_exclusion.eq('')]
        for sample in ['transition_stable','continuous_member']:
            r=rr[rr.continuous_member if sample=='continuous_member' else np.ones(len(rr),bool)].copy()
            pts=locations.merge(loc[(loc.congress==new)&loc.reason.eq('matched')][['unique_id','state','district']],on='unique_id',validate='1:1')
            pts=pts.merge(r[['state','district','gainer','old115','old116','old117']],on=['state','district'],validate='m:1')
            for old in [115,116,117]:
                c=f'old{old}'
                if not r[c].notna().any():continue
                a=loc[(loc.congress==old)&loc.reason.eq('matched')][['unique_id','district']].rename(columns={'district':'old_mapped'})
                pts=pts.merge(a,on='unique_id',how='left',validate='1:1')
                pts=pts[pts[c].isna()|pts[c].eq(pts.old_mapped)].drop(columns='old_mapped')
            f=p.merge(pts[['unique_id','state','district','gainer']],on='unique_id',validate='m:1')
            for sector,fuels in [('wind_solar',['Wind','Solar']),('storage',['Storage'])]:
                z=f[f['Fuel Category'].isin(fuels)&f.Year.between(first-1,transition+1)]
                agg=z.groupby(['state','district','Year'])['Capacity (MW)'].sum().rename('mw')
                grid=r[['state','district','gainer']].merge(pd.DataFrame({'Year':range(first-1,transition+2)}),how='cross')
                grid=grid.merge(agg,on=['state','district','Year'],how='left',validate='1:1').fillna({'mw':0})
                grid=grid.sort_values(['state','district','Year'])
                grid['net_mw']=grid.groupby(['state','district']).mw.diff()
                grid=grid[grid.Year>=first].rename(columns={'Year':'year'})
                assert not grid.net_mw.isna().any()
                tag=dict(transition=transition,first_year=first,threshold=threshold,sample=sample,sector=sector)
                panels.append(grid.assign(**tag))
                pre=grid[grid.year<transition].groupby(['state','district']).net_mw.mean()
                post=grid[grid.year>=transition].groupby(['state','district']).net_mw.mean()
                delta=r.set_index(['state','district'])[['gainer']].assign(delta=post-pre).reset_index()
                for state_fe in [False,True]:
                    results.append(dict(**tag,test='post_minus_pre_relative_change',state_fe=state_fe,**diagnostic_fit(delta,'delta',state_fe)))
                base=grid[grid.year.eq(transition-1)].set_index(['state','district']).net_mw
                for year,a in grid.groupby('year'):
                    d=a.set_index(['state','district']).copy();d['change']=d.net_mw-base;d=d.reset_index()
                    res=diagnostic_fit(d,'change',True) if year!=transition-1 else dict(estimate=0.,se=0.,lo=0.,hi=0.,p=np.nan)
                    events.append(dict(**tag,year=year,**res))
                joint.append(dict(**tag,**joint_pre(grid,'net_mw',transition)))
                slopes=[]
                for (st,dist),g in grid[grid.year<transition].groupby(['state','district']):
                    slopes.append([st,dist,g.gainer.iloc[0],np.polyfit(g.year,g.net_mw,1)[0]])
                slopes=pd.DataFrame(slopes,columns=['state','district','gainer','slope'])
                results.append(dict(**tag,test='pre_annual_net_addition_slope',state_fe=True,**diagnostic_fit(slopes,'slope',True)))
                mixed=r.groupby('state').gainer.nunique();mixed=set(mixed[mixed.eq(2)].index)
                for scope,rg in [('all',r),('within_state_overlap',r[r.state.isin(mixed)])]:
                    for arm,a in rg.groupby('gainer'):
                        v=grid.merge(a[['state','district']],on=['state','district'],validate='m:1')
                        postv=v[v.year>=transition]
                        changes=postv.groupby(['state','district']).net_mw
                        support.append(dict(**tag,scope=scope,gainer=arm,districts=len(a),states=a.state.nunique(),
                            post_net_mw=postv.net_mw.sum(),post_districts_with_net_addition=int(changes.apply(lambda x:(x>1e-6).any()).sum()),
                            post_districts_with_any_change=int(changes.apply(lambda x:(abs(x)>1e-6).any()).sum())))
    for name,rows in [('diagnostics',results),('event_study',events),('support',support),('joint_pretests',joint)]:
        pd.DataFrame(rows).to_csv(OUT/f'{name}.csv',index=False)
    pd.concat(panels).to_csv(OUT/'district_year_panel.csv',index=False)
    e=pd.DataFrame(events)
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for ax,t in zip(axes,[2019,2023]):
        a=e[e.transition.eq(t)&e.threshold.eq(.99 if t==2019 else .5)&e['sample'].eq('continuous_member')&e.sector.eq('wind_solar')].sort_values('year')
        ax.errorbar(a.year,a.estimate,yerr=[a.estimate-a.lo,a.hi-a.estimate],fmt='o-',capsize=3)
        ax.axhline(0,color='gray');ax.axvline(t-.5,color='gray',ls='--')
        ax.set(title=f'{t} majority transition',xlabel='Year',ylabel='Relative change in net wind/solar MW\nversus last pre-year, per district')
    fig.savefig(OUT/'pretrends_capacity.png',dpi=160);plt.close(fig)
    pd.DataFrame([{'file':str(x.relative_to(ROOT)),'sha256':hashlib.sha256(x.read_bytes()).hexdigest()} for x in
                  [INP/'plants_in_operation.csv',INP/'retirements.csv',Path(__file__)]]).to_csv(OUT/'input_hashes.csv',index=False)
    print(json.dumps(summary,indent=2))
    print(pd.DataFrame(joint).query('sample=="continuous_member"').to_string(index=False))


if __name__=='__main__':main()
