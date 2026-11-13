"""2023 within-state matched DiD inputs. Matching never uses 2022+ outcomes."""
from pathlib import Path
import hashlib
import sys
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
import shapely

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'majority_alignment_audit'))
from audit import geographies
A=ROOT/'majority_alignment_audit/verified_run'
FEATURES=['log_clean_capacity','log_fossil_capacity','log_prior_announcements','log_core_area']
MATCH=FEATURES+['prior_announcement_slope']
CONFIGS=[('primary','transition_stable',.5),('continuous_member','continuous_member',.5),
         ('overlap80','transition_stable',.8)]


def prepare(out):
    out.mkdir(parents=True,exist_ok=False)
    r=pd.read_csv(A/'latest/district_roster.csv').fillna({'transition_exclusion':''})
    r=r[r.transition.eq(2023)&r.first_year.eq(2019)&r.transition_exclusion.eq('')].copy()
    ann=pd.read_csv(A/'latest/district_year_panel.csv')
    capacity=pd.read_csv(A/'power_plants/district_year_panel.csv')
    # Reuse the audited location-ID construction exactly, retaining generator multiplicities.
    raw=pd.read_csv(ROOT/'data/RHG_PowerPlants_TS2026_20261106/plants_in_operation.csv')
    raw=raw[raw.Scenario.eq('ts_mid')].copy()
    cols=['EIA Plant ID','Latitude','Longitude','State Code']
    locations=raw[cols].drop_duplicates().reset_index(drop=True).assign(unique_id=lambda d:np.arange(len(d)))
    raw=raw.merge(locations,on=cols,validate='m:1')
    loc=pd.read_csv(A/'power_plants/facility_geography.csv')
    loc=loc[loc.reason.eq('matched')]
    base=r[r.threshold.eq(.5)].copy()
    l=loc[loc.congress.eq(118)][['unique_id','state','district']].merge(
        base[['state','district','old116','old117']],on=['state','district'],validate='m:1')
    for n in [116,117]:
        l=l.merge(loc[loc.congress.eq(n)][['unique_id','district']].rename(columns={'district':'old_mapped'}),on='unique_id',validate='1:1')
        l=l[l[f'old{n}'].eq(l.old_mapped)].drop(columns='old_mapped')
    z=raw[raw.Year.eq(2021)].merge(l[['unique_id','state','district']],on='unique_id',validate='m:1')
    clean=z[z['Fuel Category'].isin(['Wind','Solar'])].groupby(['state','district'])['Capacity (MW)'].sum()
    fossil=z[z['Fuel Category'].isin(['Coal','Natural Gas','Oil'])].groupby(['state','district'])['Capacity (MW)'].sum()
    maps={n:g.to_crs(6933).assign(geometry=lambda x:x.geometry.make_valid().simplify(50)).set_index(['state','district'])
          for n,g in geographies().items()}
    areas=[]
    for row in base.itertuples():
        g=maps[118].loc[(row.state,row.district),'geometry']
        for n in [116,117]:g=g.intersection(maps[n].loc[(row.state,getattr(row,f'old{n}')),'geometry'])
        areas.append((row.state,row.district,g.area/1e6))
    areas=pd.DataFrame(areas,columns=['state','district','core_area_km2']).set_index(['state','district']).core_area_km2
    base=base.set_index(['state','district']).assign(clean_capacity_2021=clean,fossil_capacity_2021=fossil,core_area_km2=areas)
    base[['clean_capacity_2021','fossil_capacity_2021']]=base[['clean_capacity_2021','fossil_capacity_2021']].fillna(0)
    assert base.core_area_km2.gt(0).all()
    allunits=[];allpairs=[];allbalance=[];allannual=[];assignment=[]
    for config,sample,h in CONFIGS:
        units=r[r.threshold.eq(h)&(r.continuous_member if sample=='continuous_member' else True)].copy()
        both=units.groupby('state').gainer.nunique();units=units[units.state.isin(both[both.eq(2)].index)]
        units=units.set_index(['state','district'])
        for col in ['clean_capacity_2021','fossil_capacity_2021','core_area_km2']:units[col]=base[col]
        a=ann[ann.transition.eq(2023)&ann.first_year.eq(2019)&ann.threshold.eq(h)&ann['sample'].eq(sample)]
        cp=a[a.sector.eq('clean_power')]
        prior=cp[cp.year.le(2021)].groupby(['state','district']).projects.mean()
        slope=cp[cp.year.le(2021)].groupby(['state','district']).apply(lambda g:np.polyfit(g.year-2019,g.projects,1)[0])
        units['log_clean_capacity']=np.log1p(units.clean_capacity_2021)
        units['log_fossil_capacity']=np.log1p(units.fossil_capacity_2021)
        units['log_core_area']=np.log1p(units.core_area_km2)
        units['log_prior_announcements']=np.log1p(prior)
        units['prior_announcement_slope']=np.arcsinh(slope)
        assert units[MATCH].notna().all().all()
        units=units.reset_index().assign(unit=lambda x:x.state+'-'+x.district.astype(int).astype(str))
        scaling=units[MATCH].std().replace(0,1)
        scaled=units[MATCH]/scaling
        records=[]
        for state,g in units.groupby('state'):
            ir=g.index[g.gainer.eq(1)];idem=g.index[g.gainer.eq(0)]
            cost=np.sqrt(((scaled.loc[ir].to_numpy()[:,None,:]-scaled.loc[idem].to_numpy()[None,:,:])**2).mean(axis=2))
            left,right=linear_sum_assignment(cost)
            for i,j in zip(left,right):
                ii,jj=ir[i],idem[j];keep=cost[i,j]<=1.5
                assignment.append(dict(config=config,state=state,R=units.loc[ii,'unit'],D=units.loc[jj,'unit'],distance=cost[i,j],retained=keep))
                if keep:records.append((state,ii,jj,cost[i,j]))
        pairs=[]
        for k,(state,i,j,cost) in enumerate(records):
            rr,dd=units.loc[i],units.loc[j]
            pair=dict(config=config,pair=f'{config}_{k:03}',state=state,R=rr.unit,D=dd.unit,distance=cost)
            for f in FEATURES:pair[f]=(rr[f]+dd[f])/2
            pairs.append(pair)
        pairs=pd.DataFrame(pairs);assert pairs.R.is_unique and pairs.D.is_unique
        matched=set(pairs.R)|set(pairs.D)
        units['matched']=units.unit.isin(matched);units['config']=config
        for f in MATCH:
            before=units.groupby('gainer')[f].mean();after=units[units.matched].groupby('gainer')[f].mean()
            allbalance.append(dict(config=config,feature=f,before_smd=(before[1]-before[0])/scaling[f],after_smd=(after[1]-after[0])/scaling[f]))
        # Carry identical units into all outcomes. No outcome-dependent sample selection.
        for outcome in ['clean_announcements','manufacturing_announcements','wind_solar_net_mw']:
            if outcome=='wind_solar_net_mw':
                t=capacity[capacity.transition.eq(2023)&capacity.threshold.eq(h)&capacity['sample'].eq(sample)&capacity.sector.eq('wind_solar')].copy()
                t['y']=t.net_mw
            else:
                t=a[a.sector.eq('clean_power' if outcome=='clean_announcements' else 'manufacturing')].copy();t['y']=t.projects
            t['unit']=t.state+'-'+t.district.astype(int).astype(str)
            t=t[t.unit.isin(units.unit)]
            allannual.append(t[['state','district','unit','gainer','year','y']].assign(config=config,outcome=outcome))
            pre=t[t.year.between(2019,2022)].groupby('unit').y.mean()
            post=t[t.year.between(2023,2024)].groupby('unit').y.mean()
            placebo=t[t.year.eq(2022)].set_index('unit').y-t[t.year.eq(2021)].set_index('unit').y
            predicted={}
            for u,g in t[t.year.between(2019,2022)].groupby('unit'):
                fit=np.polyfit(g.year-2022,g.y,1);predicted[u]=np.polyval(fit,[1,2]).mean()
            adjusted=post-pd.Series(predicted)
            pre_no_2022=t[t.year.between(2019,2021)].groupby('unit').y.mean()
            for name,values in [('pre',pre),('post',post),('delta',post-pre),('placebo',placebo),('trend_adjusted',adjusted),('exclude2022',post-pre_no_2022)]:
                units[f'{outcome}_{name}']=units.unit.map(values)
                pairs[f'{outcome}_{name}_R']=pairs.R.map(values)
                pairs[f'{outcome}_{name}_D']=pairs.D.map(values)
                pairs[f'{outcome}_{name}']=pairs.R.map(values)-pairs.D.map(values)
        assert np.isfinite(pairs.select_dtypes('number')).all().all()
        allunits.append(units);allpairs.append(pairs)
    pd.concat(allunits).to_csv(out/'units.csv',index=False)
    pd.concat(allpairs).to_csv(out/'pairs.csv',index=False)
    pd.concat(allannual).to_csv(out/'annual.csv',index=False)
    pd.DataFrame(allbalance).to_csv(out/'balance.csv',index=False)
    pd.DataFrame(assignment).to_csv(out/'matching_ledger.csv',index=False)
    files=[A/'latest/district_roster.csv',A/'latest/district_year_panel.csv',A/'power_plants/district_year_panel.csv',
           A/'power_plants/facility_geography.csv',ROOT/'data/RHG_PowerPlants_TS2026_20261106/plants_in_operation.csv',Path(__file__)]
    pd.DataFrame([dict(file=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files]).to_csv(out/'inputs.csv',index=False)
    print(pd.concat(allpairs).groupby('config').agg(pairs=('pair','size'),states=('state','nunique')))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    prepare(parser.parse_args().out)
