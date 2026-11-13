"""Balance extensions and mutually exclusive technology outcomes."""
import numpy as np
import pandas as pd
from common import ROOT,OLD,AUDIT,MAIN,BASE,EXTRA,cardinality,add_outcomes

CATEGORIES=['solar_announcements','wind_announcements','storage_announcements',
            'battery_manufacturing','solar_manufacturing','other_manufacturing']


def category_panel(units,out):
    raw=pd.read_csv(ROOT/'data/CleanInvestmentMonitor_Download/manufacturing_energy_and_industry_facility_metadata.csv',skiprows=5)
    raw['date']=pd.to_datetime(raw.Announcement_Date,format='mixed')
    loc=pd.read_csv(AUDIT/'latest/facility_geography.csv')
    f=raw[raw.date.dt.year.between(2019,2024)].merge(loc[loc.congress.eq(118)&loc.reason.eq('matched')][['unique_id','state','district']],on='unique_id',validate='1:1')
    f=f.merge(units[['unit','state','district','old116','old117']],on=['state','district'],validate='m:1')
    for old in [116,117]:
        f=f.merge(loc[loc.congress.eq(old)&loc.reason.eq('matched')][['unique_id','district']].rename(columns={'district':'old_mapped'}),on='unique_id',validate='1:1')
        f=f[f[f'old{old}'].eq(f.old_mapped)].drop(columns='old_mapped')
    f=f[~(f.date.dt.year.eq(2023)&f.date.dt.month.eq(1)&f.date.dt.day.le(2))].copy()
    f['year']=f.date.dt.year;f['outcome']='outside'
    for tech in ['Solar','Wind','Storage']:
        f.loc[f.Segment.eq('Energy and Industry')&f.Technology.eq(tech),'outcome']=tech.lower()+'_announcements'
    f.loc[f.Segment.eq('Manufacturing'),'outcome']='other_manufacturing'
    f.loc[f.Segment.eq('Manufacturing')&f.Technology.eq('Batteries'),'outcome']='battery_manufacturing'
    f.loc[f.Segment.eq('Manufacturing')&f.Technology.eq('Solar'),'outcome']='solar_manufacturing'
    f=f[f.outcome.ne('outside')];assert f.unique_id.is_unique
    f[['unique_id','unit','state','district','year','outcome','date','Technology','Subcategory']].to_csv(out/'category_facility_assignments.csv',index=False)
    grid=units[['unit','state','district','gainer']].merge(pd.DataFrame({'year':range(2019,2025)}),how='cross').merge(pd.DataFrame({'outcome':CATEGORIES}),how='cross')
    counts=f.groupby(['unit','year','outcome']).size().rename('y')
    grid=grid.merge(counts,on=['unit','year','outcome'],how='left',validate='1:1').fillna({'y':0})
    assert grid.y.sum()==len(f)
    grid.to_csv(out/'category_annual.csv',index=False)
    return grid


def main(out):
    units=pd.read_csv(OLD/'units.csv');units=units[units.config.eq('primary')].copy().reset_index(drop=True)
    units=units.rename(columns={c:c.replace('_exclude2022','_exclude_last_pre') for c in units if c.endswith('_exclude2022')})
    annual=pd.read_csv(OLD/'annual.csv');annual=annual[annual.config.eq('primary')].copy()
    assert len(units)==196 and units.unit.is_unique
    cats=category_panel(units,out)
    for major,subset in [('clean_announcements',CATEGORIES[:3]),('manufacturing_announcements',CATEGORIES[3:])]:
        summed=cats[cats.outcome.isin(subset)].groupby(['unit','year']).y.sum().sort_index()
        original=annual[annual.outcome.eq(major)].set_index(['unit','year']).y.sort_index()
        assert np.allclose(summed,original)
    units=add_outcomes(units,cats,list(range(2019,2023)),[2023,2024])
    for outcome,prefix in [('manufacturing_announcements','manufacturing'),('wind_solar_net_mw','net_mw')]:
        pre=annual[annual.outcome.eq(outcome)&annual.year.le(2021)].pivot(index='unit',columns='year',values='y')
        avg=pre.mean(axis=1);slope=(pre[2021]-pre[2019])/2
        units['log_prior_manufacturing' if prefix=='manufacturing' else 'prior_net_mw']=units.unit.map(np.log1p(avg) if prefix=='manufacturing' else np.arcsinh(avg))
        units['prior_'+prefix+'_slope']=units.unit.map(np.arcsinh(slope))
    assert units[BASE+EXTRA].notna().all().all()
    pairs,ledger=cardinality(units,BASE+EXTRA,BASE,out,'balanced2023')
    units.to_csv(out/'units2023.csv',index=False);pairs.to_csv(out/'pairs2023.csv',index=False)
    pd.DataFrame([ledger]).to_csv(out/'matching_summary2023.csv',index=False)
    return units,pairs,pd.concat([annual,cats],ignore_index=True)
