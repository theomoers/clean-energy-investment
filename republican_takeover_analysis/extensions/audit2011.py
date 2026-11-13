"""2011 majority takeover: fixed 2007-12 geography, dated terms, RHG history."""
import sys,json
import numpy as np
import pandas as pd
import geopandas as gp
import pyproj
from common import ROOT,add_outcomes,cardinality
sys.path.insert(0,str(ROOT/'majority_alignment_audit'))
from audit import member_at,diagnostic_fit,joint_pre
from historical_party_panel import FIPS_TO_STATE,read_map


def terms():
    rows=[]
    for p in sorted((ROOT/'data/redesign_raw/legislators').glob('*.json')):
        for person in json.loads(p.read_text()):
            for t in person['terms']:
                if t['type']!='rep' or t['end']<'2007' or t['start']>'2013':continue
                for a in t.get('party_affiliations') or [t]:
                    rows.append(dict(state=t['state'],district=int(t['district']),
                        start=max(pd.Timestamp(t['start']),pd.Timestamp(a['start'])),
                        end=min(pd.Timestamp(t['end']),pd.Timestamp(a['end'])+pd.Timedelta(days=1)),
                        party=a['party'],member=person['id']['bioguide'],
                        name=person['name'].get('official_full',person['name']['first']+' '+person['name']['last'])))
    return pd.DataFrame(rows).drop_duplicates()


def main(out):
    out.mkdir()
    pyproj.network.set_network_enabled(False)
    states=set(FIPS_TO_STATE.values())-{'DC'}
    maps=read_map('112',ROOT/'data/redesign_raw/boundaries/tl_2011_us_cd112.zip','CD112FP',states).to_crs(4269)
    maps=maps[maps.district.lt(90)].reset_index(drop=True)
    assert len(maps)==435 and not maps.duplicated(['state','district']).any()
    tt=terms();tt.to_csv(out/'dated_terms.csv',index=False)
    roster=[]
    for r in maps.itertuples():
        pre=member_at(tt,r.state,r.district,pd.Timestamp('2010-12-15'))
        post=member_at(tt,r.state,r.district,pd.Timestamp('2011-01-15'))
        reason=('vacant or other party' if pre[0] not in ['Republican','Democrat'] or post[0] not in ['Republican','Democrat']
                else 'member or party changed' if pre[:2]!=post[:2] else '')
        a=tt[(tt.state==r.state)&tt.district.eq(r.district)&(tt.start<pd.Timestamp('2013-01-01'))&(tt.end>pd.Timestamp('2007-01-03'))].sort_values('start')
        cursor=pd.Timestamp('2007-01-03');continuous=True;opening_gap_days=0
        for v in a.itertuples():
            # Source terms sometimes start at the Jan4/5 swearing-in, not Jan3.
            # Only tolerate <=7 days at an odd-year January Congress opening.
            gap=(v.start-cursor).days
            opening=(0<gap<=7 and cursor.month==1 and cursor.day<=7 and cursor.year%2==1
                     and v.start.year==cursor.year and v.start.month==1 and v.start.day<=7)
            if opening:opening_gap_days+=gap
            if (v.start>cursor and not opening) or v.party!=post[0] or v.member!=post[1]:continuous=False
            cursor=max(cursor,v.end)
        if cursor<pd.Timestamp('2013-01-01'):continuous=False
        roster.append(dict(state=r.state,district=r.district,prior_party=pre[0],prior_member=pre[1],prior_name=pre[2],
            party=post[0],member=post[1],name=post[2],gainer=int(post[0]=='Republican'),exclusion=reason,
            continuous_member=continuous,opening_gap_days=opening_gap_days,unit=f'{r.state}-{r.district}'))
    roster=pd.DataFrame(roster)
    elections=pd.read_csv(ROOT/'data/MIT_election_lab/1976-2024-house.tab',low_memory=False)
    el=elections[elections.year.eq(2010)&elections.stage.eq('GEN')&~elections.special.fillna(False)].copy()
    # Fusion lines aggregate by candidate. Save inconsistencies, never silently relabel.
    votes=el.groupby(['state_po','district','candidate'],dropna=False).candidatevotes.sum().reset_index()
    winners=votes.sort_values('candidatevotes',ascending=False).drop_duplicates(['state_po','district'])
    party=el.sort_values('candidatevotes',ascending=False).drop_duplicates(['state_po','district','candidate'])
    winners=winners.merge(party[['state_po','district','candidate','party']],on=['state_po','district','candidate'],validate='1:1')
    winners=winners.rename(columns={'state_po':'state','party':'mit_party','candidate':'mit_candidate'})
    roster=roster.merge(winners[['state','district','mit_party','mit_candidate','candidatevotes']],on=['state','district'],how='left',validate='1:1')
    roster['mit_party_agrees']=roster.party.str.upper().eq(roster.mit_party)
    # Only the transition-stable eligible seats need an unambiguous validating winner.
    roster.loc[roster.exclusion.eq('')&~roster.mit_party_agrees,'exclusion']='MIT winner party disagrees or unavailable'
    roster.to_csv(out/'district_roster.csv',index=False)
    raw=pd.read_csv(ROOT/'data/RHG_PowerPlants_TS2026_20261106/plants_in_operation.csv')
    raw=raw[raw.Scenario.eq('ts_mid')].copy()
    cols=['EIA Plant ID','Latitude','Longitude','State Code']
    locations=raw[cols].drop_duplicates().reset_index(drop=True).assign(location_id=lambda d:np.arange(len(d)))
    raw=raw.merge(locations,on=cols,validate='m:1')
    valid=locations.Latitude.between(-90,90)&locations.Longitude.between(-180,180)&~(locations.Latitude.eq(0)&locations.Longitude.eq(0))
    pts=gp.GeoDataFrame(locations[valid],geometry=gp.points_from_xy(locations.loc[valid,'Longitude'],locations.loc[valid,'Latitude']),crs=4326).to_crs(4269)
    j=gp.sjoin(pts,maps,how='left',predicate='intersects')
    j['reason']=np.select([j.location_id.duplicated(keep=False),j.state.isna(),j['State Code'].ne(j.state)],
                          ['multiple polygons','outside polygons','state mismatch'],default='matched')
    j=j.drop_duplicates('location_id')
    loc=locations.merge(j[['location_id','state','district','reason']],on='location_id',how='left',validate='1:1')
    loc.reason=loc.reason.fillna('invalid coordinates')
    # Approximate near-boundary distance in equal-area meters; only a warning.
    pp=j.to_crs(6933);pol=maps.to_crs(6933)
    near={r.location_id: r.geometry.distance(pol.geometry.iloc[int(r.index_right)].boundary)
          for r in pp.itertuples() if pd.notna(r.index_right)}
    loc['boundary_distance_m']=loc.location_id.map(near)
    loc.to_csv(out/'plant_geography.csv',index=False)
    f=raw.merge(loc[loc.reason.eq('matched')][['location_id','state','district']],on='location_id',validate='m:1')
    national=raw[raw.Year.between(2006,2012)].groupby(['Year','Fuel Category'])['Capacity (MW)'].sum().rename('national_mw')
    mapped=f[f.Year.between(2006,2012)].groupby(['Year','Fuel Category'])['Capacity (MW)'].sum().rename('mapped_mw')
    pd.concat([national,mapped],axis=1).to_csv(out/'capacity_geography_coverage.csv')
    # Operating sites and positive plant-level stock changes are support metrics,
    # not announcements, gross construction, or independent treatment assignments.
    plant=f[f['Fuel Category'].isin(['Wind','Solar'])&f.Year.between(2006,2012)].groupby(
        ['state','district','EIA Plant ID','Year'])['Capacity (MW)'].sum().unstack('Year').fillna(0)
    plant=plant.reindex(columns=range(2006,2013),fill_value=0)
    ps=plant.reset_index()[['state','district','EIA Plant ID']].copy()
    ps['operating_post']=(plant[[2011,2012]].to_numpy()>0).any(axis=1)
    ps['positive_net_change_post']=((plant[2011]-plant[2010]>1e-6)|(plant[2012]-plant[2011]>1e-6)).to_numpy()
    ps['net_mw_2011_2012']=(plant[2012]-plant[2010]).to_numpy()
    ps.to_csv(out/'plant_support.csv',index=False)
    clean=f[f['Fuel Category'].isin(['Wind','Solar'])&f.Year.between(2006,2012)]
    agg=clean.groupby(['state','district','Year'])['Capacity (MW)'].sum().rename('mw')
    grid=roster[['state','district','gainer','unit']].merge(pd.DataFrame({'Year':range(2006,2013)}),how='cross')
    grid=grid.merge(agg,on=['state','district','Year'],how='left',validate='1:1').fillna({'mw':0}).sort_values(['unit','Year'])
    grid['y']=grid.groupby('unit').mw.diff()
    annual=grid[grid.Year.ge(2007)].rename(columns={'Year':'year'}).assign(outcome='wind_solar_net_mw')
    assert len(annual)==435*6 and annual.y.notna().all()
    annual.to_csv(out/'all_district_years.csv',index=False)
    # Features fixed at end 2009; 2010 held out from matching as an anticipation/placebo year.
    base=f[f.Year.eq(2009)]
    c=base[base['Fuel Category'].isin(['Wind','Solar'])].groupby(['state','district'])['Capacity (MW)'].sum()
    fossil=base[base['Fuel Category'].isin(['Coal','Natural Gas','Oil'])].groupby(['state','district'])['Capacity (MW)'].sum()
    area=maps.to_crs(6933).set_index(['state','district']).geometry.area/1e6
    features=['log_clean_capacity','log_fossil_capacity','log_core_area','prior_net_mw','prior_net_mw_slope']
    allunits=[];allpairs=[];ledgers=[];support=[];diagnostics=[];events=[];joint=[]
    for label,continuous in [('transition_stable',False),('continuous_member',True)]:
        u=roster[roster.exclusion.eq('') & (roster.continuous_member if continuous else True)].copy()
        support.append(dict(sample=label,scope='all_eligible',districts=len(u),states=u.state.nunique(),R=int(u.gainer.sum()),D=int((1-u.gainer).sum())))
        mixed=u.groupby('state').gainer.nunique();u=u[u.state.isin(mixed[mixed.eq(2)].index)].copy()
        u=u.set_index(['state','district']).assign(log_clean_capacity=np.log1p(c),log_fossil_capacity=np.log1p(fossil),log_core_area=np.log1p(area)).fillna({'log_clean_capacity':0,'log_fossil_capacity':0}).reset_index()
        a=annual[annual.unit.isin(u.unit)].copy()
        pre=a[a.year.le(2009)].pivot(index='unit',columns='year',values='y')
        u['prior_net_mw']=u.unit.map(np.arcsinh(pre.mean(axis=1)))
        u['prior_net_mw_slope']=u.unit.map(np.arcsinh((pre[2009]-pre[2007])/2))
        u=add_outcomes(u,a,list(range(2007,2011)),[2011,2012])
        config='audit2011_'+label;u['config']=config
        support.append(dict(sample=label,scope='mixed_states',districts=len(u),states=u.state.nunique(),R=int(u.gainer.sum()),D=int((1-u.gainer).sum()),
            active_R=int(((u.wind_solar_net_mw_post>0)&u.gainer.eq(1)).sum()),active_D=int(((u.wind_solar_net_mw_post>0)&u.gainer.eq(0)).sum())))
        for version in ['delta','trend_adjusted','placebo']:
            diagnostics.append(dict(config=config,outcome='wind_solar_net_mw',version=version,**diagnostic_fit(u,'wind_solar_net_mw_'+version)))
        pregrid=a.rename(columns={'y':'net_mw'})
        joint.append(dict(config=config,**joint_pre(pregrid,'net_mw',2011)))
        ref=a[a.year.eq(2010)].set_index('unit').y
        for year,g in a.groupby('year'):
            g=g.copy();g['change']=g.y-g.unit.map(ref)
            events.append(dict(config=config,year=year,**diagnostic_fit(g,'change')))
        pairs,ledger=cardinality(u,features,features,out,config)
        allunits.append(u);allpairs.append(pairs);ledgers.append(ledger)
    units=pd.concat(allunits);pairs=pd.concat(allpairs)
    units.to_csv(out/'units.csv',index=False);pairs.to_csv(out/'pairs.csv',index=False)
    pd.DataFrame(ledgers).to_csv(out/'matching_summary.csv',index=False)
    pd.DataFrame(support).to_csv(out/'support.csv',index=False)
    pd.DataFrame(diagnostics).to_csv(out/'unmatched_diagnostics.csv',index=False)
    pd.DataFrame(events).to_csv(out/'event_study.csv',index=False)
    pd.DataFrame(joint).to_csv(out/'joint_pretests.csv',index=False)
    return units,pairs,annual,features
