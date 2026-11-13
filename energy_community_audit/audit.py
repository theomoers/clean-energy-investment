"""Support and oracle-power audit of the deployment bonus, frozen CIM 2024Q4."""
from pathlib import Path
import hashlib
import re
import geopandas as gp
import networkx as nx
import numpy as np
import pandas as pd
import shapely
from scipy.stats import t as student

ROOT = Path(__file__).resolve().parent
I, O = ROOT/'inputs', ROOT/'results'
O.mkdir(exist_ok=True)


def sheet(name):
    return pd.read_excel(I/(name+'.xlsx'),skiprows=1)


def dated_cores(geo, types):
    mines=pd.read_csv(I/'mines.zip',sep='|',encoding='latin1',usecols=[
        'MINE_ID','COAL_METAL_IND','CURRENT_MINE_STATUS','CURRENT_STATUS_DT','CURRENT_MINE_TYPE','LONGITUDE','LATITUDE'])
    mines=mines[mines.COAL_METAL_IND.eq('C') & mines.CURRENT_MINE_STATUS.isin(['Abandoned','Abandoned and Sealed'])
                & mines.CURRENT_MINE_TYPE.isin(['Surface','Underground'])].copy()
    mines['date']=pd.to_datetime(mines.CURRENT_STATUS_DT,format='%m/%d/%Y',errors='coerce')
    mines=mines[mines.date.ge('2000-01-01')].rename(columns={'MINE_ID':'id','LATITUDE':'latitude','LONGITUDE':'longitude'})
    gen=pd.read_parquet(I/'core_eia860__scd_generators.parquet',columns=['plant_id_eia','generator_retirement_date',
        'energy_source_code_1','report_date'])
    for col in ['generator_retirement_date','report_date']:gen[col]=pd.to_datetime(gen[col])
    gen=gen[gen.energy_source_code_1.isin(['ANT','BIT','LIG','SUB','WC','RC']) &
            gen.generator_retirement_date.ge('2010-01-01') & gen.report_date.le('2023-12-31')]
    plants=gen.groupby('plant_id_eia').generator_retirement_date.min().reset_index().merge(
        pd.read_parquet(I/'core_eia__entity_plants.parquet',columns=['plant_id_eia','latitude','longitude']),on='plant_id_eia')
    plants=plants.rename(columns={'plant_id_eia':'id','generator_retirement_date':'date'})
    plants=plants[plants.date.lt('2022-08-01')]  # EIA retirement dates have monthly precision.
    points=pd.concat([mines.assign(kind='Mine closure'),plants.assign(kind='Generating unit retirement')])
    points=points[points.date.lt('2022-08-16')].dropna(subset=['latitude','longitude'])
    pts=gp.GeoDataFrame(points,geometry=gp.points_from_xy(points.longitude,points.latitude),crs=4326).to_crs(geo.crs)
    joins=gp.sjoin(pts,geo[['GEOID','geometry']],predicate='within')
    good=joins.apply(lambda r:r.kind in types.get(r.GEOID,''),axis=1)
    valid=joins.loc[good,['kind','id','date','GEOID']].drop_duplicates()
    valid.to_csv(O/'dated_closures.csv',index=False)
    return valid.groupby('GEOID').date.min()


def geography():
    geo=gp.read_file(I/'tracts.zip',columns=['GEOID','geometry']).sort_values('GEOID').reset_index(drop=True)
    geo=geo[~geo.GEOID.str[:2].isin(['72','78','60','66','69'])].reset_index(drop=True)
    assert geo.GEOID.is_unique
    v1=sheet('EC_CC_V1'); types=dict(zip(v1.iloc[:,2].astype(int).astype(str).str.zfill(11),v1.iloc[:,3]))
    v3=set(sheet('EC_CC_V3').iloc[:,2].astype(int).astype(str).str.zfill(11))
    assert set(types)<=v3
    historical=dated_cores(geo,types)
    county_lists=[]
    for y in (2022,2023):
        d=sheet(f'EC_MSA_FFE_U{y}_AllCounties')
        id=d.iloc[:,0].astype(int).astype(str).str.zfill(2)+d.iloc[:,1].astype(int).astype(str).str.zfill(3)
        county_lists.append(set(id[d.iloc[:,6]==0]))
    clean_counties=set.intersection(*county_lists)
    # Only one copy of each edge. Positive shared boundary excludes corner contacts.
    eligible=geo[geo.GEOID.isin(v3)]
    l,r=geo.sindex.query(eligible.geometry,predicate='intersects')
    gl=eligible.geometry.values[l]; gr=geo.geometry.values[r]
    keep=(eligible.index.values[l]!=r)
    l,r,gl,gr=l[keep],r[keep],gl[keep],gr[keep]
    shared=shapely.length(shapely.intersection(shapely.boundary(gl),shapely.boundary(gr)))>1e-7
    a,b=eligible.GEOID.values[l[shared]],geo.GEOID.values[r[shared]]
    e=pd.DataFrame({'inside':a,'outside':b})
    inner=e[e.outside.isin(v3)]
    outer=e[~e.outside.isin(v3)&e.inside.map(types).eq('Directly adjoining')].copy()
    outer=outer[outer.inside.str[:2]==outer.outside.str[:2]]
    # Components include contiguous designated tracts across state lines and shared eligible-design controls.
    graph=nx.from_pandas_edgelist(inner,'inside','outside'); graph.add_nodes_from(eligible.GEOID)
    graph.add_edges_from(outer.itertuples(index=False,name=None))
    labels={i:min(c) for c in nx.connected_components(graph) for i in c}
    outer['region']=outer.inside.map(labels)
    earliest=inner[inner.outside.isin(historical.index)].assign(date=lambda x:x.outside.map(historical)).groupby('inside').date.min()
    outer['closure_date']=outer.inside.map(earliest)
    outer['other_route_clean']=outer.inside.str[:5].isin(clean_counties)&outer.outside.str[:5].isin(clean_counties)
    outer['same_county']=outer.inside.str[:5]==outer.outside.str[:5]
    table=(I/'irs_48c_round1.html').read_text().split('APPENDIX C Section 48C(e) Energy Communities Census Tracts</strong>')[1].split('</table>')[0]
    manufacturing={n.zfill(11) for n in re.findall(r'>(\d{10,11})</td>',table)}
    assert len(manufacturing)==4184
    for side in ['inside','outside']:outer['manufacturing_setaside_'+side]=outer[side].isin(manufacturing)
    if (I/'epa_brownfields.csv').exists():
        bf=pd.read_csv(I/'epa_brownfields.csv')
        points=gp.GeoDataFrame(bf,geometry=gp.points_from_xy(bf.lon,bf.lat),crs=4326).to_crs(geo.crs)
        seen=set(gp.sjoin(points,geo[['GEOID','geometry']],predicate='intersects').GEOID)
        for side in ['inside','outside']:outer['known_brownfield_'+side]=outer[side].isin(seen)
    outer.to_csv(O/'boundary_edges.csv',index=False)
    return outer


def projects():
    f=pd.read_csv(ROOT.parent/'data/manufacturing_energy_and_industry_facility_metadata.csv',skiprows=5,
                  dtype={'census_tract_2020_geoid':'string'},low_memory=False)
    f=f[f.Segment.eq('Energy and Industry')&f.Technology.isin(['Solar','Wind','Storage'])&f.LatLon_Valid].copy()
    f['tract']=f.census_tract_2020_geoid.str.zfill(11)
    f=f[f.tract.str.fullmatch(r'\d{11}').fillna(False)].copy()
    assert f.unique_id.is_unique
    date=pd.to_datetime(f.Announcement_Date)
    f['period']=np.select([date.between('2018-01-01','2019-12-31'),date.between('2020-01-01','2021-12-31'),
                    date.between('2023-07-01','2024-12-31')],['early','late','post'],default='omit')
    return f[f.period!='omit']


def summarize(name,edges,f):
    arms=pd.concat([edges[[side,'region']].drop_duplicates().rename(columns={side:'tract'}).assign(side=side)
                    for side in ['inside','outside']],ignore_index=True)
    assert arms.tract.is_unique
    events=f.merge(arms,on='tract',validate='m:1')
    counts=events.groupby(['region','side','period']).size().unstack(fill_value=0)
    cells=arms.groupby(['region','side']).size().rename('tracts').to_frame().join(counts).fillna(0)
    for p in ['early','late','post']:
        if p not in cells: cells[p]=0
    cells['pre']=cells.early+cells.late
    cells['delta']=100*(cells.post/1.5-cells.pre/4)/cells.tracts
    cells['placebo']=100*(cells.late-cells.early)/(2*cells.tracts)
    r=cells.unstack('side');r.columns=['_'.join(i) for i in r.columns]
    r['did']=r.delta_inside-r.delta_outside
    r['pre_placebo']=r.placebo_inside-r.placebo_outside
    total=r.post_inside+r.post_outside
    if name=='primary':
        arms.to_csv(O/'primary_tracts.csv',index=False)
        events[['unique_id','Technology','Announcement_Date','tract','region','side','period']].to_csv(O/'primary_events.csv',index=False)
    r.to_csv(O/(name+'_regions.csv'))
    out=dict(spec=name,regions=len(r),edges=len(edges),states=arms.tract.str[:2].nunique(),
             post_either=(total>0).sum(),post_both=((r.post_inside>0)&(r.post_outside>0)).sum(),
             pre_both=((r.pre_inside>0)&(r.pre_outside>0)).sum(),
             both_periods_both=((r.pre_inside*r.pre_outside*r.post_inside*r.post_outside)>0).sum(),
             event_equivalent_regions=total.sum()**2/(total**2).sum() if total.sum() else 0,
             largest_region_event_share=total.max()/total.sum())
    for side in ['inside','outside']:
        for col in ['tracts','early','late','pre','post']:out[col+'_'+side]=int(r[col+'_'+side].sum())
        out['post_regions_'+side]=int(r['post_'+side].gt(0).sum())
    return out,r


def power(r,fraction,weighting):
    # Oracle high/low label supplied for free. Exactly half the independent regions held out.
    # Actual region-level change contrasts supply empirical noise, randomly permuted under the null.
    rng=np.random.default_rng(48045)
    d=r.did.to_numpy();n=int(len(d)*fraction);k=n//2
    orders=rng.random((5000,len(d))).argsort(axis=1)[:,:n]
    a,b=d[orders[:,:k]],d[orders[:,k:]]
    weights=(np.ones(len(r)) if weighting=='equal_region' else
             (1/(1/r.tracts_inside+1/r.tracts_outside)).to_numpy())
    def moments(values,w):
        w=w/w.sum(axis=1,keepdims=True)
        avg=(w*values).sum(axis=1)
        v=(w**2*(values-avg[:,None])**2).sum(axis=1)/(1-(w**2).sum(axis=1))
        return avg,v,1/(w**2).sum(axis=1)
    ma,va,na=moments(a,weights[orders[:,:k]])
    mb,vb,nb=moments(b,weights[orders[:,k:]])
    se=np.sqrt(va+vb);diff=ma-mb
    with np.errstate(divide='ignore',invalid='ignore'):
        df=(va+vb)**2/(va**2/(na-1)+vb**2/(nb-1))
        threshold=student.ppf(.975,df)*se
    baseline=100*r.post_outside.sum()/r.tracts_outside.sum()/1.5
    def rejection(effect):return np.mean(np.abs(diff+effect)>threshold)
    hi=max(10.,baseline)
    while rejection(hi)<.8 and hi<10000:hi*=2
    lo=0
    for _ in range(40):
        mid=(hi+lo)/2
        if rejection(mid)>=.8:hi=mid
        else:lo=mid
    return dict(fraction=fraction,weights=weighting,validation_regions=n,high=k,low=n-k,control_announcements_per_100_tract_years=baseline,
                equal_region_control_rate=100*(r.post_outside/r.tracts_outside).mean()/1.5,
                null_rejection=rejection(0),power_at_half_baseline=rejection(.5*baseline),
                power_at_full_baseline=rejection(baseline),mde80=hi,mde_as_multiple=hi/baseline)


if __name__=='__main__':
    edge=geography();f=projects()
    masks={'geographic_ceiling':np.ones(len(edge),dtype=bool),'clean_counties':edge.other_route_clean,
           'primary':edge.other_route_clean & edge.closure_date.notna(),
           'same_county':edge.other_route_clean & edge.closure_date.notna() & edge.same_county,
           'pre2018':edge.other_route_clean & edge.closure_date.lt('2018-01-01')}
    if 'known_brownfield_inside' in edge:
        masks['brownfield_screen']=masks['primary'] & ~edge.known_brownfield_inside & ~edge.known_brownfield_outside
    rows=[];pows=[]
    for name,mask in masks.items():
        out,r=summarize(name,edge.loc[mask],f);rows.append(out)
        for fraction in [.5,1.]:
            for weight in ['equal_region','tract_precision']:
                pows.append(dict(spec=name,**power(r,fraction,weight)))
    pd.DataFrame(rows).to_csv(O/'support.csv',index=False)
    pd.DataFrame(pows).to_csv(O/'oracle_power.csv',index=False)
    paths=[*sorted(I.iterdir()),ROOT.parent/'data/manufacturing_energy_and_industry_facility_metadata.csv']
    pd.DataFrame([(p.name,hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths],columns=['file','sha256']).to_csv(O/'inputs.csv',index=False)
    print(pd.DataFrame(rows).to_string(index=False));print(pd.DataFrame(pows).to_string(index=False))
