"""Geographic joins and transformations for the fixed district cores."""
from pathlib import Path
import argparse,json,sys,zipfile
import geopandas as gp
import numpy as np
import pandas as pd
import shapely

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];RAW=HERE/'raw'
SOURCE=ROOT/'republican_takeover_analysis/overlap_forest/results_current/areas.csv'
sys.path.insert(0,str(ROOT/'revision_analysis'))
from panel import geographies

def cores():
 if (HERE/'cores.gpkg').exists():return gp.read_file(HERE/'cores.gpkg')
 u=pd.read_csv(SOURCE)
 maps={n:g.to_crs(6933).assign(geometry=lambda x:x.geometry.make_valid().simplify(50)).set_index(['state','district']) for n,g in geographies().items()}
 geoms=[]
 for r in u.itertuples():
  geom=maps[118].loc[(r.state,r.district),'geometry']
  for n in [116,117]:geom=geom.intersection(maps[n].loc[(r.state,int(getattr(r,f'old{n}'))),'geometry'])
  assert np.isclose(geom.area/1e6,r.core_area_km2,rtol=1e-9)
  geoms.append(geom)
 g=gp.GeoDataFrame(u[['unit','state','core_area_km2']],geometry=geoms,crs=6933)
 g.to_file(HERE/'cores.gpkg',driver='GPKG');return g

def tract_crosswalk(c):
 p=HERE/'tract_core_crosswalk.csv'
 if p.exists():return pd.read_csv(p,dtype={'GEOID':str})
 tr=gp.read_file(ROOT/'energy_community_audit/inputs/tracts.zip').to_crs(c.crs)
 tr.geometry=tr.geometry.make_valid()
 left,right=tr.sindex.query(c.geometry,predicate='intersects')
 area=shapely.area(shapely.intersection(c.geometry.values[left],tr.geometry.values[right]))
 keep=area>1.;left=left[keep];right=right[keep];area=area[keep]
 x=pd.DataFrame(dict(unit=c.unit.values[left],GEOID=tr.GEOID.values[right],
                     intersect_km2=area/1e6,tract_fraction=area/tr.geometry.area.values[right],
                     core_fraction=area/c.geometry.area.values[left]))
 assert x.tract_fraction.max()<1.000001
 x.to_csv(p,index=False);return x

def socio(x):
 wanted={'b01003':[1],'b17001':[1,2],'b15003':[1,22,23,24,25],
         'b23025':[3,5],'b19025':[1],'b11001':[1]}
 pieces=[]
 for table,indices in wanted.items():
  cols=[f'{table.upper()}_E{i:03}' for i in indices]
  d=pd.read_csv(RAW/f'acsdt5y2021-{table}.dat',sep='|',usecols=['GEO_ID']+cols)
  d=d[d.GEO_ID.str.startswith('1400000US')].copy()
  d['GEOID']=d.GEO_ID.str[-11:];d=d.drop(columns='GEO_ID')
  assert d.GEOID.is_unique
  d[cols]=d[cols].apply(pd.to_numeric,errors='coerce').mask(d[cols].lt(0))
  pieces.append(d.set_index('GEOID'))
 d=pd.concat(pieces,axis=1).reset_index()
 j=x.merge(d,on='GEOID',how='left',validate='m:1')
 rows=[]
 for unit,g in j.groupby('unit'):
  row=dict(unit=unit,acs_area_coverage=g.loc[g.B01003_E001.notna(),'core_fraction'].sum(),
           tract_count=len(g),partial_tracts=int(g.tract_fraction.between(1e-5,.99999).sum()))
  for col in d.columns.drop('GEOID'):
   # No imputation of missing sentinel values as zeros.
   row[col]=(g[col]*g.tract_fraction).sum(min_count=1)
   denom=(g.B01003_E001*g.tract_fraction).sum()
   row[col+'_population_coverage']=(g.loc[g[col].notna(),'B01003_E001']*g.loc[g[col].notna(),'tract_fraction']).sum()/denom
  row['income_observed_households']=(g.B11001_E001.where(g.B19025_E001.notna())*g.tract_fraction).sum(min_count=1)
  rows.append(row)
 s=pd.DataFrame(rows).set_index('unit')
 s['population_2021']=s.B01003_E001
 s['poverty_share']=s.B17001_E002/s.B17001_E001
 s['unemployment_share']=s.B23025_E005/s.B23025_E003
 s['college_share']=s[[f'B15003_E{i:03}' for i in [22,23,24,25]]].sum(axis=1)/s.B15003_E001
 s['mean_household_income_2021']=s.B19025_E001/s.income_observed_households
 s.to_csv(HERE/'acs_core_audit.csv')
 return s

def resources(c):
 out=c[['unit']].copy().set_index('unit')
 from shapely.geometry import box
 for parameter,name,dx,dy in [('ALLSKY_SFC_SW_DWN','solar_kwh_m2_day',1.,1.),('WS50M','wind50m_ms',.625,.5)]:
  cells={}
  for p in sorted(RAW.glob(f'nasa_{parameter}_*.json')):
   d=json.loads(p.read_text());assert d['header']['end']=='20211231'
   for f in d['features']:
    lon,lat=f['geometry']['coordinates'][:2]
    values=f['properties']['parameter'][parameter]
    months={k:v for k,v in values.items() if len(k)==6 and k[-2:]!='13'}
    assert len(months)==144 and min(months.values())>=0
    # Weight monthly daily-average resources by days, including leap years.
    days=np.array([pd.Period(k[:4]+'-'+k[4:],freq='M').days_in_month for k in months])
    avg=float(np.average(list(months.values()),weights=days))
    key=(lon,lat)
    if key in cells:assert np.isclose(cells[key],avg)
    cells[key]=avg
  g=gp.GeoDataFrame({'value':list(cells.values())},geometry=[box(a-dx/2,b-dy/2,a+dx/2,b+dy/2) for a,b in cells],crs=4326).to_crs(c.crs)
  left,right=g.sindex.query(c.geometry,predicate='intersects')
  area=shapely.area(shapely.intersection(c.geometry.values[left],g.geometry.values[right]))
  j=pd.DataFrame(dict(unit=c.unit.values[left],area=area,num=area*g.value.values[right]))
  a=j.groupby('unit')[['area','num']].sum()
  out[name]=a.num/a.area
  out[name+'_coverage']=a.area/c.set_index('unit').geometry.area
  assert out[name+'_coverage'].between(.99999,1.00001).all(),out[name+'_coverage'].describe()
  g.to_file(HERE/f'{name}_cells.gpkg',driver='GPKG')
 out.to_csv(HERE/'resource_core_audit.csv');return out

def grid(c):
 from scipy.spatial import cKDTree
 from shapely.geometry import box
 z=zipfile.ZipFile(RAW/'eia8602021.zip')
 plants=pd.read_excel(z.open('2___Plant_Y2021.xlsx'),skiprows=1)
 gen=pd.read_excel(z.open('3_1_Generator_Y2021.xlsx'),sheet_name='Operable',skiprows=1)
 ids=set(gen.loc[gen['Operating Year'].le(2021)&gen.Status.isin(['OP','SB','OS','OA']),'Plant Code'])
 plants=plants[plants['Plant Code'].isin(ids)].copy()
 v=plants[['Grid Voltage (kV)','Grid Voltage 2 (kV)','Grid Voltage 3 (kV)']].apply(pd.to_numeric,errors='coerce')
 plants['max_grid_kv']=v.mask(~v.between(1,1200) if hasattr(v,'between') else ~((v>=1)&(v<=1200))).max(axis=1)
 for col in ['Latitude','Longitude']:plants[col]=pd.to_numeric(plants[col],errors='coerce')
 plants['valid_location']=plants.Latitude.between(20,51)&plants.Longitude.between(-130,-60)
 plants['used_hv_node']=plants.max_grid_kv.ge(115)&plants.valid_location
 plants[['Plant Code','State','Latitude','Longitude','max_grid_kv','valid_location','used_hv_node']].to_csv(HERE/'grid_node_audit.csv',index=False)
 nodes=plants[plants.used_hv_node]
 points=gp.GeoDataFrame(nodes,geometry=gp.points_from_xy(nodes.Longitude,nodes.Latitude),crs=4326).to_crs(5070)
 tree=cKDTree(np.column_stack([points.geometry.x,points.geometry.y]))
 rows=[];samples=[]
 # Five-km equal-area quadrature, polygon-clipped cell representative points.
 projected=c.to_crs(5070);projected.geometry=projected.geometry.make_valid()
 for r in projected.itertuples():
  x0,y0,x1,y1=r.geometry.bounds
  cells=[box(x,y,x+5000,y+5000) for x in np.arange(np.floor(x0/5000)*5000,x1,5000) for y in np.arange(np.floor(y0/5000)*5000,y1,5000)]
  cuts=shapely.intersection(np.array(cells,dtype=object),r.geometry)
  area=shapely.area(cuts);cuts=cuts[area>1];area=area[area>1]
  p=shapely.point_on_surface(cuts);xx=shapely.get_x(p);yy=shapely.get_y(p)
  dist=tree.query(np.column_stack([xx,yy]))[0]/1000
  assert np.isclose(area.sum(),r.geometry.area,rtol=1e-5)
  rows.append(dict(unit=r.unit,mean_distance_hv_node_km=np.average(dist,weights=area),
                   area_within25km_hv_node=np.average(dist<=25,weights=area),grid_quadrature_cells=len(area)))
  samples.extend(dict(unit=r.unit,x5070=a,y5070=b,area_km2=w/1e6,distance_hv_node_km=d) for a,b,w,d in zip(xx,yy,area,dist))
 pd.DataFrame(samples).to_csv(HERE/'grid_quadrature.csv',index=False)
 out=pd.DataFrame(rows).set_index('unit');out.to_csv(HERE/'grid_core_audit.csv');return out

def ira(x):
 tr=gp.read_file(ROOT/'energy_community_audit/inputs/tracts.zip')[['GEOID','geometry']]
 cc=pd.read_excel(RAW/'EC_CC_V1.xlsx',skiprows=1)
 direct=set(cc.loc[cc.iloc[:,3].str.contains('Mine closure|Generating unit retirement',case=False,regex=True),cc.columns[2]].astype(int).astype(str).str.zfill(11))
 closures=pd.read_csv(ROOT/'energy_community_audit/results/dated_closures.csv',dtype={'GEOID':str})
 dated=set(closures.loc[closures.date.lt('2022-01-01'),'GEOID'])&direct
 # Statutory adjacency includes corner contacts. No post-2021 closure enters.
 inside=tr[tr.GEOID.isin(dated)]
 _,r=tr.sindex.query(inside.geometry,predicate='intersects')
 eligible=set(tr.GEOID.iloc[r])|dated
 pd.DataFrame({'GEOID':sorted(eligible)}).to_csv(HERE/'pre2022_coal_exposed_tracts.csv',index=False)
 ffe=pd.read_excel(RAW/'EC_MSA_FFE_V1.xlsx',skiprows=1)
 counties=set(ffe.iloc[:,0].astype(int).astype(str).str.zfill(2)+ffe.iloc[:,1].astype(int).astype(str).str.zfill(3))
 cbp=pd.read_csv(RAW/'cbp20co.zip',dtype={'fipstate':str,'fipscty':str,'naics':str})
 cbp['county']=cbp.fipstate.str.zfill(2)+cbp.fipscty.str.zfill(3)
 subset=cbp[cbp.naics.isin(['335911','335912','334413'])]
 subset.to_csv(HERE/'cbp_component_related_counties.csv',index=False)
 est=subset.groupby('county').est.sum()
 trarea=tr.to_crs(6933).assign(county=lambda d:d.GEOID.str[:5])
 countyarea=trarea.groupby('county').geometry.apply(lambda s:s.area.sum()/1e6)
 d=x.copy();d['coal']=d.GEOID.isin(eligible);d['ffe']=d.GEOID.str[:5].isin(counties)
 d['industrial_density']=d.GEOID.str[:5].map(est).fillna(0)/d.GEOID.str[:5].map(countyarea)*1000
 d['ira_coal_area_share']=d.core_fraction*d.coal
 d['ira_ffe_area_share']=d.core_fraction*d.ffe
 d['component_related_est_per1000km2']=d.core_fraction*d.industrial_density
 out=d.groupby('unit')[['ira_coal_area_share','ira_ffe_area_share','component_related_est_per1000km2']].sum()
 out.to_csv(HERE/'ira_core_audit.csv');return out

def main():
 p=argparse.ArgumentParser();p.add_argument('--cores-only',action='store_true');a=p.parse_args()
 c=cores();print('Verified fixed cores:',len(c),flush=True)
 if a.cores_only:return
 x=tract_crosswalk(c);print('Tract intersections:',len(x),flush=True)
 parts=[]
 for name,fn in [('acs',lambda:socio(x)),('resources',lambda:resources(c)),('grid',lambda:grid(c)),('ira',lambda:ira(x))]:
  parts.append(fn());print(name,'complete',flush=True)
 d=pd.concat(parts,axis=1)
 d['log_population_density']=np.log1p(d.population_2021/c.set_index('unit').core_area_km2)
 d['log_mean_household_income']=np.log(d.mean_household_income_2021)
 d['log_distance_hv_node']=np.log1p(d.mean_distance_hv_node_km)
 d['log_component_related_density']=np.log1p(d.component_related_est_per1000km2)
 features=['log_population_density','poverty_share','unemployment_share','college_share','log_mean_household_income',
           'solar_kwh_m2_day','wind50m_ms','log_distance_hv_node','area_within25km_hv_node',
           'ira_coal_area_share','ira_ffe_area_share','log_component_related_density']
 assert np.isfinite(d[features]).all().all()
 assert d.acs_area_coverage.gt(.99).all(),d.acs_area_coverage.describe()
 d.reindex(pd.read_csv(SOURCE).unit).to_csv(HERE/'covariates.csv')
 (HERE/'features.json').write_text(json.dumps(features,indent=2)+'\n')
 print(d[features].describe().to_string(),flush=True)

if __name__=='__main__':main()
