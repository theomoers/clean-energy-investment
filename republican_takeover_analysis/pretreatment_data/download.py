"""Download immutable public source responses. No outcomes are inspected."""
from pathlib import Path
import argparse, concurrent.futures as cf, hashlib, json, time
import requests

HERE=Path(__file__).resolve().parent
RAW=HERE/'raw'
SOURCES={
 'eia8602021.zip':('https://www.eia.gov/electricity/data/eia860/archive/xls/eia8602021.zip','2021','2021 EIA-860 annual archive; historical revised file'),
 'EC_CC_V1.xlsx':('https://home.treasury.gov/system/files/131/EC_CC_V1.xlsx','through 2023-02-07','Use only separately dated closures before 2022; designation published 2023'),
 'EC_MSA_FFE_V1.xlsx':('https://home.treasury.gov/system/files/131/EC_MSA_FFE_V1.xlsx','2010-2020 employment','2023 reconstruction from CBP facts released by April 2022; partial eligibility only'),
 'cbp20co.zip':('https://www2.census.gov/programs-surveys/cbp/datasets/2020/cbp20co.zip','2020','2020 County Business Patterns; broad battery/semiconductor industrial footprint'),
}
for table in ['b01003','b17001','b15003','b23025','b19025','b11001']:
 SOURCES[f'acsdt5y2021-{table}.dat']=(f'https://www2.census.gov/programs-surveys/acs/summary_file/2021/table-based-SF/data/5YRData/acsdt5y2021-{table}.dat','2017-2021','Official ACS table-based summary file, released December 2022')

def get(item):
 name,(url,period,note)=item;p=RAW/name
 if not p.exists():
  for attempt in range(4):
   try:
    r=requests.get(url,timeout=180,headers={'User-Agent':'CleanInvestmentPreTreatmentResearch/1.0'})
    r.raise_for_status()
    if name.endswith('.zip'):assert r.content[:2]==b'PK',(name,r.text[:100])
    if name.endswith('.xlsx'):assert r.content[:2]==b'PK'
    if name.endswith('.json'):r.json()
    p.write_bytes(r.content);break
   except Exception:
    if attempt==3:raise
    time.sleep(2*(attempt+1))
 print(name,p.stat().st_size,flush=True)
 return dict(file=name,url=url,reference_period=period,note=note,
             retrieved_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime(p.stat().st_mtime)),
             bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--resources',action='store_true');a=parser.parse_args()
 RAW.mkdir(parents=True,exist_ok=True)
 sources=dict(SOURCES)
 if a.resources:
  # Build requests from the fixed core bounding boxes. 5-degree tiles are
  # independent of facilities/outcomes, with native cell coverage checked later.
  import geopandas as gp
  cores=gp.read_file(HERE/'cores.gpkg').to_crs(4326)
  tiles=set()
  import math
  for g in cores.geometry:
   x0,y0,x1,y1=g.bounds
   for x in range(math.floor(x0/5)*5,math.floor(x1/5)*5+1,5):
    for y in range(math.floor(y0/5)*5,math.floor(y1/5)*5+1,5):
     from shapely.geometry import box
     if g.intersects(box(x,y,x+5,y+5)):tiles.add((x,y))
  sources={}
  for x,y in sorted(tiles):
   for param in ['ALLSKY_SFC_SW_DWN','WS50M']:
    name=f'nasa_{param}_{x}_{y}.json'
    url=('https://power.larc.nasa.gov/api/temporal/monthly/regional?'
         f'latitude-min={y}&latitude-max={y+5}&longitude-min={x}&longitude-max={x+5}'
         f'&parameters={param}&community=RE&start=2010&end=2021&format=JSON')
    sources[name]=(url,'2010-2021','NASA POWER retrospective physical resource, monthly regional native grid')
 with cf.ThreadPoolExecutor(max_workers=3) as pool:rows=list(pool.map(get,sources.items()))
 old=json.loads((HERE/'download_manifest.json').read_text()) if (HERE/'download_manifest.json').exists() else []
 byname={r['file']:r for r in old};byname.update({r['file']:r for r in rows})
 (HERE/'download_manifest.json').write_text(json.dumps(list(byname.values()),indent=2)+'\n')

if __name__=='__main__':main()
