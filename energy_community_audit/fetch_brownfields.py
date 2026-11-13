"""Freeze an EPA known-property point screen. This is not a tax-eligibility database."""
import csv
import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

URL='https://services.arcgis.com/cJ9YHowT8TU7DUyn/arcgis/rest/services/Cleanups_in_my_Community_Sites/FeatureServer/0/query'
def query(**params):
    with urlopen(URL+'?'+urlencode(dict(f='json',**params)),timeout=45) as response:
        result=json.load(response)
    if 'error' in result: raise RuntimeError(result['error'])
    return result

if __name__=='__main__':
    ids=query(where='BF_PROPERTY_NAME IS NOT NULL',returnIdsOnly='true')
    key=ids['objectIdFieldName'];ids=sorted(ids['objectIds'])
    rows=[]
    for i in range(0,len(ids),1500):
        batch=ids[i:i+1500]
        page=query(where=f'BF_PROPERTY_NAME IS NOT NULL AND {key} BETWEEN {batch[0]} AND {batch[-1]}',outFields=key,returnGeometry='true',outSR=4326)
        for f in page['features']:
            point=f.get('geometry') or {}
            if 'x' in point:rows.append((f['attributes'][key],point['x'],point['y']))
        assert len(page['features'])==len(batch)
    assert len({r[0] for r in rows})==len(rows)
    with (Path(__file__).parent/'inputs/epa_brownfields.csv').open('w') as out:
        writer=csv.writer(out);writer.writerow(['id','lon','lat']);writer.writerows(rows)
    print(len(rows),'geolocated properties from',len(ids),'records')
