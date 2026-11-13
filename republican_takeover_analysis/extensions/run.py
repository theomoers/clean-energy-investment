"""One-command rebuild. Refuses to overwrite an output directory."""
from pathlib import Path
import argparse,os,sys,subprocess,hashlib,json
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/majority-mpl')
from common import ROOT,OLD,AUDIT
import prepare2023,analyze,audit2011
assert Path(analyze.__file__).parent == Path(__file__).parent


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--bootstrap-reps',type=int,default=999)
    args=parser.parse_args();out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    u,p,a=prepare2023.main(out);a.to_csv(out/'annual2023.csv',index=False)
    print('2023 matching complete',len(p),'pairs',flush=True)
    u11,p11,a11,f11=audit2011.main(out/'audit2011')
    print('2011 audit complete',p11.groupby('config').size().to_dict(),flush=True)
    analyze.main(out,u,p,a,u11,p11,a11,f11,args.bootstrap_reps)
    subprocess.run(['/usr/local/bin/Rscript','--vanilla',str(Path(__file__).with_name('forest.R')),str(out)],check=True)
    import finalize
    finalize.main(out)
    sources=[OLD/'units.csv',OLD/'pairs.csv',OLD/'annual.csv',AUDIT/'latest/facility_geography.csv',
        ROOT/'data/CleanInvestmentMonitor_Download/manufacturing_energy_and_industry_facility_metadata.csv',
        ROOT/'data/RHG_PowerPlants_TS2026_20261106/plants_in_operation.csv',
        ROOT/'data/MIT_election_lab/1976-2024-house.tab',ROOT/'data/redesign_raw/boundaries/tl_2011_us_cd112.zip']
    sources+=sorted((ROOT/'data/redesign_raw/legislators').glob('*.json'))
    sources+=sorted(Path(__file__).parent.glob('*.py'))+sorted(Path(__file__).parent.glob('*.R'))+[Path(__file__).with_name('PROTOCOL.txt')]
    import pandas as pd
    pd.DataFrame([dict(file=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sources]).to_csv(out/'input_code_hashes.csv',index=False)
    (out/'run_settings.json').write_text(json.dumps(dict(bootstrap_reps=args.bootstrap_reps,seed=11481,python=sys.version),indent=2))
    print('Completed',out,flush=True)

if __name__=='__main__':main()
