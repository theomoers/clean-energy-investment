"""One command, fresh destination, both releases and historical power audit."""
import argparse
from pathlib import Path
import sys
import hashlib
import pandas as pd
import importlib.util
import audit
import power_plants
import supplement
import influence
spec=importlib.util.spec_from_file_location('majority_verify',Path(__file__).with_name('verify.py'))
verify=importlib.util.module_from_spec(spec)
spec.loader.exec_module(verify)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',required=True,type=Path)
    parser.add_argument('--resume',action='store_true',help='Reuse completed announcement stages only after verifying their input hashes')
    parser.add_argument('--verify-only',action='store_true',help='Verify a completed build without rebuilding it')
    args=parser.parse_args()
    base=args.out.resolve()
    if args.verify_only:
        assert base.is_dir()
        verify.HERE=base;verify.main();return
    base.mkdir(parents=True,exist_ok=args.resume)
    source=audit.ROOT/'data/CleanInvestmentMonitor_Download/manufacturing_energy_and_industry_facility_metadata.csv'
    for name in ['frozen','latest','replication_latest']:
        manifest=base/name/'input_hashes.csv'
        if args.resume and manifest.exists():
            for row in pd.read_csv(manifest).itertuples():
                assert hashlib.sha256((audit.ROOT/row.file).read_bytes()).hexdigest()==row.sha256,row.file
            print('Reusing verified inputs for',name,flush=True)
            continue
        print('Building',name,flush=True)
        sys.argv=['audit.py','--out',str(base/name)]
        if name!='frozen':
            sys.argv+=['--facilities',str(source),'--crosswalk-cache',str(base/'frozen/geographic_crosswalk.csv')]
        audit.main()
    power_plants.HERE=base;power_plants.OUT=base/'power_plants'
    supplement.HERE=base;influence.HERE=base;verify.HERE=base
    power_plants.main();supplement.main();influence.main();verify.main()
    print('Verified audit saved to',base,flush=True)


if __name__=='__main__':main()
