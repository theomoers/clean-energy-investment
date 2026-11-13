"""One-command rebuild from frozen public inputs; optionally fetch missing raw files."""
from pathlib import Path
import argparse,subprocess,sys
HERE=Path(__file__).resolve().parent
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
 p.add_argument('--from-derived',action='store_true',help='Use the included frozen covariates and area panel; do not rebuild raw-data joins.')
 p.add_argument('--download',action='store_true');p.add_argument('--bootstrap-reps',type=int,default=999);a=p.parse_args()
 if a.from_derived and a.download:p.error('--from-derived and --download are mutually exclusive')
 if a.out.exists():p.error('Choose a fresh output directory; existing results will not be overwritten.')
 def run(script,*args):subprocess.run([sys.executable,str(HERE/script),*map(str,args)],check=True)
 if a.download:
  run('download.py');run('build.py','--cores-only');run('download.py','--resources')
 if not a.from_derived:run('build.py')
 run('analyze.py','--out',a.out,'--bootstrap-reps',a.bootstrap_reps)
 run('diagnostics.py',a.out,'--bootstrap-reps',a.bootstrap_reps)
 run('regressions.py',a.out)
 run('verify.py',a.out,*(['--skip-raw'] if a.from_derived else []))
if __name__=='__main__':main()
