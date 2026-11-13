"""Rebuild into a new directory. Requires the verified feasibility inputs."""
from pathlib import Path
import argparse
import os
import shutil
import subprocess
import sys
import tempfile

HERE=Path(__file__).resolve().parent

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();out=args.out.resolve()
    if out.exists():raise FileExistsError(out)
    env=os.environ.copy();env.setdefault('MPLCONFIGDIR',tempfile.mkdtemp(prefix='takeover-mpl-'))
    rscript=shutil.which('Rscript') or '/usr/local/bin/Rscript'
    for cmd in [[sys.executable,str(HERE/'prepare.py'),'--out',str(out)],
                [rscript,'--vanilla',str(HERE/'forest.R'),str(out)],
                [sys.executable,str(HERE/'summarize.py'),str(out)]]:
        subprocess.run(cmd,cwd=HERE.parent,env=env,check=True)
    print('Completed',out)

if __name__=='__main__':main()
