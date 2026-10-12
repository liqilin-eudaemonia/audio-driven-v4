"""CPU-only resumable input fingerprint preparation; never overwrite a baseline."""
import argparse
import fcntl
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from omni_av_p3_staged.integrity import prepare_inputs
from omni_av_p3_staged.orchestration import read,sha


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--control-root',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    if args.workers<1:raise ValueError('Positive worker count required')
    args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'prepare.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        prepare_inputs(read(args.manifest),args.output,sha(args.manifest),args.control_root,args.workers)


if __name__=='__main__':main()
