"""Run raw-PCM model qualification on verified local common-runtime source."""
import argparse
import json
import os
from pathlib import Path
import runpy
import sys
from tools.phonon2_common_runtime import verify


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--runtime-root',required=True)
    a,remaining=p.parse_known_args();root=Path(__file__).resolve().parents[1];runtime=Path(a.runtime_root).resolve()
    manifest=json.loads((root/'docs/phonon2/waveform_runtime_dependency.json').read_text())['runtime_manifest']
    verify(runtime,manifest)
    sys.path.insert(0,str(runtime));import opentpu
    opentpu.__path__.append(str(root/'opentpu'));sys.path.insert(0,str(root))
    os.environ['OTPU_TARGET']='f2-hbm:p4rl8a64c150'
    sys.argv=['tools.phonon2_qualify_waveform','--root',str(root),*remaining]
    runpy.run_module('tools.phonon2_qualify_waveform',run_name='__main__')


if __name__=='__main__':main()
