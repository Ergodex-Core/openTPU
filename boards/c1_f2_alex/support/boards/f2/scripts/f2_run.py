#!/usr/bin/env python3
"""Run an existing upstream CLI with the F2 transport, without editing upstream.
Usage: f2_run.py tools/validate.py --model /path/to/Qwen3-0.6B ...
"""
from pathlib import Path
import os, runpy, sys
ROOT=Path(__file__).resolve().parents[3]
os.environ.setdefault('OTPU_RUN_DIR',str(ROOT/'build/f2/runtime'))
os.environ.setdefault('OTPU_TARGET',str(ROOT/'targets/f2.toml'))
os.environ.setdefault('HF_HOME',str(ROOT/'build/f2/hf-cache'))
os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY','1')
os.environ.setdefault('HF_HUB_OFFLINE','1')
os.environ.setdefault('XDG_CACHE_HOME',str(ROOT/'build/f2/runtime/cache'))
(ROOT/'build/f2/runtime/tmp').mkdir(parents=True,exist_ok=True)
os.environ.setdefault('TMPDIR',str(ROOT/'build/f2/runtime/tmp'))
sys.path.insert(0,str(ROOT))
from opentpu.host import board
from opentpu.host.f2_transport import F2Transport
board.XdmaTransport=F2Transport
if len(sys.argv)<2:raise SystemExit(__doc__)
script=(ROOT/sys.argv[1]).resolve()
if not script.is_relative_to(ROOT) or not script.is_file():
    raise SystemExit('Choose an existing upstream script')
sys.argv=sys.argv[1:];sys.argv[0]=str(script)
runpy.run_path(str(script),run_name='__main__')
