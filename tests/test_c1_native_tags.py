"""Independent native payload/tag conservation across stalls and pointer wrap."""
from pathlib import Path
import shutil
import subprocess
import pytest

ROOT=Path(__file__).resolve().parents[1]

@pytest.mark.skipif(not shutil.which('verilator'),reason='Verilator required')
def test_all_engine_tags_and_payloads(tmp_path):
    cmd=['verilator','--binary','--timing','-Wno-fatal','-Wno-WIDTH','--top-module','tb_c1_native_packed','--Mdir',str(tmp_path/'obj'),str(ROOT/'rtl/c1/tb_native_tags.sv'),str(ROOT/'rtl/c1/otpu_c1_native.sv')]
    build=subprocess.run(cmd,capture_output=True,text=True,timeout=180)
    assert build.returncode==0,build.stdout+build.stderr
    run=subprocess.run([str(tmp_path/'obj/Vtb_c1_native_packed')],capture_output=True,text=True,timeout=60)
    assert run.returncode==0,run.stdout+run.stderr
    assert 'C1_NATIVE_PACKED_EXACT groups=32768 native_reads=16384 rows=64 cycles=3966' in run.stdout
