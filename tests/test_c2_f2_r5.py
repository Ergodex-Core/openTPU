"""Pinned C2 R5 fabric RTL; behavioral SmartConnect/HBM, never FPGA evidence.

The DUT is the imported fbf21635 fabric, including CDC FIFOs and native AXI
adapters. The converter models width/burst conversion and response IDs, not
vendor IP implementation, timing, or capacity. Builds are fresh, isolated, and
limited to two compiler jobs; logs remain under build/c2-r5-verification/.
"""
from pathlib import Path
import hashlib
import shutil
import subprocess
import tempfile

import pytest


ROOT = Path(__file__).resolve().parents[1]
SUPPORT = ROOT / "boards/c2_f2/support/rtl"
PINNED = {
    "shell/otpu_afifo.sv": "fea05da515b0226a9da43c9d7f535c5ee0b7870a86fb15e1142d254d9ff257da",
    "shell/otpu_skidbuf.sv": "7458d8d07bf9de3a2a72aacf6125d32ab65465531423e6ab7ef0a5975c9512ee",
    "shell/otpu_link.sv": "bf5aba348bf56ac8742fcbf6702471de0301a0197ba0c6f12991f4c7037836a3",
    "boards/f2/otpu_native_axi.sv": "8971ec9fd12c58329c4b6a8c6895b434c903db1cffbea168cffad8966a5d3427",
    "boards/f2/otpu_f2_hbm_port.sv": "f02f8167f051b4f1609f83e28dc4063976d0f280796ccb22fb983ff45ba02aff",
    "boards/f2/otpu_f2_hbm_host.sv": "e8cb761da03a59cc7188e12258881b888e3ff2dac33629054c1a0866a3e53afb",
    "boards/f2/otpu_f2_hbm_mem.sv": "93ace20db879f78a7017bd0631a7255fe3cc7e0d28318260170e6fe300c5ee07",
}


@pytest.fixture(scope="module")
def fabric_run():
    for name, expected in PINNED.items():
        assert hashlib.sha256((SUPPORT / name).read_bytes()).hexdigest() == expected, name
    compiler = shutil.which("verilator")
    if compiler is None:
        pytest.skip("Verilator is required for the C2 R5 fabric regression")
    parent = ROOT / "build/c2-r5-verification"
    parent.mkdir(parents=True, exist_ok=True)
    build_root = Path(tempfile.mkdtemp(prefix="run-", dir=parent))
    binaries = {}

    def run(*, clocks=(45, 36, 20), hazard=1, seed=1, nb=6, cycles=8000):
        # 45:36:20 has the exact 200:250:450 MHz ratio. It is a simulation
        # timescale, not an assertion that any FPGA path meets those periods.
        key = (*clocks, hazard)
        out = build_root / ("c%d-a%d-h%d-haz%d" % key)
        if key not in binaries:
            out.mkdir()
            cmd = [compiler, "--binary", "-j", "2", "-Wno-fatal",
                   "-Wno-WIDTHEXPAND", "-Wno-WIDTHTRUNC", "-Wno-UNUSEDSIGNAL",
                   "-Wno-UNUSEDPARAM", "-O3", "--x-assign", "0", "--x-initial", "0",
                   "--top-module", "tb_c2_f2_r5", "-Mdir", str(out),
                   "-GNPORT=8", f"-GCORE_HALF={clocks[0]}",
                   f"-GAXI_HALF={clocks[1]}", f"-GHBM_HALF={clocks[2]}", f"-GHAZ={hazard}",
                   *(str(SUPPORT / name) for name in PINNED),
                   str(ROOT / "sim/verilator/tb_c2_f2_r5_sc.sv"),
                   str(ROOT / "sim/verilator/tb_c2_f2_r5.sv")]
            result = subprocess.run(cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, timeout=600)
            (out / "compile.log").write_text(result.stdout)
            assert result.returncode == 0, f"{out / 'compile.log'}\n{result.stdout[-12000:]}"
            binaries[key] = out / "Vtb_c2_f2_r5"
        result = subprocess.run([str(binaries[key]), f"+seed={seed}", f"+nb={nb}",
                                 f"+cycles={cycles}"], cwd=out, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=600)
        log = out / f"seed{seed}-nb{nb}-cycles{cycles}.log"
        log.write_text(result.stdout)
        assert result.returncode == 0, f"{log}\n{result.stdout}"
        return result.stdout, log

    return run


@pytest.mark.parametrize("clocks", [(45, 36, 20), (3, 5, 2)])
@pytest.mark.parametrize("seed,nb", [(1, 1), (7, 6), (31, 64)])
def test_pinned_r5_native_and_pcis(fabric_run, clocks, seed, nb):
    output, log = fabric_run(clocks=clocks, seed=seed, nb=nb)
    assert "FAIL" not in output and "PASS" in output, f"{log}\n{output}"
    assert "C2 R5 COVERAGE credits=128 per PC; 4KiB boundaries; IDs0/1;" in output


def test_direction_barrier_negative_control(fabric_run):
    # Removing the native read/write hazard barrier must make the independent
    # byte scoreboard catch stale data; an unrelated failure is insufficient.
    output, log = fabric_run(hazard=0, seed=1, nb=1, cycles=4000)
    assert "FAIL" in output and "stale or wrong data" in output, f"{log}\n{output}"
