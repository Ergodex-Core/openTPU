"""Cycle-exact timing cuts: independent traffic scoreboards and RTL assertions."""
import pytest

from opentpu import rtlsim


@pytest.mark.parametrize("depth,lanes", [(2, 4), (8, 8), (16, 8)])
def test_collective_registered_head(have_verilator, depth, lanes):
    exe = rtlsim.build("tb_coll_port", [rtlsim.RTL / "top/otpu_pkg.sv",
                                      rtlsim.RTL / "top/otpu_coll.sv",
                                      rtlsim.TB / "tb_coll_port.sv"],
                      {"F": depth, "LANES": lanes})
    out = rtlsim.run_sim([exe], timeout=60)
    assert out.returncode == 0, out.stdout + out.stderr
    assert "HEAD PASS checks=11970" in out.stdout, out.stdout


@pytest.mark.parametrize("D", [32, 128])
@pytest.mark.parametrize("boot", [False, True])
def test_loop_predecode_forwarding(have_verilator, D, boot):
    """LOOP predecode sees post-edge R, including a return while its ir is held.

    Start with a run-argument LOOP; cover LI, same/different-source ADDI, R0,
    32-bit wrap, RLD, zero/one/two trips, nested loops and one-instruction bodies.
    D=128 also exercises the four instruction positions in an IMEM row.
    """
    import numpy as np

    from opentpu import Config, isa as I
    from opentpu.isasim import Machine

    cfg = Config(S=1, D=D, MCOLS=4, LANES=8, DRAM_BYTES=1 << 16,
                 TMEM_WORDS=1 << 12, IMEM_WORDS=1 << 12)
    dram = np.zeros(cfg.DRAM_BYTES, np.uint8)
    dram[:32] = np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype="<u4").view(np.uint8)
    step = I.addi(9, 9, 32)
    prog = [I.loop(1, 0, rcount=8), step, I.ld(0, 0, 8),
            I.li(0, 99), I.loop(1, 0), I.halt()]
    for n in (0, 1, 2):
        prog += [I.li(1, n), I.loop(1, 0, rcount=1), step,
                 I.li(1, 0xFFFFFFFF), I.addi(1, 1, n + 1),
                 I.loop(1, 0, rcount=1), step,
                 I.li(2, 0), I.addi(1, 2, n), I.loop(1, 0, rcount=1), step,
                 I.li(1, 0xFFFFFFFF), I.loop(1, n + 1, rcount=1), step,
                 I.rld(1, n, raw=True), I.loop(1, 0, rcount=1), step]
    prog += [I.rld(0, 2, raw=True), I.loop(1, 0), I.halt(),
             I.loop(3, 2), I.loop(1, 2), step, I.nop(),
             I.st(0x1000, 0, 8, ra=9), I.halt()]
    args = [2, 0, 0, 0, 0, 0, 0, 0]
    ref = Machine(cfg, [prog], [dram.copy()], args=args).run()
    drams, tmems, stats = rtlsim.run(
        cfg, [prog], [dram.copy()], args=args, boot=boot, max_cycles=20_000,
        uarch={"WIN": 16, "SEQ_WC": 0})
    assert np.array_equal(drams[0], ref.slices[0].dram)
    assert np.array_equal(tmems[0], ref.slices[0].tmem)
    assert stats["instructions"] == [ref.slices[0].icount]
    at = 0x1000 + 32 * (2 + 5 * (0 + 1 + 2) + 2 * 2)
    assert np.array_equal(drams[0][at:at + 32], dram[:32])
    print(f"LOOP_FORWARD D={D} boot={boot}: {stats}")


_DISPATCH_MODES = [
    pytest.param((), id="normal"),
    pytest.param(("+hz_ino=1",), id="inorder"),
    pytest.param(("+hz_rob=1",), id="rob"),
    pytest.param(("+hz_ino=1", "+hz_rob=1"), id="inorder-rob"),
]


@pytest.mark.parametrize("waitc", [0, 1])
@pytest.mark.parametrize("overrides", _DISPATCH_MODES)
def test_dispatch_completed_slot_lifetime(have_verilator, waitc, overrides):
    """Completion ends command consumption even if the ROB delays retirement."""
    exe = rtlsim.build("tb_seq_dispatch", [rtlsim.RTL / "top/otpu_pkg.sv",
                                         rtlsim.RTL / "tech/generic/otpu_tech_ram.sv",
                                         rtlsim.RTL / "seq/otpu_seq.sv",
                                         rtlsim.TB / "tb_seq_dispatch.sv"],
                      {"WAITC": waitc})
    out = rtlsim.run_sim([exe, *overrides], timeout=60)
    assert out.returncode == 0, out.stdout + out.stderr
    assert "DISPATCH_LIFETIME PASS" in out.stdout, out.stdout
    print(out.stdout)


@pytest.mark.parametrize("waitc", [0, 1])
@pytest.mark.parametrize("overrides", _DISPATCH_MODES)
def test_dispatch_preselection(have_verilator, monkeypatch, waitc, overrides):
    """Window reuse, concurrent starts/completions and both dependency schemes.

    Exercise in-order and ROB overrides separately and together; the RTL checks
    every preselected field and command read against the original selector,
    including idle reloads after completion but before ROB retirement.
    """
    import numpy as np

    from opentpu import Config
    from opentpu.isasim import Machine
    from test_rtl import _hazard_program, _images

    monkeypatch.setenv("OTPU_WAITS", "1")
    rng = np.random.default_rng(3017)
    cfg = Config(S=1)
    prog, images = _hazard_program(rng, cfg), _images(rng, 1)
    ref = Machine(cfg, [prog], [i.copy() for i in images]).run()
    drams, tmems, stats = rtlsim.run(
        cfg, [prog], [i.copy() for i in images], max_cycles=1_000_000,
        uarch={"WIN": 16, "SEQ_WC": waitc},
        plusargs=list(overrides))
    assert np.array_equal(drams[0], ref.slices[0].dram)
    assert np.array_equal(tmems[0], ref.slices[0].tmem)
    assert stats["instructions"] == [ref.slices[0].icount]
    print(f"DISPATCH_PRESELECT waitc={waitc} overrides={overrides}: {stats}")
