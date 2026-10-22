"""A batched Qwen3 image retains the ordinary generate loop for sequence 0."""
import dataclasses
from types import SimpleNamespace

import numpy as np

from opentpu import isa as I, progcache as PC
from opentpu.host import gen_warm as GW
from opentpu.llm import generate as G
from opentpu.llm.qwen3 import PREFILL_ROWS, Engine, device_config

from test_autodecode import _tiny


def _words(progs):
    parts = progs if isinstance(progs, tuple) else (progs,)
    return tuple(tuple(I.assemble(p).tobytes() for p in part) for part in parts)


def test_slot0_warm_cache_matches_batch_image_and_isa(tmp_path, monkeypatch):
    """The spawned prewarm rebuilds the same batched image, including slot 0's KV offsets.

    Both bucket programs and their cache keys must match the runtime engine, and using the
    warmed programs must give the same tokens as an uncached ISA engine across a bucket cut.
    The batch area is allocated after the image; it is not part of a generate program.
    """
    monkeypatch.setenv("OTPU_PROG_CACHE", str(tmp_path))
    PC.clear()
    W, spec = _tiny("qwen3")
    cfg = device_config(spec, 512, batch=4, rows=PREFILL_ROWS, lookup=True, S=1)
    cfg = dataclasses.replace(cfg, DRAM_BYTES=2 * cfg.DRAM_BYTES)
    kw = dict(cap=512, cfg=cfg, batch=4, resident=True, batch_area=True,
              prompt_runs=False)
    prompt = [int(t) for t in np.random.default_rng(7).integers(0, 1000, 248)]
    ref = Engine(spec, W, prog_cache=False, **kw)
    first = int(np.argmax(ref.prefill(prompt)))
    want = ref.generate_card(first, 12, stop_ids=[])

    eng = Engine(spec, W, prog_cache=True, **kw)
    assert eng.can_generate and eng.image.batch == eng.batch == 4 and eng.bat is not None
    assert eng.layout[3] == 4       # the batch size is part of the program-cache key
    sampling = dict(temperature=0.7, top_k=20, top_p=0.9, repetition_penalty=1.1)
    p = GW.start(eng, sampling)
    assert p is not None
    p.join(300)
    assert p.exitcode == 0

    chain = bool(getattr(eng.backend, "chains", False))
    PC.clear()                       # force reads of the spawned process's disk entries
    for bucket in (1, 2):
        for samp in GW.samplings(sampling):
            what = GW.gen_what(eng.block, eng._conv_lo, bucket, chain, samp,
                               eng.gen_debug, eng.gen_split)
            assert PC.has(eng.layout, what)
            cached = PC.get(eng.layout, what, lambda: (_fail_compile(), None))[0]
            direct = G.compile_bucket(eng.image, bucket, what[2], eng.block, chain=chain,
                                      samp=samp, debug=eng.gen_debug, split=eng.gen_split)
            assert _words(cached) == _words(direct)

    tok = int(np.argmax(eng.prefill(prompt)))
    assert tok == first
    PC.clear()                       # generate_card must use the disk entries itself
    before = dict(PC.stats)
    got = eng.generate_card(tok, 12, stop_ids=[])
    assert got == want and eng.pos == len(prompt) + len(got)
    assert PC.stats["compile"] == before["compile"]
    assert PC.stats["disk"] - before["disk"] >= 2
    PC.clear()


def _fail_compile():
    raise AssertionError("prewarmed bucket was unexpectedly recompiled")


def test_other_batch_layout_stays_unwarmed(tmp_path, monkeypatch):
    """A custom multi-sequence image has no Qwen3 slot-0 layout guarantee."""
    monkeypatch.setenv("OTPU_PROG_CACHE", str(tmp_path))
    fake = SimpleNamespace(can_generate=True, prog_cache=True, batch=4,
                           spec=object(), image=SimpleNamespace(batch=4))
    assert GW.start(fake) is None
