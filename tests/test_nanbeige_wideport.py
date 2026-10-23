"""Compact untied head layout and strict Nanbeige on the pinned wide-port ISA."""
from dataclasses import replace

import numpy as np
import pytest

from opentpu import quant
from opentpu.llm.nanbeige import Engine, Spec
from opentpu.llm.nanbeige_strict import StrictEngine, plan
from opentpu.target import config, load, arch_hash
from test_nanbeige import tiny_weights


def wideport():
    cfg = config(load("f2-hbm:p4rl8a64c150"))
    assert arch_hash(cfg) == 3714814630
    return cfg


@pytest.mark.parametrize("tied", [False, True])
@pytest.mark.parametrize("embedding", ["int8", "f32"])
def test_compact_head_keeps_logits_and_device_gather_bit_exact(tied, embedding):
    cfg = replace(wideport(), DRAM_BYTES=16 << 20)
    spec = Spec(128, 2, 6, 1, 128, 256, 128, tied=tied, embed=embedding, eos=())
    weights = tiny_weights(spec)
    outputs = []
    for grouped in (False, True):
        eng = Engine(spec, weights, cfg=replace(cfg, ROWLANES=grouped), cap=128,
                     rows=3, resident=True, embed_host=False, runtime_host=False,
                     pipeline=False, prog_cache=False)
        image = eng.image
        assert (image.head is None) == (grouped and not tied)
        assert bool(image.head_rlq) == grouped
        if grouped and not tied:
            # The vocabulary descriptor uses grouped weights; embeddings own a table.
            assert image.descriptors(0).head.data.const == image.head_rlq[0]
            assert not image.lookup.get("head_gsc")
        if grouped and tied and embedding == "int8":
            assert image.lookup["embed_q"][0] == image.head[0]
            assert image.lookup["head_gsc"]
        # Multi-row prefill walks groups; decode uses actual row lanes.
        logits = [eng.prefill([3, 7, 19]), eng.step(9), eng.step(11)]
        outputs.append(np.asarray(logits).view(np.uint32))
    np.testing.assert_array_equal(*outputs)


def test_strict_wideport_first_token_greedy_and_reset_match_baseline():
    cfg = replace(wideport(), DRAM_BYTES=16 << 20)
    spec = Spec(128, 2, 6, 1, 128, 256, 128, eos=())
    weights = tiny_weights(spec)
    engines = [StrictEngine(spec, weights, cfg=c, backend="isa") for c in
               (replace(cfg, MCOLS=4, PORTB=1, ROWLANES=False, ALINE=32, T2=False), cfg)]
    for prompt, count in (([3, 7, 19], 4), ([8, 2], 3), ([3, 7, 19], 4)):
        out = []
        for eng in engines:
            read = eng.backend.read
            at = eng.image.lookup["gen"]["out"]
            def tokens_only(sid, addr, n, read=read, at=at):
                assert at <= addr and addr + n <= at + 4 * (eng.cap + 1)
                return read(sid, addr, n)
            eng.backend.read = tokens_only
            out.append(eng.generate_strict(prompt, count)["tokens"])
        assert out[0] == out[1]


def test_production_wideport_image_fits_without_a_duplicate_head():
    cfg = wideport()
    spec = Spec(3072, 22, 48, 8, 128, 10752, 166144, eps=1e-5, theta=7e7)
    result = plan(spec, cfg)
    assert result["required_bytes"] == 4247789568 < 1 << 32
    assert result["head_storage"] == "row_lanes_only"
    assert result["programs"] == [{"bucket": 1, "prefill_words": [7136],
        "generate_words": [[7576]], "split_generate": False}]
    image = replace(spec, embed="int8").image(cfg, 128, lookup=True, embed_host=False)
    assert image.head is None and image.head_rl == 4
    assert all(n <= cfg.IMEM_WORDS for p in result["programs"]
               for part in p["generate_words"] for n in part)
    # The exact bytes avoided are one standard quantized head, with its scale rows.
    saved = spec.vocab * (quant.row_bytes(spec.hidden, "int8", cfg.D) +
                          quant.scale_bytes(spec.hidden, "int8", cfg.D))
    assert saved > 467869696  # the prior wide-port layout's measured deficit


def test_nonzero_base_is_explicitly_unsupported():
    spec = Spec(128, 2, 6, 1, 128, 256, 128)
    with pytest.raises(ValueError, match="base=0"):
        spec.image(wideport(), 128, base=4096)


def test_other_models_keep_their_existing_head_layout():
    from opentpu.llm.qwen3 import Spec as QwenSpec
    spec = QwenSpec(128, 2, 4, 1, 128, 256, 128, tied=False)
    image = spec.image(wideport(), 128, lookup=True, embed_host=False)
    assert image.head is not None and image.head_rlq is not None
