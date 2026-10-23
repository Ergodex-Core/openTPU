"""Device LSTM cache, TDT edge cases, UTF-8 and bounded state/reset tests."""
import dataclasses

import numpy as np
import pytest

from opentpu import Config, language as ol
from opentpu.compiler import Tensor
from opentpu.asr.phonon2_int6 import Int6Weight
from opentpu.asr.phonon2_tdt import (DecoderShape, compile_decoder, weight_shapes, token_control,
                                   FRAME, STATUS, STEPS, ZERO_RUN, conditional, integer_equal)
from tools.phonon2_tdt_reference import text_table, reference, read_result, load_text_asset
from tools.phonon2_program_device import ProgramSlotDevice


def weights(shape, seed=410):
    rng, result = np.random.default_rng(seed), {}
    for name, (kind, dims) in weight_shapes(shape).items():
        if kind == "int6":
            result[name] = Int6Weight(rng.integers(-32, 32, dims, np.int8),
                                     np.full(dims[0], np.float16(.1 / np.sqrt(dims[1]))))
        else:
            result[name] = rng.normal(0, .05, dims).astype(np.float16).astype(np.float32)
    return result


def run(plan, x, device=None, check=True):
    device = device or ProgramSlotDevice(plan.cfg)
    if device.machine is None:
        device.reset(plan.image.size)
    device.constants(plan.image, ((0, plan.input_at),))
    device.memory[plan.input_at:plan.input_at + x.nbytes] = x.view(np.uint8).ravel()
    for p in plan.programs:
        device.phase(p, check=check)
    return read_result(device, plan), device


@pytest.mark.parametrize("passes", [1, 2])
def test_two_layer_lstm_state_and_logits_against_native_torch(passes):
    shape = DecoderShape(64, 32, 8, 7, 3)
    cfg = Config(D=32, ACT_BLOCKS=128, DRAM_BYTES=8 << 20, IMEM_WORDS=32768, SEQ_WC=True)
    w = weights(shape)
    # A nonblank zero-duration path exercises recurrent state across >10 steps.
    w["joint.head.bias"][:] = -10
    w["joint.head.bias"][[5, 8]] = 10
    table = text_table(["<unk>", "▁é", "<pad>", "<eos>", "▁", "▁世界", "!", "<blank>"], {0, 2, 3, 7}, 8)
    x = np.random.default_rng(23).normal(size=(3, 64)).astype(np.float32)
    plan = compile_decoder(w, table, 3, 2, cfg, shape, passes=passes, max_steps=12, audit=True)
    actual, _ = run(plan, x)
    expected = reference(w, x, 2, shape, max_steps=12, table=table)
    assert actual['state'][STATUS] == expected['status'] == 3
    assert actual['state'][FRAME] == 0 and actual['state'][ZERO_RUN] == 12
    assert actual['state'][STEPS] == 12
    assert actual['text'] == expected['text'] == ('世界' + ' 世界' * 11).encode()
    np.testing.assert_array_equal(actual['trace'], [[5, 0, 0, 1]] * 12)
    for i, record in enumerate(expected['records']):
        state = np.concatenate((record['hidden'][:, 0], record['cell'][:, 0])).ravel()
        if passes == 1:
            # Compare accumulated activation quantization over the whole state;
            # relative error of a single near-zero coordinate is ill-conditioned.
            assert np.linalg.norm(actual['audit_hidden'][i] - state) / np.linalg.norm(state) < .03
            assert np.linalg.norm(actual['audit_cache'][i] - record['cache'][0]) / np.linalg.norm(record['cache'][0]) < .03
        else:
            np.testing.assert_allclose(actual['audit_hidden'][i], state, atol=2e-5, rtol=.001)
            np.testing.assert_allclose(actual['audit_cache'][i], record['cache'][0], atol=2e-5, rtol=.001)
    if passes == 2:
        import torch
        # Independent built-in two-layer LSTM audit of the literal eager oracle.
        lstm = torch.nn.LSTM(shape.hidden, shape.hidden, 2, batch_first=True)
        lstm.load_state_dict({n.removeprefix('decoder.lstm.'):torch.from_numpy(a.dense() if hasattr(a,'dense') else a.copy())
                              for n,a in w.items() if n.startswith('decoder.lstm.')})
        embeddings = torch.from_numpy(w['decoder.embedding.weight'].dense()[[7]+[5]*11])[None]
        with torch.no_grad():
            _, (hh,cc) = lstm(embeddings)
        np.testing.assert_allclose(expected['records'][-1]['hidden'], hh.numpy(), atol=1e-6, rtol=1e-5)
        np.testing.assert_allclose(expected['records'][-1]['cell'], cc.numpy(), atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize('token,duration,status,steps', [(7, 0, 1, 2), (7, 4, 1, 1), (3, 0, 2, 1)])
@pytest.mark.parametrize('stored,valid',[(1,1),(3,2)])
def test_blank_zero_duration_mask_eos_and_cache_reuse(token, duration, status, steps,stored,valid):
    shape = DecoderShape(64, 32, 8, 7, 3)
    cfg = Config(D=32, DRAM_BYTES=8 << 20, IMEM_WORDS=32768, SEQ_WC=True)
    w = weights(shape);w['joint.head.bias'][:] = -20;w['joint.head.bias'][[token, 8 + duration]] = 20
    table = text_table([str(i) for i in range(8)], {0, 2, 3, 7}, 8)
    x = np.random.default_rng(24).normal(size=(stored, 64)).astype(np.float32)
    plan = compile_decoder(w, table, stored, valid, cfg, shape, max_steps=8, audit=True)
    actual, _ = run(plan, x)
    steps=min(steps,valid)
    assert actual['state'][STATUS] == status and actual['state'][STEPS] == steps
    if token == 7:
        assert actual['text'] == b''
        if steps == 2:
            np.testing.assert_array_equal(actual['audit_hidden'][0], actual['audit_hidden'][1])
            np.testing.assert_array_equal(actual['audit_cache'][0], actual['audit_cache'][1])
    assert actual['trace'][:, 2].max() < valid  # masked encoder row never read


def test_decoder_reset_after_different_recurrent_history_is_bitwise():
    shape = DecoderShape(64, 32, 8, 7, 3)
    cfg = Config(D=32, DRAM_BYTES=8 << 20, IMEM_WORDS=32768, SEQ_WC=True)
    table = text_table([str(i) for i in range(8)], {0, 2, 3, 7}, 8)
    w = weights(shape);w['joint.head.bias'][:] = -20;w['joint.head.bias'][[5, 8]] = 20
    plan = compile_decoder(w, table, 3, 2, cfg, shape, max_steps=12, audit=True)
    rng = np.random.default_rng(25);x = rng.normal(size=(3, 64)).astype(np.float32)
    _, device = run(plan, rng.normal(size=x.shape).astype(np.float32))
    reused, _ = run(plan, x, device);fresh, _ = run(plan, x)
    for name in ['state','trace','hidden','cell','cache','audit_hidden','audit_cache']:
        np.testing.assert_array_equal(reused[name], fresh[name])
    assert reused['text'] == fresh['text']


@pytest.mark.parametrize('frames,valid,max_steps', [(0, 0, None), (377, 376, None), (8, 9, None), (8, 7, 0), (8, 7, 3761)])
def test_decoder_guards(frames, valid, max_steps):
    with pytest.raises(ValueError):
        compile_decoder({}, None, frames, valid, Config(D=32), DecoderShape(64, 32, 8, 7, 3), max_steps=max_steps)


def test_every_official_piece_metaspace_and_specials_against_installed_tokenizers():
    from pathlib import Path
    tokenizers = pytest.importorskip('tokenizers')
    table, pieces, policy = load_text_asset(Path(__file__).resolve().parents[1])
    tokenizer = tokenizers.Tokenizer(tokenizers.models.BPE({p:i for i,p in enumerate(pieces)}, [], unk_token='<unk>'))
    tokenizer.decoder = tokenizers.decoders.Metaspace(replacement='▁', prepend_scheme='always', split=True)
    tokenizer.add_special_tokens([pieces[i] for i in policy['skip_special_ids']])
    # Every piece in first and subsequent position, plus skipped prefixes,
    # repeated spaces, repeats (no CTC deduplication), and multilingual pieces.
    for following in (0, 1):
        for i in range(len(pieces)):
            ids = ([5] if following else []) + [i]
            selected, seen = [], 0
            for token in ids:
                if table[2][token]:
                    bank = min(seen, 1);n = int(table[1][token, bank])
                    selected.append(bytes(table[0][token, bank, :n]));seen += 1
            assert b''.join(selected).decode() == tokenizer.decode(ids, skip_special_tokens=True)
    assert tokenizer.decode([0,2,8192,3], skip_special_tokens=True) == ''


@ol.jit
def scripted_control(state, logits, trace, text, text_lengths, table, lengths, visible,
                     shape: DecoderShape, valid: int, steps: int):
    for step in ol.range(steps):
        active = integer_equal(ol.load(state[STATUS:STATUS+1]), 0.0)
        conditional(active, lambda: token_control(state, ol.load(logits[step:step+1,:]), trace, text,
                    text_lengths, table, lengths, visible, shape, valid, step, steps))


def test_device_text_empty_first_piece_repeats_specials_and_zero_duration_sequence():
    shape = DecoderShape(64,32,8,7,3)
    cfg = Config(D=32,DRAM_BYTES=8<<20,IMEM_WORDS=32768,SEQ_WC=True)
    table = text_table(['<unk>','▁','<pad>','<eos>','▁é','▁世界','!','<blank>'],{0,2,3,7},8)
    plan = compile_decoder(weights(shape),table,3,2,cfg,shape,max_steps=7)
    # No CTC dedup; blank0 advances, EOS stops, later records stay untouched.
    tokens,durations=[1,4,4,7,3,5,5],[0,0,0,0,0,4,4]
    logits=np.full((7,13),-10,np.float32)
    for i,(token,duration) in enumerate(zip(tokens,durations)):logits[i,[token,8+duration]]=10
    device=ProgramSlotDevice(cfg);device.reset(plan.image.size+logits.nbytes)
    device.constants(plan.image,((0,plan.input_at),));device.phase(plan.programs[0],check=True)
    device.memory[plan.image.size:plan.image.size+logits.nbytes]=logits.view(np.uint8).ravel()
    def desc(name):
        d=plan.checkpoints[name];s=tuple(d['shape']);return Tensor(d['offset'],s,(s[1],1) if len(s)==2 else (1,))
    # The immutable text tables follow the identity in the prefix; find them
    # from the exact packed table bytes, avoiding a second production path.
    prefix=plan.image[:plan.input_at].tobytes();table_at=prefix.index(table[0].tobytes())
    lengths_at=(table_at+table[0].nbytes+127)//128*128
    visible_at=(lengths_at+table[1].nbytes+127)//128*128
    args={n:desc(n) for n in ['state','trace','text','text_lengths']}
    args.update(logits=Tensor(plan.image.size,logits.shape,(13,1)),table=Tensor(table_at,(128,),(1,)),
                lengths=Tensor(lengths_at,(8,2),(2,1)),visible=Tensor(visible_at,(8,),(1,)),shape=shape,valid=2,steps=7)
    device.phase(scripted_control.trace(cfg,0,args).finish(),check=True)
    actual=read_result(device,plan)
    assert actual['text']==' é é'.encode()
    assert actual['state'][STATUS]==2 and actual['state'][STEPS]==5
    np.testing.assert_array_equal(actual['trace'][:,2],[0,0,0,0,1])


def test_tied_joint_heads_choose_first_token_and_duration_independently():
    shape=DecoderShape(64,32,8,7,3);cfg=Config(D=32,DRAM_BYTES=8<<20,IMEM_WORDS=32768,SEQ_WC=True)
    w=weights(shape);w['joint.head.weight']=Int6Weight(np.zeros((13,32),np.int8),np.zeros(13,np.float16));w['joint.head.bias'][:]=0
    table=text_table([str(i) for i in range(8)],{0,2,3,7},8)
    plan=compile_decoder(w,table,1,1,cfg,shape,max_steps=3)
    actual,_=run(plan,np.ones((1,64),np.float32))
    np.testing.assert_array_equal(actual['trace'],[[0,0,0,1]]*3)
    assert actual['state'][STATUS]==3 and actual['text']==b''
