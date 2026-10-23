#!/usr/bin/env python3
"""Pinned real-checkpoint codec-frame acceptance; CPU references/ISA only."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tests')]
from opentpu.llm.qwen3_tts_shared import shared_tts_config, profile_metadata
from opentpu.llm.qwen3_tts_talker import TalkerSpec
from opentpu.llm.qwen3_tts_frame import CodecFrameImage, CodecFrameISASession, check_predictor_config
from opentpu.llm.qwen3_tts_prompt import custom_voice_prefill
from tools.qwen3_tts_checkpoint_acceptance import BF16Checkpoint, metrics, sha256
from tools.qwen3_tts_stack_acceptance import selection
from qwen3_tts_frame_reference import FrameReference


def source_hashes():
    paths = [Path(__file__).resolve(), ROOT/'opentpu/llm/qwen3_tts_frame.py', ROOT/'opentpu/kernels/tts_frame.py',
             ROOT/'tests/qwen3_tts_frame_reference.py', ROOT/'tests/qwen3_tts_stack_reference.py']
    return {str(p.relative_to(ROOT)):sha256(p) for p in paths}


def run(a):
    started = time.monotonic()
    code_sha256 = source_hashes()  # validate provenance paths before expensive execution
    config = json.loads((ROOT / 'tests/fixtures/qwen3_tts_06b_config.json').read_text())
    fixture = json.loads((ROOT / 'tests/fixtures/qwen3_tts_prompts.json').read_text())
    check_predictor_config(config)
    reader = BF16Checkpoint(a.checkpoint, json.loads(a.manifest.read_text()))
    image = CodecFrameImage(TalkerSpec.from_config(config), shared_tts_config())
    weights = {k: reader.tensor(k, shape) for k, (_, _, shape) in image.parameters.items()}
    packed = image.build(weights, max_bytes=2 << 30)
    session = CodecFrameISASession(image, packed)
    refs = {'fp32': FrameReference(image, weights), 'bf16': FrameReference(image, weights, bf16=True)}
    # Conditional predictor diagnostics use identical ISA hidden/IDs, separating
    # local predictor numerical error from open-loop talker/ID divergence.
    conditional = {'fp32': FrameReference(image, weights), 'bf16': FrameReference(image, weights, bf16=True)}
    rows, frames, replay = [], [], None
    max_error = .035  # preserved logit relative-L2 acceptance, never relaxed
    for prompt in fixture['prompts']:
        schedule = custom_voice_prefill(prompt['input_ids'], config)
        session.reset()
        for ref in refs.values():
            ref.reset()
        # Poison all future KV bytes, including frame-local predictor caches.
        for address, size in image.cache_regions().values():
            session.machine.slices[0].dram[address:address + size] = 0x3f
        for position, row in enumerate(schedule):
            wants = {name: ref.prefill(row) for name, ref in refs.items()}
            logits, raw, masked = session.step(row.text_id, row.codec_id, input_kind=row.kind)
            rows.append(dict(text=prompt['text'], position=position, kind=row.kind,
                             last_prefill=position + 1 == len(schedule), device_ids=[raw, masked],
                             references={n: dict(ids=list(selection(w['logits'])), error=metrics(logits, w['logits']))
                                         for n, w in wants.items()}))
        for index in range(a.frames):
            before_hidden = session.read('normalized')
            captured, sums = [], []
            def observe(name, sess):
                if name.startswith('predictor.head.'):
                    desc = image.predictor.logits
                    logits = sess.machine.slices[0].dram[desc.base.const:desc.base.const + image.audio_vocab * 4].view(np.float32).copy()
                    captured.append(logits)
                if name == 'codec.feedback':
                    sums.append(sess.read('codec_sum'))
            ids = session.frame(observe)
            expected = {n: ref.frame() if not ref.stopped else None for n, ref in refs.items()}
            record = dict(text=prompt['text'], frame=index, position=session.position - 1,
                          ids=None if ids is None else ids.tolist(), references={})
            if ids is None:
                record['eos'] = True
                record['references'] = {n: {'eos': w is None} for n, w in expected.items()}
                frames.append(record)
                a.output.with_suffix('.frames.partial.json').write_text(json.dumps(frames, allow_nan=False) + '\n')
                break
            for name, want in expected.items():
                ref = conditional[name]
                _, summed, head_rows = ref.predictor_frame(before_hidden, int(ids[0]), force_ids=ids)
                record['references'][name] = dict(
                    eos=want is None,
                    open_loop_ids=None if want is None else want['ids'].tolist(),
                    open_loop_predictor=None if want is None else [dict(
                        error=metrics(x, y['logits']), greedy=y['greedy'], agrees=int(ids[i + 1]) == y['greedy'])
                        for i, (x, y) in enumerate(zip(captured, want['predictor']))],
                    conditional_predictor=[dict(error=metrics(x, y['logits']), greedy=y['greedy'],
                                                agrees=int(ids[i + 1]) == y['greedy'])
                                           for i, (x, y) in enumerate(zip(captured, head_rows))],
                    embedding_sum_error=metrics(sums[0], ref.talker.array(summed)),
                    feedback_error=None if want is None else metrics(session.read('logits'), want['feedback']['logits']))
            frames.append(record)
            a.output.with_suffix('.frames.partial.json').write_text(json.dumps(frames, allow_nan=False) + '\n')
            progress = dict(text=prompt['text'], frame=index + 1, ids=ids.tolist(),
                            fp32_agree=int(sum(x == y for x, y in zip(ids, record['references']['fp32']['open_loop_ids'] or []))))
            a.output.with_suffix('.progress.json').write_text(json.dumps(progress))
            print(json.dumps(progress), flush=True)
        if replay is None:
            first_ids = [f['ids'] for f in frames if f['text'] == prompt['text']]
            first_logits = session.read('logits')
            session.reset()
            for row in schedule:
                session.step(row.text_id, row.codec_id, input_kind=row.kind)
            replay_ids = []
            for _ in first_ids:
                got = session.frame()
                replay_ids.append(None if got is None else got.tolist())
                if got is None:
                    break
            replay = dict(ids_equal=replay_ids == first_ids,
                          logits_bitwise_equal=bool(np.array_equal(first_logits.view(np.uint32), session.read('logits').view(np.uint32))))
    summary = {}
    active = [f for f in frames if f['ids'] is not None]
    finals = [r for r in rows if r['last_prefill']]
    for name in refs:
        op = [h for f in active for h in (f['references'][name]['open_loop_predictor'] or [])]
        cond = [h for f in active for h in f['references'][name]['conditional_predictor']]
        id_agree = sum(sum(x == y for x, y in zip(f['ids'], f['references'][name]['open_loop_ids'] or [])) for f in active)
        summary[name] = dict(
            generated_ids=len(active) * image.codebooks, open_loop_id_agreement=id_agree,
            complete_frame_agreement=sum(f['ids'] == f['references'][name]['open_loop_ids'] for f in active),
            open_loop_predictor_greedy_agreement=sum(h['agrees'] for h in op), predictor_decisions=len(cond),
            conditional_predictor_greedy_agreement=sum(h['agrees'] for h in cond),
            max_open_loop_predictor_logit_error=max((h['error']['relative_l2'] for h in op), default=0),
            max_conditional_predictor_logit_error=max(h['error']['relative_l2'] for h in cond),
            max_feedback_logit_error=max((f['references'][name]['feedback_error']['relative_l2'] for f in active if f['references'][name]['feedback_error']), default=0),
            max_embedding_sum_error=max(f['references'][name]['embedding_sum_error']['relative_l2'] for f in active),
            prefill_final_greedy_agreement=sum(r['device_ids'][1] == r['references'][name]['ids'][1] for r in finals),
            prefill_all_row_fixed_mask_greedy_agreement=sum(r['device_ids'][1] == r['references'][name]['ids'][1] for r in rows),
            prefill_all_rows=len(rows), final_prefill_rows=len(finals))
        summary[name]['eos_decision_agreement'] = sum(bool(f.get('eos', False)) == f['references'][name]['eos'] for f in frames)
        summary[name]['eos_decisions'] = len(frames)
    accepted = all(s['open_loop_id_agreement'] == s['generated_ids'] and
                   s['eos_decision_agreement'] == s['eos_decisions'] and
                   s['max_open_loop_predictor_logit_error'] <= max_error and
                   s['max_feedback_logit_error'] <= max_error for s in summary.values()) and all(replay.values())
    result = dict(profile=profile_metadata(), checkpoint_sha256=reader.manifest['sha256'] if hasattr(reader, 'manifest') else sha256(a.checkpoint),
                  runtime_revision='30100afb075374381601a6e01d9f914a58dbec91',
                  input_contract='Official non_streaming CustomVoice prompt; generated codec frames, no waveform/audio-quality claim',
                  generation=dict(do_sample=False, subtalker_dosample=False, repetition_penalty=1.0, min_new_tokens=2),
                  precision='INT8 matrices/activations/KV; FP32 embeddings, reductions, residuals, norms, RoPE',
                  summary=summary, frames=frames, reset_poison_replay=replay, accepted=accepted,
                  thresholds=dict(logit_relative_l2=max_error, generated_greedy_agreement=1.0),
                  device_dram_bytes=image.nbytes, seconds=time.monotonic() - started,
                  code_sha256=code_sha256,
                  physical_hardware_access=False, upstream_model_code_executed=False, tts_enabled=False)
    details = a.output.with_suffix('.prefill.json')
    details.write_text(json.dumps(rows, allow_nan=False) + '\n')
    result['prefill_receipt_sha256'] = sha256(details)
    a.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'frames'}, indent=2), flush=True)
    return 0 if accepted else 2


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--frames', type=int, default=3)
    args = p.parse_args()
    if not 2 <= args.frames <= 4:
        p.error('bounded multi-frame validation requires 2..4 frames')
    sys.exit(run(args))
