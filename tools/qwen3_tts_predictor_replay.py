#!/usr/bin/env python3
"""Real-checkpoint exact-MX predictor replay; CPU diagnostic only, no FPGA."""
import argparse
import json
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'tests')]
from opentpu.llm.qwen3_tts_shared import shared_tts_config, profile_metadata
from opentpu.llm.qwen3_tts_talker import TalkerSpec
from opentpu.llm.qwen3_tts_frame import CodecFrameImage, CodecFrameISASession, check_predictor_config
from opentpu.llm.qwen3_tts_prompt import custom_voice_prefill
from tools.qwen3_tts_checkpoint_acceptance import BF16Checkpoint, sha256
from qwen3_tts_predictor_exact import PredictorExactReplay


def run(a):
    started = time.monotonic()
    paths = [Path(__file__).resolve(), ROOT/'tests/qwen3_tts_predictor_exact.py', ROOT/'tests/qwen3_tts_exact_mx.py']
    hashes = {str(p.relative_to(ROOT)):sha256(p) for p in paths}
    config = json.loads((ROOT/'tests/fixtures/qwen3_tts_06b_config.json').read_text())
    prompt = json.loads((ROOT/'tests/fixtures/qwen3_tts_prompts.json').read_text())['prompts'][0]
    check_predictor_config(config)
    reader = BF16Checkpoint(a.checkpoint, json.loads(a.manifest.read_text()))
    image = CodecFrameImage(TalkerSpec.from_config(config), shared_tts_config())
    weights = {k:reader.tensor(k, shape) for k, (_, _, shape) in image.parameters.items()}
    session = CodecFrameISASession(image, image.build(weights, max_bytes=2<<30))
    replay = PredictorExactReplay(image, weights)
    for row in custom_voice_prefill(prompt['input_ids'], config):
        session.step(row.text_id, row.codec_id, input_kind=row.kind)
    ids = []
    for frame in range(2):
        codes = session.frame(replay.observe)
        ids.append(codes.tolist())
        a.output.with_suffix('.partial.json').write_text(json.dumps(replay.rows))
        print(json.dumps(dict(frame=frame + 1, codes=codes.tolist(), mismatches=sum(not r['bitwise_equal'] for r in replay.rows))), flush=True)
    result = dict(profile=profile_metadata(), prompt=prompt, ids=ids, rows=replay.rows,
                  layer_comparisons=sum(r['part'].startswith('predictor.layer.') for r in replay.rows),
                  head_comparisons=sum(r['part'].startswith('predictor.head.') for r in replay.rows),
                  all_bitwise_equal=all(r['bitwise_equal'] for r in replay.rows),
                  seconds=time.monotonic()-started, code_sha256=hashes,
                  scope='Independent operator replay sharing documented FP32/MX primitives and offline packer; not quality acceptance',
                  physical_hardware_access=False, tts_enabled=False)
    a.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2),flush=True)
    return 0 if result['all_bitwise_equal'] else 2


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    sys.exit(run(p.parse_args()))
