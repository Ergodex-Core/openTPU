#!/usr/bin/env python3
"""Pinned data-only tokenizer checkpoint acquisition/reader; no remote code loader."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import time
import urllib.request
import numpy as np

MODEL = 'Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice'
REVISION = '85e237c12c027371202489a0ec509ded67b5e4b5'
SOURCE = '022e286b98fbec7e1e916cb940cdf532cd9f488e'
FILES = {
    'speech_tokenizer/model.safetensors': (682293092, '836b7b357f5ea43e889936a3709af68dfe3751881acefe4ecf0dbd30ba571258'),
    'model.safetensors': (1811626576, 'bc3c7e785eb961179c25450d1acff03f839e0002f2f3a5aeb67b5735c0fa2adb'),
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


class AudioCheckpoint:
    def __init__(self, path):
        self.path = Path(path)
        expected, sha = FILES['speech_tokenizer/model.safetensors']
        if self.path.stat().st_size != expected or digest(self.path) != sha:
            raise ValueError('decoder checkpoint size/LFS checksum mismatch')
        with self.path.open('rb') as f:
            size, = struct.unpack('<Q', f.read(8))
            if size != 63320:
                raise ValueError('pinned tokenizer header length mismatch')
            raw = f.read(size)
        self.header, self.offset = json.loads(raw), size + 8
        entries = [v for k, v in self.header.items() if k != '__metadata__']
        if len(entries) != 496 or any(e['dtype'] != 'F32' for e in entries):
            raise ValueError('pinned tokenizer tensor contract mismatch')
        cursor = 0
        for e in sorted(entries, key=lambda v: v['data_offsets'][0]):
            start, stop = e['data_offsets']
            if start != cursor or stop - start != 4 * __import__('math').prod(e['shape']):
                raise ValueError('invalid tokenizer tensor span')
            cursor = stop
        if self.offset + cursor != expected:
            raise ValueError('tokenizer data extent mismatch')
        self.header_sha256 = hashlib.sha256(raw).hexdigest()

    def tensor(self, name, shape):
        e = self.header[name]
        start, stop = e['data_offsets']
        if e['dtype'] != 'F32' or tuple(e['shape']) != tuple(shape):
            raise ValueError(f'{name}: decoder tensor shape/dtype mismatch')
        a = np.memmap(self.path, mode='r', dtype='<f4', offset=self.offset + start, shape=tuple(shape)).copy()
        if not np.isfinite(a).all():
            raise ValueError(f'{name}: nonfinite decoder parameter')
        return a


def acquire(directory, include_main=False):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((directory/'official-model-metadata.json').read_text())
    if metadata.get('sha') != REVISION:
        raise ValueError('pinned official metadata revision mismatch')
    records = []
    started = time.monotonic()
    for name, (expected, sha) in FILES.items():
        if name == 'model.safetensors' and not include_main:
            continue
        entry = next(e for e in metadata['siblings'] if e['rfilename'] == name)
        if entry['size'] != expected or entry['lfs']['sha256'] != sha:
            raise ValueError('official LFS metadata mismatch')
        path = directory / ('audio-model.safetensors' if name.startswith('speech_') else 'model.safetensors')
        if path.exists():
            if path.stat().st_size != expected or digest(path) != sha:
                raise ValueError('existing checkpoint failed verification; leave it intact')
        else:
            part = path.with_suffix('.part')
            if part.exists():
                raise RuntimeError('partial download exists; inspect original job before retrying')
            request = urllib.request.Request(f'https://huggingface.co/{MODEL}/resolve/{REVISION}/{name}',
                                            headers={'User-Agent':'OpenTPU-Qwen-TTS-data-provenance/1'})
            h, count = hashlib.sha256(), 0
            with urllib.request.urlopen(request, timeout=60) as response, part.open('xb') as out:
                while True:
                    block = response.read(8 << 20)
                    if not block:
                        break
                    count += len(block)
                    if count > expected or time.monotonic() - started > 1200:
                        raise RuntimeError('bounded checkpoint download exceeded size/time limit')
                    h.update(block)
                    out.write(block)
                    (directory/'download-progress.json').write_text(json.dumps({'file':name,'bytes':count,'expected':expected}))
            if count != expected or h.hexdigest() != sha:
                raise ValueError('download LFS verification failed; preserve partial file for inspection')
            part.rename(path)
        record = dict(model_id=MODEL, model_revision=REVISION, source_revision=SOURCE,
                      file=name, local_path=str(path), file_bytes=expected, file_sha256=sha,
                      verified_lfs_sha256=True, license='Apache-2.0', remote_code_executed=False)
        (directory/('audio-checkpoint-manifest.json' if name.startswith('speech_') else 'checkpoint-manifest.json')).write_text(json.dumps(record,indent=2)+'\n')
        records.append(record)
        print(json.dumps(record),flush=True)
    (directory/'acquisition-complete.json').write_text(json.dumps({'files':records,'seconds':time.monotonic()-started},indent=2)+'\n')


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory',type=Path,required=True)
    p.add_argument('--include-main',action='store_true')
    a=p.parse_args()
    acquire(a.directory,a.include_main)
