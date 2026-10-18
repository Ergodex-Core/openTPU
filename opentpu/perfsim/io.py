"""Compatible input bundles and reproducibility receipts; no evaluator mutation."""
from dataclasses import asdict, fields
import hashlib
import json
from pathlib import Path
import platform
import mmap
import subprocess
import numpy as np
from .. import isa as I
from ..isasim import Config

ROOT=Path(__file__).resolve().parents[2]


def digest(path):
    with open(path,'rb') as f:
        if Path(path).stat().st_size == 0:
            return hashlib.sha256(b'').hexdigest()
        with mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ) as data:
            return hashlib.sha256(data).hexdigest()


def state_hash(state):
    return {k:hashlib.sha256(getattr(state,k).tobytes()).hexdigest()
            for k in ('dram','tmem','act','ascale','adoff')}


def source_hashes():
    # Hash this simulator and its evaluator. Upstream ISA/compiler/RTL are identified
    # by the repository revision, not a claim of a separately clean working tree.
    paths=[*ROOT.glob('opentpu/perfsim/*.py'),*ROOT.glob('tests/perfsim/**/*.py'),*ROOT.glob('tests/perfsim/fixtures/*.json'),
           *ROOT.glob('tools/perfsim/**/*')]
    return {str(p.relative_to(ROOT)):digest(p) for p in sorted(paths)
            if p.is_file() and '__pycache__' not in p.parts and 'receipts' not in p.parts}


def manifest(inputs):
    # Read ref metadata directly: source hashes identify dirty working files without
    # invoking git status (which can block on host fsmonitor/file-provider services).
    dot=ROOT/'.git'
    gd=dot if dot.is_dir() else Path(dot.read_text().strip().split(': ',1)[1])
    common=(gd/(gd/'commondir').read_text().strip()).resolve() if (gd/'commondir').exists() else gd
    head=(gd/'HEAD').read_text().strip()
    branch=head[5:] if head.startswith('ref: ') else None
    commit=head if branch is None else None
    if branch:
        for path in (gd/branch,common/branch):
            if path.exists(): commit=path.read_text().strip(); break
        if commit is None and (common/'packed-refs').exists():
            for line in (common/'packed-refs').read_text().splitlines():
                if line.endswith(' '+branch): commit=line.split()[0]; break
    return dict(schema_version=1,git_commit=commit,branch=branch,
                working_tree_status='content-addressed source snapshot; cleanliness not inferred',
                python=platform.python_version(),numpy=np.__version__,
                inputs={str(p):digest(p) for p in inputs},sources=source_hashes(),
                dependency_lock=json.loads((ROOT/'tools/perfsim/dependencies.lock.json').read_text()),
                source_hash_scope='perfsim implementation and evaluator; upstream ISA/compiler/RTL identified by git commit',
                label='opentpu-perfsim-estimate',physical_implementation_validated=False)


def write_json(path, value):
    Path(path).write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')


def save_bundle(path,cfg,program,image,*,args=(),labels=None,metadata=None):
    path=Path(path); path.mkdir(parents=True,exist_ok=True)
    (path/'prog_0.hex').write_text(''.join(f'{int(w):08x}\n' for w in I.assemble(program)))
    data=np.pad(np.asarray(image,np.uint8),(0,-len(image)%4))
    data.view('<u4').astype('>u4').tofile(path/'dram_0.bin')
    write_json(path/'input.json',dict(config=asdict(cfg),instruction_counts=[len(program)],args=list(args),
                                    labels=labels or {},metadata=metadata or {}))


def load_bundle(path):
    path=Path(path)
    obj=json.loads((path/'input.json').read_text())
    cfgdict=obj['config_export'] if 'config_export' in obj else obj['config']
    unknown=set(cfgdict)-{f.name for f in fields(Config)}
    if unknown:
        raise ValueError(f'unsupported architecture fields in bundle: {sorted(unknown)}; not silently dropped')
    cfg=Config(**cfgdict)
    if cfg.S!=1:
        raise ValueError('bundle must contain one slice; multi-slice timing unsupported')
    n=obj['instruction_counts'][0]
    words=[int(v,16) for v in (path/'prog_0.hex').read_text().split()]
    if len(words)>8*n:
        raise ValueError('instruction count inconsistent with hex file')
    words += [0]*(8*n-len(words)) # rtlsim/LPN bundles truncate trailing zero words
    prog=[I.Instr.decode(words[k:k+8]) for k in range(0,len(words),8)]
    image=np.fromfile(path/'dram_0.bin',dtype='>u4').astype('<u4').view(np.uint8)
    if image.size>cfg.DRAM_BYTES:
        raise ValueError('DRAM image exceeds configuration')
    return cfg,prog,image,obj
