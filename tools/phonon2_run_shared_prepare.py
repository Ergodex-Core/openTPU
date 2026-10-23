"""Prepare device artifacts using verified common-runtime source, offline only."""
import argparse
import json
import os
from pathlib import Path
import runpy
import sys
from tools.phonon2_common_runtime import verify
from tools.phonon2_runner_runtime import manifest as host_manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runtime-root', required=True)
    a, remaining = p.parse_known_args()
    root, runtime = Path(__file__).resolve().parents[1], Path(a.runtime_root).resolve()
    common = json.loads((root/'docs/phonon2/tdt_runtime_dependency.json').read_text())['runtime_manifest']
    actual = json.loads((runtime/'source_manifest.json').read_text())
    allowed = [common, host_manifest(root), host_manifest(root, include_test_fake=True)]
    if actual not in allowed:
        raise RuntimeError('unrecognized common/host runtime source manifest')
    verify(runtime, actual)
    sys.path.insert(0, str(runtime))
    import opentpu
    opentpu.__path__.append(str(root/'opentpu'))
    sys.path.insert(0, str(root))
    os.environ['OTPU_TARGET'] = 'f2-hbm:p4rl8a64c150'
    sys.argv = ['tools.phonon2_prepare_runner', '--root', str(root), *remaining]
    runpy.run_module('tools.phonon2_prepare_runner', run_name='__main__')


if __name__ == '__main__':
    main()
