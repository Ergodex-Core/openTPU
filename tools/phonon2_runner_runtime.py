"""Prepare/verify pinned common plus Board host sources as local Git data only.

No imports of the host driver, transport creation, installation or card access.
The optional FakeTransport source is solely for offline native-protocol tests.
"""
import argparse
import json
from pathlib import Path
from tools.phonon2_common_runtime import prepare, verify


def manifest(root, *, include_test_fake=False):
    binding = json.loads((Path(root)/'docs/phonon2/runner_source_binding.json').read_text())
    files = binding['runtime_files'] | binding['host_files']
    if include_test_fake:
        # Audited data from the same revision, unused in the physical path.
        files = files | {'opentpu/host/fake.py': '2798a73585dd6c196b36998c9f22d364659a0534dfe3807b9c9e48ab04d1f296'}
    return {'revision': binding['runtime_revision'], 'files': files}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepare-from-git'); p.add_argument('--destination'); p.add_argument('--verify-root')
    p.add_argument('--include-test-fake', action='store_true')
    a = p.parse_args()
    m = manifest(Path(__file__).resolve().parents[1], include_test_fake=a.include_test_fake)
    if a.prepare_from_git and a.destination and not a.verify_root:
        prepare(a.prepare_from_git, a.destination, m)
    elif a.verify_root and not a.prepare_from_git and not a.destination:
        verify(a.verify_root, m)
    else:
        p.error('choose --verify-root, or --prepare-from-git plus --destination')
    print('pinned common/host runtime verified; no device access')


if __name__ == '__main__':
    main()
