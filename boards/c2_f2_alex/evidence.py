#!/usr/bin/env python3
"""Offline portable C2/Alex evidence verification; never executes bundled code.

The manifest SHA256 must come from an independently trusted collection receipt.
Original absolute paths are data, resolved only through explicit role mappings.
No live HDK/git, AWS, FPGA, archive extraction, publication, or file writes occur.

Schema 1 uses bundle-manifest.json, excluded from its own ``files`` map. The
caller supplies that file's SHA256 from an independent collection receipt.
``files`` maps every regular relative path to {sha256, bytes}; no extra files or
empty directories are accepted. ``roots`` maps each ROLES entry to
{original_absolute_path, bundle_relative_path}. ``source_snapshots`` separately
maps each SOURCE_ROLES entry to {revision, bundle_relative_path}. ``receipts``
maps each RECEIPTS entry to {relative_path, sha256}. ``hardware`` contains the
original source_revision and its eight-character build_id. No receipt is edited
to relocate it. The dependencies role is the observer directory containing the
reviewed attest_ip_dependencies.py and attestation/{identity,before,complete}.json.
The authorized CPU builder instead preserves a separate successful synth-retry
root with complete.json, identity.json, before/after dependency inventories and
byte-identical original private-IP provenance. A failed original synthesis is
never promoted to completion. instance_id identifies the build host, not an
authorized FPGA runtime target.

This is evidence replay under a trusted collection hash, not authentication of
an unknown submitter's assertions. Archive preflight never grants qualification.
"""
from __future__ import annotations

import argparse
import codecs
import datetime
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
import tarfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MANIFEST = 'bundle-manifest.json'
# INSTANCE remains the sole physical FPGA target; manifests identify the builder.
INSTANCE = 'i-0c78de2253f66519d'
CPU_BUILDER = 'i-06cf8792fb8f15534'
BUILDERS = (INSTANCE, CPU_BUILDER)
FRESH_SOURCE = '5ad534f62adc53d2a0455e72835affd72dca633f'
FRESH_DRIVER_SHA256 = '795503db92c74432d09803a4b36b10f73d16d9dc8431e6b9dae52d8847e608ad'
FRESH_IDENTITY_SHA256 = '5cf454ae2e4f82c5d1f54ec217d1014bf1d50bae6b76d42cf17f4bca2d22fcd7'
FRESH_SOURCE_RECORD_SHA256 = '8d1d35b750db5404cccb63ba9b29c36ce26d108aaddbbfdf69aa61e48880b72f'
# Reviewed CPU-only retry: immutable original IP, audit-only source change.
REVIEWED_CPU_RETRY_DRIVERS = frozenset({'76c9d44d7af3dc0b22e7cb029187d7fb2a1e2665278bd6ce772299765796b2fe'})
FRESH_IP_RECEIPT = {'bytes': 187446, 'sha256': '86d68a579e3483480baef9f864d841f17d9835443000cdbf39eeb87ff720a553'}
MAX_FILES, MAX_TOTAL, MAX_FILE, MAX_MANIFEST = 100000, 64 << 30, 16 << 30, 64 << 20
MAX_SUPPLEMENTAL_TEXT = 256 << 20
ROLES = ('ip', 'synth', 'link', 'implement', 'refine', 'qualify', 'simulation', 'simulation_private', 'installed', 'dependencies')
SOURCE_ROLES = ('hardware', 'ip', 'link', 'implement', 'refine', 'qualify', 'simulation')
RECEIPTS = {'ip': 'ip-complete.json', 'synth': 'synth-complete.json',
            **{stage: 'stage-result.json' for stage in ('link', 'implement', 'refine', 'qualify')},
            'simulation': 'simulation-complete.json', 'dependencies': 'attestation/complete.json'}
def schema_roles(schema):
    return ROLES + (('review',) if schema == 2 else ())


def schema_sources(schema):
    return SOURCE_ROLES + (('review',) if schema == 2 else ())


def schema_receipts(schema, instance=INSTANCE):
    return {**RECEIPTS, **({'dependencies': 'complete.json'} if instance == CPU_BUILDER else {}), **({'review': 'review-result.json'} if schema == 2 else {})}


DEPENDENCY_SOURCE = '264d84134da11630622002cd869cf88b07e49aec'
OBSERVER_SHA256 = 'e243c6f260187a26dbea41811c0d5c068a265401f4fa8fd821425bfdd56b4615'
SYNCHRONOUS_WRAPPER_SHA256 = '006e3feacbf3e15c16e92187fe0e4ce1fca9e749bc445f1ae7a71d9a302f0cbb'
REUSED_IP_RECEIPT_SHA256 = 'c21aae37ad72606811b17c9b212aa26d0ef6f8b31196494057c86eb16bd39799'
# LINK is a structural gate, deliberately not route acceptance. Preserve the
# reviewed pending-only helpers from 8af52428/1639daac and successful cec9b011
# LINK (receipt 5c79489ceb14949b1d7c5051787e44fe959a8402882d38ade93c32f6f1db7ec6)
# when a later final-route classifier is reviewed. The original IP receipt retains its old audit
# input, which did not synthesize the retry hardware. Final QUALIFY, hardware
# audit, and all other inputs still require current reviewed bytes.
# The 0d2a5082 LINK helper predates only the routed receive-flow predicate;
# audit_linked uses the unchanged structural fabric path (routed=0).
REVIEWED_HISTORICAL = {('link', 'boards/c2_f2_alex/qualify_c2_alex.tcl'):
                      {'6306dc7dc5f8c9112ce62b4140ccc0aa2846cd05fb624ef48b1af87140ab623c',
                       'c072db502318cadcd4a38b47438fcaef69a5b91fe7090374a629558e502eef9d',
                       'ab5b795d03a0a6663996536b69d65187df887be550bcf9988d51e27b263dc38a'},
                      ('ip', 'boards/c2_f2_alex/audit_c2_alex.tcl'):
                      {'5e1999a5f1ad2f515c9fac78bfd616c5e4932bce4ee07c77cc38f63916df1c2e',
                       'a9d91bef071f935089dc8cbc157d053253f00fe7dd667d5e6dca7ccb8cd4d349'}}
# Preserve the executed stage runner only for completed pre-qualification roles.
for _stage in ('link', 'implement', 'refine'):
    REVIEWED_HISTORICAL[(_stage, 'boards/c2_f2_alex/implement.py')] = {
        '812b64f8afab2c3c8ff2d5235d6cdcb0eaec38b486a4ff2f786360144e956295'}

DEPENDENCY_TREES = tuple(f'{stage}/ip_project/{project}.{suffix}'
                        for stage, project in (('ip', 'f2_ip'), ('pcis', 'pcis'), ('clocks', 'clocks'))
                        for suffix in ('srcs', 'gen'))
FINAL_REPORTS = ('objects.tcldata', 'summary.tcldata', 'supplemental.tcldata', 'timing.rpt', 'route_status.rpt',
                 'setup.rpt', 'hold.rpt', 'drc.rpt', 'methodology.rpt', 'cdc.rpt', 'bus_skew.rpt',
                 'clock_interaction.rpt', 'exception_coverage.rpt', 'ignored_exceptions.rpt',
                 'waivers.rpt', 'utilization.rpt', 'timing_all_ids.xdc')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(value):
    require(isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value), 'Invalid SHA256')
    return value


def unique(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, 'Duplicate JSON key: ' + key)
        result[key] = value
    return result


def decode(raw):
    return json.loads(raw, object_pairs_hook=unique,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite JSON value')))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def relative(name):
    require(isinstance(name, str) and name and len(name) <= 4096, 'Invalid member name')
    require(not name.startswith('/') and not any(ord(c) < 32 or ord(c) == 127 for c in name)
            and '\\' not in name, 'Unsafe member name')
    require(all(part not in ('', '.', '..', '.git') for part in name.split('/')), 'Unsafe member path')
    return name


def absolute(name):
    require(isinstance(name, str) and name.startswith('/') and name != '/', 'Invalid original absolute path')
    relative(name[1:])
    return name


def metadata(value):
    require(isinstance(value, dict) and set(value) == {'sha256', 'bytes'}, 'Malformed file metadata')
    digest(value['sha256'])
    require(type(value['bytes']) is int and 0 <= value['bytes'] <= MAX_FILE, 'Invalid file length')


def manifest_data(raw, expected_sha256):
    require(len(raw) <= MAX_MANIFEST and hashlib.sha256(raw).hexdigest() == digest(expected_sha256),
            'Manifest differs from independently trusted SHA256')
    m = decode(raw)
    require(set(m) == {'schema', 'kind', 'instance_id', 'hardware', 'roots', 'source_snapshots', 'receipts', 'files'},
            'Unknown/missing manifest fields')
    require(type(m['schema']) is int and m['schema'] in (1, 2) and m['kind'] == 'c2-alex-portable-evidence' and m['instance_id'] in BUILDERS,
            'Wrong evidence schema or dedicated instance')
    hardware = m['hardware']
    require(set(hardware) == {'source_revision', 'build_id'} and
            re.fullmatch('[0-9a-f]{40}', hardware['source_revision']) and
            hardware['build_id'] == hardware['source_revision'][:8], 'Invalid hardware revision/build ID')
    require(set(m['roots']) == set(schema_roles(m['schema'])) and set(m['source_snapshots']) == set(schema_sources(m['schema'])), 'Incomplete relocation roles')
    prefixes = {}
    for role, row in m['roots'].items():
        require(set(row) == {'original_absolute_path', 'bundle_relative_path'}, 'Malformed run relocation')
        absolute(row['original_absolute_path']); relative(row['bundle_relative_path'])
        prefixes['run:' + role] = row['bundle_relative_path']
    for role, row in m['source_snapshots'].items():
        require(set(row) == {'revision', 'bundle_relative_path'} and re.fullmatch('[0-9a-f]{40}', row['revision']),
                'Malformed source snapshot')
        prefixes['source:' + role] = relative(row['bundle_relative_path'])
    for first, a in prefixes.items():
        for second, b in prefixes.items():
            if first == second:
                continue
            nested_private = first == 'run:simulation_private' and second == 'run:simulation'
            require(a != b and (not a.startswith(b + '/') or nested_private), 'Ambiguous overlapping relocation roots')
    for key in ('original_absolute_path', 'bundle_relative_path'):
        require(m['roots']['simulation_private'][key] == m['roots']['simulation'][key] + '/private-ip',
                'Simulation copy relocation must preserve private-ip ancestry')
    originals = {role: row['original_absolute_path'] for role, row in m['roots'].items()}
    for a, first in originals.items():
        for b, second in originals.items():
            if a != b:
                require(first != second and (not first.startswith(second + '/') or
                        (a, b) == ('simulation_private', 'simulation') or
                        ((a, b) == ('synth', 'dependencies') and first == second + '/synthesis')),
                        'Overlapping original run mappings')
    files = m['files']
    require(isinstance(files, dict) and 0 < len(files) <= MAX_FILES and MANIFEST not in files, 'Invalid file inventory')
    total = 0
    for name, row in files.items():
        relative(name); metadata(row); total += row['bytes']
        require(any(name.startswith(prefix + '/') for prefix in prefixes.values()), 'File outside mapped evidence roots')
    require(total <= MAX_TOTAL, 'Expanded evidence exceeds size bound')
    require(set(m['receipts']) == set(schema_receipts(m['schema'], m['instance_id'])), 'Incomplete receipt inventory')
    for role, filename in schema_receipts(m['schema'], m['instance_id']).items():
        record = m['receipts'][role]
        expected = m['roots'][role]['bundle_relative_path'] + '/' + filename
        require(set(record) == {'relative_path', 'sha256'} and record['relative_path'] == expected and
                expected in files and digest(record['sha256']) == files[expected]['sha256'], 'Receipt relocation/hash mismatch')
    return m


def no_symlink(path):
    for item in (path, *path.parents):
        require(not item.is_symlink(), 'Symlink path component: ' + str(item))


def parent_names(names):
    return {str(parent) for name in names for parent in PurePosixPath(name).parents if str(parent) != '.'}


class Bundle:
    def __init__(self, root, manifest_sha256):
        self.root = Path(os.path.abspath(root))
        no_symlink(self.root)
        require(self.root.is_dir(), 'Bundle root is not a directory')
        no_symlink(self.root / MANIFEST)
        require((self.root / MANIFEST).stat().st_size <= MAX_MANIFEST, 'Oversized manifest')
        self.manifest = manifest_data((self.root / MANIFEST).read_bytes(), manifest_sha256)
        self.files = self.manifest['files']
        allowed_dirs = parent_names(self.files)
        actual = set()
        for directory, dirs, files in os.walk(self.root, followlinks=False):
            for name in dirs + files:
                path = Path(directory) / name
                mode = path.lstat().st_mode
                require(stat.S_ISDIR(mode) or stat.S_ISREG(mode), 'Link or special object in bundle')
                relative(path.relative_to(self.root).as_posix())
                if stat.S_ISDIR(mode):
                    require(path.relative_to(self.root).as_posix() in allowed_dirs, 'Extra bundle directory')
            actual.update((Path(directory) / name).relative_to(self.root).as_posix() for name in files)
        require(actual == set(self.files) | {MANIFEST}, 'Missing/extra bundle file')
        for name, row in self.files.items():
            path = self.root / name
            require(path.stat().st_size == row['bytes'] and sha(path) == row['sha256'], 'Changed bundle file: ' + name)

    def path(self, role, name=''):
        prefix = self.manifest['roots'][role]['bundle_relative_path']
        result = prefix + ('/' + relative(name) if name else '')
        return self.root / result

    def original(self, role, name=''):
        return self.manifest['roots'][role]['original_absolute_path'] + ('/' + relative(name) if name else '')

    def names(self, role):
        prefix = self.manifest['roots'][role]['bundle_relative_path'] + '/'
        return {name[len(prefix):] for name in self.files if name.startswith(prefix)}

    def record(self, role, name):
        key = self.path(role, name).relative_to(self.root).as_posix()
        require(key in self.files, 'Required evidence missing: ' + key)
        return self.files[key]

    def read(self, role, name, expected=None):
        row = self.record(role, name)
        require(expected is None or row['sha256'] == digest(expected), 'Receipt-chain hash mismatch: ' + name)
        require(row['bytes'] <= MAX_MANIFEST, 'Oversized JSON evidence')
        return decode(self.path(role, name).read_bytes())

    def text(self, role, name):
        require(0 < self.record(role, name)['bytes'] <= MAX_MANIFEST, 'Empty/oversized raw evidence: ' + name)
        return self.path(role, name).read_text()

    def supplemental_text(self):
        # Actual per-bit linked evidence exceeds the 64 MiB JSON/small-report
        # limit. Validate this one bounded report incrementally; never raise the
        # manifest/JSON parsing limit or materialize its entire text in memory.
        role, name = 'qualify', 'qualification/supplemental.tcldata'
        row = self.record(role, name)
        require(0 < row['bytes'] <= MAX_SUPPLEMENTAL_TEXT, 'Empty/oversized supplemental evidence')
        decoder = codecs.getincrementaldecoder('utf-8')('strict')
        digest_value = hashlib.sha256()
        total = 0
        with self.path(role, name).open('rb') as stream:
            while chunk := stream.read(1 << 20):
                total += len(chunk)
                require(total <= MAX_SUPPLEMENTAL_TEXT, 'Supplemental evidence grew beyond limit')
                decoder.decode(chunk, final=False)
                digest_value.update(chunk)
            decoder.decode(b'', final=True)
        require(total == row['bytes'] and digest_value.hexdigest() == row['sha256'],
                'Supplemental evidence changed during verification')

    def inventory(self, role, records, *, required=(), exact=None):
        require(isinstance(records, dict) and records and set(required) <= records.keys(), 'Incomplete bound file inventory')
        for name, wanted in records.items():
            relative(name); metadata(wanted)
            require(self.record(role, name) == wanted, 'Frozen inventory mismatch: ' + role + '/' + name)
        if exact is not None:
            require(set(records) == exact, 'Inventory omits/adds bound files: ' + role)

    def source(self, role, revision, records, *, current=()):
        snapshot = self.manifest['source_snapshots'][role]
        require(snapshot['revision'] == revision, 'Source revision role mismatch: ' + role)
        require(isinstance(records, dict) and records, 'Missing source inventory')
        for name, wanted in records.items():
            relative(name); metadata(wanted)
            key = snapshot['bundle_relative_path'] + '/' + name
            require(self.files.get(key) == wanted, 'Source snapshot mismatch: ' + role + '/' + name)
            if name in current:
                accepted = {sha(ROOT / name)} | REVIEWED_HISTORICAL.get((role, name), set())
                require(wanted['sha256'] in accepted, 'Source differs from reviewed local validator: ' + role + '/' + name)

    def receipt(self, role):
        record = self.manifest['receipts'][role]
        return self.read(role, schema_receipts(self.manifest['schema'], self.manifest['instance_id'])[role], record['sha256'])


def local_module(name):
    # These are reviewed local pure validators, never paths from the bundle.
    spec = importlib.util.spec_from_file_location('c2_alex_evidence_' + name, HERE / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    old = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = old
    return module


def kv(text, keys):
    result = {}
    for line in text.splitlines():
        require('=' in line, 'Malformed key/value evidence')
        key, value = line.split('=', 1)
        require(key in keys and key not in result, 'Unknown/duplicate key/value evidence')
        result[key] = value
    require(set(result) == set(keys), 'Missing key/value evidence')
    return result


def tcl_list(text):
    """Parse Tcl list data without substitution or evaluation (dict/list output)."""
    result, i = [], 0
    require(len(text) <= MAX_MANIFEST, 'Tcl evidence exceeds bound')
    while i < len(text):
        if text[i].isspace():
            i += 1; continue
        word, quoted, depth = [], text[i] == '"', 0
        if text[i] == '{':
            depth = 1; i += 1
            while i < len(text) and depth:
                char = text[i]
                if char == '\\' and i + 1 < len(text):
                    word.extend(text[i:i+2]); i += 2; continue
                if char == '{': depth += 1
                if char == '}': depth -= 1
                require(depth < 100, 'Tcl nesting exceeds bound')
                if depth: word.append(char)
                i += 1
            require(depth == 0, 'Unbalanced Tcl braces')
        else:
            if quoted: i += 1
            while i < len(text) and (quoted or not text[i].isspace()):
                char = text[i]
                if quoted and char == '"':
                    i += 1; quoted = False; break
                if char == '\\':
                    require(i + 1 < len(text), 'Trailing Tcl escape')
                    char = {'n': '\n', 'r': '\r', 't': '\t'}.get(text[i+1], text[i+1]); i += 1
                word.append(char); i += 1
            require(not quoted, 'Unclosed Tcl quote')
        require(i == len(text) or text[i].isspace(), 'Malformed Tcl list separator')
        result.append(''.join(word))
    return result


def tcl_dict(text):
    words = tcl_list(text)
    require(len(words) % 2 == 0, 'Odd Tcl dictionary')
    return unique(zip(words[::2], words[1::2]))


def fresh_dependency_attestation(x):
    """Replay a reviewed CPU retry without relabeling its failed predecessor."""
    complete = x.receipt('dependencies')
    require(complete.get('kind') == 'c2-m7a-reused-ip-synthesis-complete-pending-functional-validation'
            and complete.get('hardware_actions') is False and complete.get('route_qualified') is False
            and 'failure.json' not in x.names('dependencies'), 'Missing/failed CPU synthesis retry')
    # Receipt success and dependency equality are replayed, not inferred from
    # the driver's exit code or its dependencies_unchanged flag.
    bindings = {'identity': 'identity.json', 'driver': 'driver.py', 'source_record': 'source-record.json',
                'dependencies_before': 'dependencies-before.json', 'dependencies_after': 'dependencies-after.json'}
    for key, name in bindings.items():
        require(complete.get(key) == x.record('dependencies', name), 'CPU retry binding differs: ' + key)
    require(complete['driver']['sha256'] in REVIEWED_CPU_RETRY_DRIVERS, 'Unreviewed CPU synthesis retry driver')
    original_files = {name: x.record('dependencies', name) for name in
                      ('original-identity.json', 'original-driver.py', 'original-source-record.json')}
    require(complete.get('original_files') == original_files, 'Original private-IP provenance bindings differ')
    require(original_files['original-identity.json']['sha256'] == FRESH_IDENTITY_SHA256 and
            original_files['original-driver.py']['sha256'] == FRESH_DRIVER_SHA256 and
            original_files['original-source-record.json']['sha256'] == FRESH_SOURCE_RECORD_SHA256,
            'Unreviewed original private-IP executor')
    identity = x.read('dependencies', 'identity.json')
    original = x.read('dependencies', 'original-identity.json')
    source = x.read('dependencies', 'source-record.json')
    original_source = x.read('dependencies', 'original-source-record.json')
    before = x.read('dependencies', 'dependencies-before.json')
    after = x.read('dependencies', 'dependencies-after.json')
    require(identity.get('original_files') == original_files and identity.get('driver') == complete['driver'],
            'Retry identity provenance differs')
    require(source.get('support_verified') is True and original_source.get('support_verified') is True and
            all(row.get('hardware_actions') is False for row in (source, original_source, identity, original, before)),
            'CPU source support/hardware-action provenance differs')
    ipi, syni = (x.read(role, 'inputs.json') for role in ('ip', 'synth'))
    revision = x.manifest['hardware']['source_revision']
    require(identity.get('kind') == 'c2-m7a-reused-ip-synthesis-identity' and
            original.get('kind') == 'c2-m7a-fresh-ip-synthesis-identity', 'Wrong CPU executor identity')
    require(all(row.get('instance_id') == CPU_BUILDER for row in (complete, identity, original, before)),
            'CPU dependency builder mismatch')
    require(complete.get('source_revision') == identity.get('source_revision') == source.get('revision') ==
            syni.get('source_revision') == revision and source.get('frozen_inputs') == syni.get('source_inputs'),
            'Retry source differs from synthesized source')
    require(original.get('source_revision') == original_source.get('revision') == ipi.get('source_revision') == FRESH_SOURCE
            and original_source.get('frozen_inputs') == ipi.get('source_inputs'), 'Original private-IP source differs')
    audit = 'boards/c2_f2_alex/audit_c2_alex.tcl'
    require(set(source['frozen_inputs']) == set(original_source['frozen_inputs']) and
            all(row == source['frozen_inputs'][name] for name, row in original_source['frozen_inputs'].items() if name != audit),
            'CPU retry changed hardware inputs beyond its audit')
    require(original.get('source_record_sha256') == FRESH_SOURCE_RECORD_SHA256 and
            identity.get('source_record') == complete['source_record'], 'Source record identity differs')
    require(identity.get('hdk') == original.get('hdk') == x.original('installed') and
            identity.get('installed_inputs') == original.get('installed_inputs') == syni.get('installed_inputs') == ipi.get('installed_inputs'),
            'CPU retry changed installed tool inputs')
    unit, invocation = identity.get('unit'), identity.get('invocation_id')
    require(unit == f'otpu-c2-{revision[:8]}-synth-retry.service'
            and isinstance(invocation, str) and re.fullmatch('[0-9a-f]{32}', invocation), 'Invalid CPU retry invocation')
    state = identity.get('unit_state', {})
    require(complete.get('unit') == unit and complete.get('invocation_id') == before.get('invocation_id') == invocation
            and state.get('Id') == unit and state.get('InvocationID') == invocation and
            state.get('MemoryMax') == str(32 << 30) and state.get('CPUQuotaPerSecUSec') == '4s',
            'CPU retry unit identity/resource bound changed')
    require(before.get('kind') == 'c2-reused-generated-dependencies-before-synthesis'
            and before.get('source_revision') == revision, 'Wrong retry before-synthesis inventory')
    ip_record = x.record('ip', 'ip-complete.json')
    require(ip_record == FRESH_IP_RECEIPT, 'Unreviewed fresh private-IP receipt')
    require(before.get('ip_receipt') == after.get('ip_receipt') == identity.get('ip_receipt') == complete.get('ip_receipt') == ip_record and
            before.get('ip_run') == identity.get('ip_run') == complete.get('ip_run') == x.original('ip') and
            complete.get('synth_receipt') == x.record('synth', 'synth-complete.json') and
            complete.get('checkpoint') == x.record('synth', 'synth/post_synth.dcp') and
            complete.get('checkpoint_path') == x.original('synth', 'synth/post_synth.dcp'), 'CPU receipt/DCP ancestry differs')
    require(x.original('synth') == x.original('dependencies') + '/synthesis', 'CPU retry synthesis layout differs')
    command = x.read('dependencies', 'synthesis-command.json')
    require(isinstance(command, list) and len(command) == 12, 'Wrong CPU retry command')
    absolute(command[0]); absolute(source.get('source'))
    require(command[1:] == [source['source'] + '/boards/c2_f2_alex/build.py', '--hdk', x.original('installed'),
            '--out', x.original('synth'), '--ip-jobs', '1', '--stage', 'synth', '--ip-run', x.original('ip')],
            'CPU retry command source/run differs')
    exit_record = x.read('dependencies', 'synthesis-exit.json')
    require(exit_record.get('returncode') == 0, 'CPU retry synthesis failed')
    dates = [datetime.datetime.fromisoformat(row[key]) for row, key in
             ((identity, 'started_at'), (exit_record, 'finished_at'), (after, 'finished_at'), (complete, 'finished_at'))]
    require(all(t.utcoffset() == datetime.timedelta(0) for t in dates) and dates == sorted(dates),
            'CPU retry times out of order/not UTC')
    names = {name for name in x.names('ip') if any(name.startswith(tree + '/') for tree in DEPENDENCY_TREES)}
    require(before.get('files') == after.get('files'), 'Generated dependencies changed during CPU retry')
    x.inventory('ip', before.get('files'), exact=names)
    require(complete.get('dependency_files') == len(names) and
            all(any(name.startswith(tree + '/') for name in names) for tree in DEPENDENCY_TREES) and
            any(name.endswith(('.vh', '.svh')) for name in names), 'Incomplete CPU generated dependencies')


def dependency_attestation(x):
    """Replay the separately captured pre-synthesis includes proof, not its code."""
    if x.manifest['instance_id'] == CPU_BUILDER:
        return fresh_dependency_attestation(x)
    complete = x.receipt('dependencies')
    identity = x.read('dependencies', 'attestation/identity.json', digest(complete.get('identity_sha256')))
    before = x.read('dependencies', 'attestation/before.json', digest(complete.get('before_sha256')))
    schema = complete.get('schema')
    require(type(schema) is int and schema in (1, 2) and complete.get('state') == 'verified'
            and complete.get('hardware_actions') is False
            and complete.get('qualification') == 'source-provenance-only-not-timing-or-hardware'
            and 'attestation/failure.json' not in x.names('dependencies'), 'Missing/failed dependency attestation')
    hardware_revision = x.manifest['hardware']['source_revision']
    require(complete.get('source_revision') == identity.get('expected_source_revision') == hardware_revision,
            'Dependency observer hardware revision mismatch')
    require(complete.get('instance_id') == identity.get('instance_id') == INSTANCE,
            'Dependency observer instance mismatch')
    if schema == 1:
        require(complete.get('kind') == 'c2-alex-supplemental-ip-dependency-attestation'
                and hardware_revision == DEPENDENCY_SOURCE, 'Unreviewed asynchronous dependency observer')
        require(identity.get('observer_sha256') == OBSERVER_SHA256 and
                x.record('dependencies', 'attest_ip_dependencies.py')['sha256'] == OBSERVER_SHA256,
                'Unreviewed dependency observer')
        before_kind = 'c2-alex-ip-dependency-before-synthesis'
    else:
        require(complete.get('kind') == 'c2-alex-synchronous-ip-dependency-attestation'
                and identity.get('schema') == 2 and identity.get('kind') == 'c2-alex-synchronous-synthesis-identity'
                and hardware_revision != DEPENDENCY_SOURCE and complete.get('build_id') == hardware_revision[:8],
                'Unreviewed synchronous dependency attestation')
        require(identity.get('wrapper_sha256') == SYNCHRONOUS_WRAPPER_SHA256 and
                x.record('dependencies', 'run_synth_with_attestation.py')['sha256'] == SYNCHRONOUS_WRAPPER_SHA256,
                'Unreviewed synchronous synthesis wrapper')
        require(complete.get('ip_source_revision') == identity.get('ip_source_revision') == DEPENDENCY_SOURCE
                and x.manifest['receipts']['ip']['sha256'] == REUSED_IP_RECEIPT_SHA256,
                'Synchronous retry used another private-IP source/receipt')
        before_kind = 'c2-alex-ip-dependency-before-synchronous-synthesis'
        syni = x.read('synth', 'inputs.json')
        require(before.get('source_inputs') == complete.get('source_inputs') == syni.get('source_inputs')
                and before.get('ip_receipt') == x.record('ip', 'ip-complete.json'),
                'Synchronous hardware source/IP capture differs')
        require(complete.get('wrapper_command') == x.record('dependencies', 'attestation/command.json')
                and complete.get('wrapper_exit') == x.record('dependencies', 'attestation/exit.json')
                and x.read('dependencies', 'attestation/exit.json') == {'returncode': 0},
                'Synchronous synthesis command did not complete')
        command = x.read('dependencies', 'attestation/command.json')
        require(isinstance(command, list) and len(command) == 12, 'Wrong synchronous synthesis command')
        absolute(command[0]); absolute(identity.get('source'))
        require(command[1:] == [identity['source'] + '/boards/c2_f2_alex/build.py', '--hdk', x.original('installed'),
                '--out', x.original('synth'), '--stage', 'synth', '--ip-run', x.original('ip'), '--ip-jobs', '1'],
                'Synchronous synthesis command used another source/run')
    ipi = x.read('ip', 'inputs.json')
    require(ipi.get('source_revision') == DEPENDENCY_SOURCE, 'Dependency attestation used another private-IP revision')
    require(before.get('schema') == schema and before.get('kind') == before_kind
            and before.get('identity_sha256') == complete['identity_sha256'], 'Invalid before-synthesis proof')
    unit, invocation = identity.get('unit'), identity.get('invocation_id')
    require(isinstance(unit, str) and isinstance(invocation, str) and re.fullmatch('[0-9a-f]{32}', invocation),
            'Wrong observer unit/invocation')
    if schema == 1:
        require(re.fullmatch(r'otpu-c2-alex-264d8413-[a-z0-9-]+\.service', unit), 'Wrong observer unit')
    else:
        require(unit == f'otpu-c2-alex-{hardware_revision[:8]}-synth-retry.service', 'Wrong synchronous synthesis unit')
    require(complete.get('unit') == unit and complete.get('invocation_id') == invocation, 'Observer unit binding changed')
    for row in (identity.get('initial_unit', {}), before.get('unit', {}), complete.get('final_unit', {})):
        require(row.get('Id') == unit and row.get('InvocationID') == invocation and row.get('Result') == 'success'
                and row.get('ActiveState') in ('active', 'activating'), 'Observer unit state changed/failed')
    dates = [identity.get('started_utc'), before.get('started_utc'), before.get('finished_utc')]
    if schema == 2: dates.append(complete.get('launch_started_utc'))
    dates.append(complete.get('finished_utc'))
    dates = [datetime.datetime.fromisoformat(value) for value in dates]
    require(all(value.utcoffset() == datetime.timedelta(0) for value in dates) and dates == sorted(dates),
            'Observer timestamps out of order/not UTC')
    for role in ('ip', 'synth'):
        require(identity.get(role + '_run') == complete.get(role + '_run') == x.original(role),
                'Dependency observer relocation mismatch')
    bindings = {'ip_inputs': ('ip', 'inputs.json'), 'ip_receipt': ('ip', 'ip-complete.json'),
                'synth_inputs': ('synth', 'inputs.json'), 'synth_receipt': ('synth', 'synth-complete.json'),
                'ip_binding': ('synth', 'ip-binding.json'), 'synth_command': ('synth', 'synth/command.json'),
                'synth_exit': ('synth', 'synth/exit.json'), 'synth_marker': ('synth', 'synth/synth_done.txt'),
                'synth_checkpoint': ('synth', 'synth/post_synth.dcp')}
    for key, (role, name) in bindings.items():
        require(complete.get(key) == x.record(role, name), 'Dependency attestation binding mismatch: ' + key)
    require(before.get('inputs') == complete['ip_inputs'] and before.get('files') == complete.get('files'),
            'Dependencies changed during synthesis')
    names = {name for name in x.names('ip') if any(name.startswith(tree + '/') for tree in DEPENDENCY_TREES)}
    x.inventory('ip', complete.get('files'), exact=names)
    require(all(any(name.startswith(tree + '/') for name in names) for tree in DEPENDENCY_TREES)
            and any(name.endswith(('.vh', '.svh')) for name in names), 'Missing generated dependency trees/includes')


def prepared_names(b):
    """Frozen builder's exact pre-tool inventory (no generated project cache)."""
    result = {'source/rtl/' + name for name in b.CORE_SOURCES}
    result |= {'source/rtl/boards/f2/' + name for name in ('cl_opentpu.sv', 'otpu_native_axi.sv', 'otpu_hbm_geometry.sv')}
    result |= {stage + '/build.tcl' for stage in b.STAGES} | {'synth/audit_c2_alex.tcl'}
    result |= {stage + '/' + name for stage in ('synth', 'link', 'implement')
               for name in ('floorplan.tcl', 'core_timing.tcl', 'platform_constraints.tcl')}
    result |= {stage + '/rtl/' + name for stage in ('synth', 'link')
               for name in ('cl_opentpu.sv', 'otpu_native_axi.sv', 'otpu_hbm_geometry.sv')}
    return result | {stage + '/config.tcl' for stage in ('synth', 'link')}


def verify_bundle(root, manifest_sha256):
    """Replay frozen evidence under a new root; no commands or bundled imports."""
    x = Bundle(root, manifest_sha256)
    b, sim = local_module('build'), local_module('simulate')
    m = x.manifest; hardware = m['hardware']; stages = {}
    ip, synth = x.receipt('ip'), x.receipt('synth')
    ip_inputs = x.read('ip', 'inputs.json', ip['inputs_sha256'])
    inputs = x.read('synth', 'inputs.json', synth['inputs_sha256'])
    require(ip.get('schema') == 1 and ip.get('kind') == 'c2-alex-private-ip' and ip.get('qualification') == 'not-qualified', 'Wrong private-IP receipt')
    require(synth.get('schema') == 1 and synth.get('kind') == 'c2-alex-synthesis-only' and synth.get('qualification') == 'not-qualified', 'Wrong synthesis receipt')
    for role, value in (('ip', ip_inputs), ('hardware', inputs)):
        require(value.get('schema') == 1 and value.get('kind') == 'c2-alex-burst200-build-inputs' and value.get('source_status') == '', 'Wrong frozen input schema/status')
        require(re.fullmatch('[0-9a-f]{40}', value.get('source_revision', '')) and value.get('build_id') == value['source_revision'][:8], 'Invalid input revision')
        require((value.get('part'), value.get('core_khz'), value.get('hbm_mhz'), value.get('hbm_ports'), value.get('channel_bytes')) == (b.PART, 200000, 450, 8, 2**29), 'Wrong device/clock/window geometry')
        require(value.get('alex_revision') == b.ALEX_REVISION and value.get('archive_sha256') == b.ARCHIVE_SHA256 and value.get('hdk_revision') == b.HDK_REVISION and value.get('ip_revision') == b.IP_REVISION and value.get('support_manifest_sha256') == sha(b.SUPPORT / 'manifest.json'), 'Wrong reviewed source/IP provenance')
        require(value.get('hdk') == x.original('installed'), 'Installed-root relocation mismatch')
        x.source(role, value['source_revision'], value['source_inputs'], current=b.source_inputs())
        require(set(value['source_inputs']) == set(b.source_inputs()), 'Incomplete hardware/generation source inventory')
        support = decode((b.SUPPORT / 'manifest.json').read_bytes())
        support_records = {'boards/c2_f2_alex/support/' + name: {'sha256': checksum, 'bytes': (b.SUPPORT / name).stat().st_size}
                           for name, checksum in support['sources'].items()}
        support_records['boards/c2_f2_alex/support/manifest.json'] = {
            'sha256': value['support_manifest_sha256'], 'bytes': (b.SUPPORT / 'manifest.json').stat().st_size}
        x.source(role, value['source_revision'], support_records)
    require(inputs['source_revision'] == hardware['source_revision'] and
            inputs['build_id'] == hardware['build_id'], 'Hardware/IP identity differs from manifest')
    require(inputs['installed_inputs'] == ip_inputs['installed_inputs'], 'Synthesis and private IP used different installed inputs')
    x.inventory('installed', inputs['installed_inputs'], exact=x.names('installed'))
    shell_name = 'hdk/common/shell_stable/build/checkpoints/from_aws/cl_bb_routed.small_shell.dcp'
    require(inputs['installed_inputs'].get(shell_name, {}).get('sha256') == b.SHELL_SHA256, 'Wrong frozen shell checkpoint')
    x.inventory('ip', ip_inputs['prepared_files'], exact=prepared_names(b))
    x.inventory('synth', inputs['prepared_files'], exact=prepared_names(b))
    products = ip.get('products', {})
    product_names = {name for name in x.names('ip') if any(name.startswith(s + '/ip_project/') for s in ('ip', 'pcis', 'clocks')) and Path(name).suffix.lower() in b.PRODUCT_SUFFIXES}
    x.inventory('ip', products, required=(sim.BD, sim.WRAPPER, sim.PCIS), exact=product_names)
    x.read('ip', sim.BD)  # Reject duplicate JSON keys before the topology parser.
    b.validate_memory_bd(x.path('ip', sim.BD))
    for stage in ('ip', 'pcis', 'clocks'):
        require(ip_inputs['prepared_files'][stage + '/build.tcl'] == inputs['prepared_files'][stage + '/build.tcl'], 'Private-IP recipe differs from synthesis')
        require(x.text('ip', stage + '/ip_done.txt').strip() == 'IP_DONE' and x.read('ip', stage + '/exit.json') == {'returncode': 0}, 'Private-IP stage incomplete')
        x.text('ip', stage + '/console.log'); x.read('ip', stage + '/command.json')
    require(synth['ip_receipt_sha256'] == m['receipts']['ip']['sha256'] and x.read('synth', 'ip-binding.json') == {'run': x.original('ip'), 'receipt_sha256': m['receipts']['ip']['sha256']}, 'Wrong synthesis/IP binding')
    require(x.record('synth', 'synth/post_synth.dcp')['sha256'] == synth['checkpoint_sha256'] and x.read('synth', 'synth/exit.json') == {'returncode': 0} and x.text('synth', 'synth/synth_done.txt').strip() == 'C2_ALEX_SYNTH_ONLY_COMPLETE', 'Synthesis not complete')
    x.text('synth', 'synth/console.log'); x.read('synth', 'synth/command.json')
    dependency_attestation(x)
    for name in ('objects.tcldata', 'summary.tcldata', 'timing_all_ids.xdc', 'timing.rpt', 'drc.rpt', 'cdc.rpt', 'bus_skew.rpt', 'clock_interaction.rpt', 'exception_coverage.rpt', 'ignored_exceptions.rpt', 'methodology.rpt', 'waivers.rpt', 'utilization.rpt', 'REQUIRED_NEXT_STEPS.txt'):
        x.text('synth', 'synth/synth_audit/' + name)
    previous_role, previous_checkpoint = 'synth', 'synth/post_synth.dcp'
    for stage in ('link', 'implement', 'refine', 'qualify'):
        receipt = x.receipt(stage); stage_inputs = x.read(stage, 'stage-inputs.json', receipt['inputs_sha256'])
        require(receipt.get('schema') == 1 and receipt.get('kind') == 'c2-alex-checkpoint-stage' and receipt.get('stage') == stage and receipt.get('returncode') == 0 and receipt.get('hardware_actions') is False, 'Failed/wrong checkpoint stage')
        reviewed_collection = m['schema'] == 2 and stage == 'qualify'
        expected_status = 'pending-review' if reviewed_collection else ('candidate' if stage == 'implement' else 'complete')
        require(receipt.get('status') == expected_status and receipt.get('route_qualified') is (stage == 'qualify' and not reviewed_collection), 'Pending/unqualified checkpoint stage')
        require(stage_inputs.get('schema') == 1 and stage_inputs.get('kind') == 'c2-alex-checkpoint-stage-inputs' and stage_inputs.get('stage') == stage and stage_inputs.get('max_threads') == 4 and stage_inputs.get('resynthesis') is False and stage_inputs.get('hardware_actions') is False, 'Invalid stage execution contract')
        for key, expected in {'source_revision': hardware['source_revision'], 'build_id': hardware['build_id'], 'synth_run': x.original('synth'), 'synth_receipt_sha256': m['receipts']['synth']['sha256'], 'synth_inputs_sha256': synth['inputs_sha256'], 'ip_run': x.original('ip'), 'ip_receipt_sha256': m['receipts']['ip']['sha256'], 'hdk': x.original('installed'), 'shell_sha256': b.SHELL_SHA256}.items():
            require(stage_inputs.get(key) == expected, 'Stage identity/relocation mismatch: ' + key)
        wanted_previous = None if stage == 'link' else x.original(previous_role)
        wanted_receipt = None if stage == 'link' else m['receipts'][previous_role]['sha256']
        require(stage_inputs.get('previous_run') == wanted_previous and stage_inputs.get('previous_receipt_sha256') == wanted_receipt, 'Broken checkpoint ancestry')
        source_record = x.record(previous_role, previous_checkpoint)
        require(stage_inputs.get('source_checkpoint') == {'path': x.original(previous_role, previous_checkpoint), **source_record} and receipt.get('source_checkpoint_sha256') == source_record['sha256'], 'Source checkpoint hash/size/path mismatch')
        executors = {'boards/c2_f2_alex/implement.py', 'boards/c2_f2_alex/build.py'}
        prepared = {'run.tcl'}
        if stage in ('link', 'qualify'):
            prepared |= {'qualify_c2_alex.tcl', 'audit_c2_alex.tcl'}
            executors |= {'boards/c2_f2_alex/' + name for name in prepared - {'run.tcl'}}
        if stage == 'qualify':
            executors.add('boards/c2_f2_alex/simulate.py')
            if m['schema'] == 2:
                prepared.add('routed_host_ready.tcl')
                executors.add('boards/c2_f2_alex/routed_host_ready.tcl')
        if stage == 'implement': prepared |= {'floorplan.tcl', 'core_timing.tcl', 'platform_constraints.tcl'}
        require(set(stage_inputs['executor_sources']) == executors, 'Incomplete executor source inventory')
        x.source(stage, stage_inputs['executor_revision'], stage_inputs['executor_sources'], current=executors)
        x.inventory(stage, stage_inputs['prepared_files'], exact=prepared)
        if stage in ('link', 'qualify'):
            helpers = ('qualify_c2_alex.tcl', 'audit_c2_alex.tcl')
            if stage == 'qualify' and m['schema'] == 2: helpers += ('routed_host_ready.tcl',)
            for name in helpers:
                require(x.record(stage, name) == stage_inputs['executor_sources']['boards/c2_f2_alex/' + name],
                        'Stage executed a different audit helper')
        x.inventory(stage, receipt.get('outputs'), required=('run.tcl', 'command.json', 'console.log'), exact=x.names(stage) - {'stage-inputs.json', 'stage-result.json'})
        require('failure.json' not in x.names(stage), 'Stage has failure evidence')
        if stage == 'link':
            require(kv(x.text(stage, 'audit-result.txt'), ('structural_pass', 'route_qualified')) == {'structural_pass': '1', 'route_qualified': '0'}, 'Linked audit failed')
            require(x.text(stage, 'link_done.txt').strip() == 'C2_ALEX_LINK_COMPLETE_NOT_ROUTE_QUALIFIED', 'Link incomplete')
        if stage in ('implement', 'refine'):
            text = x.text(stage, 'implementation_result.txt'); timing = {}
            for key in ('setup_slack_ns', 'hold_slack_ns'):
                matches = re.findall('^' + key + r'=(.+)$', text, re.M)
                require(len(matches) == 1, 'Missing timing witness')
                timing[key] = float(matches[0]); require(math.isfinite(timing[key]), 'Nonfinite timing witness')
            require(timing == receipt.get('timing') and (stage != 'refine' or min(timing.values()) >= 0), 'Refinement timing failed/changed')
        checkpoint = 'post_link.dcp' if stage == 'link' else 'post_route.dcp'
        if stage == 'qualify':
            require(receipt.get('checkpoint') == x.original(previous_role, previous_checkpoint) and receipt.get('checkpoint_sha256') == source_record['sha256'], 'Qualifier DCP mismatch')
            require(kv(x.text(stage, 'audit-result.txt'), ('route_qualified', 'checkpoint_sha256')) == {'route_qualified': '0' if reviewed_collection else '1', 'checkpoint_sha256': source_record['sha256']}, 'Raw qualifier state or checkpoint differs')
        else:
            require(receipt.get('checkpoint') == checkpoint and receipt.get('checkpoint_sha256') == x.record(stage, checkpoint)['sha256'], 'Stage output checkpoint mismatch')
            previous_role, previous_checkpoint = stage, checkpoint
        stages[stage] = stage_inputs
    final_sha = x.record('refine', 'post_route.dcp')['sha256']
    if m['schema'] == 1:
        require('qualification/NOT_QUALIFIED.txt' not in x.names('qualify'), 'Explicit NOT_QUALIFIED evidence present')
    else:
        require('qualification/NOT_QUALIFIED.txt' in x.names('qualify'), 'Original pending collector evidence was removed')
    for name in FINAL_REPORTS:
        if name == 'supplemental.tcldata': x.supplemental_text()
        else: x.text('qualify', 'qualification/' + name)
    summary = tcl_dict(x.text('qualify', 'qualification/summary.tcldata'))
    require(summary.get('schema') == '1' and summary.get('stage') == 'final_route' and summary.get('hardware_measured') == '0' and summary.get('checkpoint_sha256') == final_sha and summary.get('checkpoint') == x.original('refine', 'post_route.dcp') and summary.get('blockers') == '', 'Final raw collection has physical blockers or mismatched DCP')
    if m['schema'] == 1:
        require(summary.get('route_qualified') == '1' and summary.get('pending_reviews') == '', 'Final raw qualification has pending review')
    else:
        require(summary.get('route_qualified') == '0' and summary.get('pending_reviews'), 'Original pending collection state was changed')
    simulation = x.receipt('simulation'); si = x.read('simulation', 'inputs.json', simulation['inputs_sha256'])
    require(simulation.get('schema') == 1 and simulation.get('kind') == 'c2-alex-vendor-fabric-simulation' and simulation.get('state') == 'passed' and simulation.get('hardware_actions') is False and simulation.get('qualification') == 'protocol-simulation-only-not-physical-timing', 'Vendor simulation not passed')
    require(stages['qualify'].get('simulation') == {'run': x.original('simulation'), 'receipt_sha256': m['receipts']['simulation']['sha256'], 'inputs_sha256': simulation['inputs_sha256']}, 'Qualification did not bind this vendor simulation')
    expected_sim = {'schema': 1, 'kind': 'c2-alex-vendor-fabric-simulation', 'ip_receipt_sha256': m['receipts']['ip']['sha256'], 'build_inputs_sha256': ip['inputs_sha256'], 'ip_source_revision': ip_inputs['source_revision'], 'ip_products': products, 'clocks_mhz': {'core': 200, 'host': 250, 'hbm_nominal': 450}, 'hbm_simulated_period_ps': 2222, 'hbm_model': 'functional-not-silicon-timing', 'hardware_actions': False, 'qualification': 'protocol-simulation-only-not-physical-timing', 'scenarios': [list(pair) for pair in sim.SCENARIOS]}
    require(all(si.get(key) == value for key, value in expected_sim.items()), 'Wrong simulation/IP/clock contract')
    require(si.get('runner_sha256') == sha(HERE / 'simulate.py') and si.get('builder_sha256') == sha(HERE / 'build.py') and si.get('sources') == {name: sha(path) for name, path in sim.SOURCES.items()}, 'Simulation source differs from reviewed files')
    source_records = {'boards/c2_f2_alex/simulate.py': {'sha256': si['runner_sha256'], 'bytes': (HERE / 'simulate.py').stat().st_size}, 'boards/c2_f2_alex/build.py': {'sha256': si['builder_sha256'], 'bytes': (HERE / 'build.py').stat().st_size}}
    source_records.update({path.relative_to(ROOT).as_posix(): {'sha256': si['sources'][name], 'bytes': path.stat().st_size} for name, path in sim.SOURCES.items()})
    x.source('simulation', m['source_snapshots']['simulation']['revision'], source_records, current=source_records)
    require(x.record('simulation_private', 'ip-complete.json')['sha256'] == m['receipts']['ip']['sha256'] and x.record('simulation_private', 'inputs.json')['sha256'] == ip['inputs_sha256'], 'Simulation copied a different IP receipt')
    x.inventory('simulation_private', products)
    private_files = {name for name in x.names('simulation_private') if any(name.startswith(stage + '/ip_project/') for stage in ('ip', 'pcis', 'clocks'))}
    x.inventory('simulation_private', si.get('simulation_ip_files'), exact=private_files)
    x.inventory('simulation', simulation.get('evidence'), required=('vivado_version.txt',))
    require('failure.json' not in x.names('simulation') and len(simulation.get('results', [])) == len(sim.SCENARIOS), 'Failed/incomplete vendor simulation')
    for (seed, phase), result in zip(sim.SCENARIOS, simulation['results']):
        prefix = f'seed-{seed}-phase-{phase}/'
        required = [prefix + name for name in ('inputs.json', 'command.json', 'console.log', 'exit.json', 'result.json', 'simulate.tcl')]
        required += [prefix + 'rtl/' + name for name in sim.SOURCES]
        require(set(required) <= simulation['evidence'].keys(), 'Scenario raw evidence not receipt-bound')
        require(x.read('simulation', prefix + 'exit.json') == {'returncode': 0} and sim.parse_result(x.text('simulation', prefix + 'console.log'), seed, phase) == result and x.read('simulation', prefix + 'result.json') == result, 'Scenario raw scoreboard differs')
        scenario_inputs = x.read('simulation', prefix + 'inputs.json')
        for name, row in scenario_inputs.items():
            require(x.record('simulation', prefix + relative(name)) == row, 'Scenario consumed input changed')
        require(set(scenario_inputs) == {'simulate.tcl'} | {'rtl/' + name for name in sim.SOURCES}, 'Scenario input inventory incomplete')
        for name in sim.SOURCES:
            require(x.record('simulation', prefix + 'rtl/' + name)['sha256'] == si['sources'][name], 'Scenario source differs')
        expected_script = sim.simulation_tcl(Path(x.original('simulation', prefix.rstrip('/'))), Path(x.original('simulation_private')), seed, phase)
        require(x.text('simulation', prefix + 'simulate.tcl') == expected_script, 'Scenario command topology/paths changed')
    if m['schema'] == 2:
        local_module('final_review').verify_review(x)
    # Check the bundle did not change during semantic validation.
    for name, row in x.files.items():
        require(sha(x.root / name) == row['sha256'], 'Bundle changed while verifying')
    require(sha(x.root / MANIFEST) == manifest_sha256, 'Manifest changed while verifying')
    return {'schema': 1, 'kind': 'verified-c2-alex-portable-evidence', 'instance_id': m['instance_id'],
            'manifest_sha256': digest(manifest_sha256), **hardware, 'route_qualified': True,
            'hardware_measured': False, 'final_dcp_sha256': final_sha,
            'final_dcp_relative_path': x.path('refine', 'post_route.dcp').relative_to(x.root).as_posix(),
            'synth_inputs_sha256': synth['inputs_sha256'],
            'synth_inputs_relative_path': x.path('synth', 'inputs.json').relative_to(x.root).as_posix(),
            'ip_receipt_sha256': m['receipts']['ip']['sha256'],
            'simulation_receipt_sha256': m['receipts']['simulation']['sha256'],
            'qualification_receipt_sha256': m['receipts']['review' if m['schema'] == 2 else 'qualify']['sha256'],
            **({'collection_receipt_sha256': m['receipts']['qualify']['sha256'], 'final_review_receipt_sha256': m['receipts']['review']['sha256']} if m['schema'] == 2 else {}),
            'files': len(x.files), 'bytes': sum(row['bytes'] for row in x.files.values())}


def inspect_archive(archive_path, manifest_sha256):
    """Bounded streaming inventory preflight only; does not extract or qualify."""
    path = Path(os.path.abspath(archive_path)); no_symlink(path)
    require(path.is_file(), 'Archive is not a regular file')
    archive_sha = sha(path)
    seen, directories, regular, manifest_raw, total, count = set(), set(), {}, None, 0, 0
    metadata_bytes = 0

    class BoundedInfo(tarfile.TarInfo):
        # tarfile reads extended metadata before yielding a member. Bound those
        # allocations too; they otherwise bypass the expanded payload checks.
        def check_header(self):
            nonlocal metadata_bytes
            require(0 <= self.size <= min(MAX_MANIFEST, 1 << 20), 'Oversized extended archive header')
            metadata_bytes += self.size
            require(metadata_bytes <= MAX_MANIFEST, 'Archive metadata size bound exceeded')

        def _proc_pax(self, archive):
            self.check_header()
            return super()._proc_pax(archive)

        def _proc_gnulong(self, archive):
            self.check_header()
            return super()._proc_gnulong(archive)

        def _proc_sparse(self, archive):
            raise ValueError('Sparse archive member')

        def _proc_gnusparse_00(self, *args):
            raise ValueError('Sparse archive member')

        _proc_gnusparse_01 = _proc_gnusparse_00
        _proc_gnusparse_10 = _proc_gnusparse_00

    with tarfile.open(path, mode='r|*', tarinfo=BoundedInfo, ignore_zeros=True) as archive:
        for member in archive:
            count += 1; require(count <= 2 * MAX_FILES, 'Archive member bound exceeded')
            name = member.name.removesuffix('/') if member.isdir() else member.name
            relative(name); require(name not in seen, 'Duplicate archive member'); seen.add(name)
            require(member.isfile() or member.isdir(), 'Link or special archive member')
            require(not member.sparse and not any(k.startswith(('GNU.sparse', 'SCHILY.realsize')) for k in member.pax_headers), 'Sparse archive member')
            if member.isdir():
                require(member.size == 0, 'Directory with nonzero size')
                directories.add(name); continue
            require(type(member.size) is int and 0 <= member.size <= (MAX_MANIFEST if name == MANIFEST else MAX_FILE), 'Oversized archive member')
            total += member.size; require(total <= MAX_TOTAL + MAX_MANIFEST, 'Expanded archive size bound exceeded')
            stream = archive.extractfile(member); require(stream is not None, 'Missing archive member stream')
            if name == MANIFEST:
                manifest_raw = stream.read(MAX_MANIFEST + 1)
                require(len(manifest_raw) == member.size, 'Manifest archive length mismatch')
            else:
                h = hashlib.sha256(); size = 0
                for block in iter(lambda: stream.read(8 << 20), b''):
                    size += len(block); require(size <= member.size, 'Archive stream exceeds declared size'); h.update(block)
                require(size == member.size, 'Truncated archive member')
                regular[name] = {'sha256': h.hexdigest(), 'bytes': size}
    require(manifest_raw is not None, 'Archive lacks manifest')
    m = manifest_data(manifest_raw, manifest_sha256)
    require(regular == m['files'], 'Archive inventory/hash differs from trusted manifest')
    require(directories <= parent_names(regular), 'Extra archive directory')
    for name in seen:
        for parent in PurePosixPath(name).parents:
            if str(parent) != '.': require(str(parent) not in regular and str(parent) != MANIFEST, 'Archive file used as path parent')
    require(sha(path) == archive_sha, 'Archive changed during inspection')
    return {'schema': 1, 'kind': 'c2-alex-archive-inventory', 'archive_sha256': archive_sha,
            'manifest_sha256': manifest_sha256, 'files': len(regular), 'bytes': sum(row['bytes'] for row in regular.values()),
            'route_qualified': False, 'next': 'extract safely then verify_bundle using the same trusted manifest SHA256'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('verify', 'inspect-archive'))
    parser.add_argument('path', type=Path)
    parser.add_argument('--manifest-sha256', required=True, help='From an independently trusted receipt, never from this archive')
    args = parser.parse_args(argv)
    result = verify_bundle(args.path, args.manifest_sha256) if args.action == 'verify' else inspect_archive(args.path, args.manifest_sha256)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
