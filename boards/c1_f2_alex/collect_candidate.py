#!/usr/bin/env python3
"""Collect pending C1/Alex candidate evidence locally; no cloud or hardware operations.

CLI requires an independently trusted SHA256 for the collection specification.
Schema1 (qualifier) or schema2 (independent raw captures), kind
c1-alex-candidate-collection-inputs contains instance_id, roots (every
 evidence.ROLES key -> original absolute directory), receipt_sha256 (every
 evidence.RECEIPTS key -> independently recorded hash), and source_checkouts
(every evidence.SOURCE_ROLES key -> {path, revision}). Source revisions are full
approved Git commit IDs, not the checkout's moving HEAD. Git object bytes are
read without running checkout scripts, hooks, filters, or bundled Python/Tcl.
Collection and receipt-chain replay never grant physical qualification.

Output is fresh only: collection-inputs.json, bundle/, evidence.tar, and finally
collection.json. A failed attempt retains its partial files and failure.json;
there is no overwrite/resume switch. Existing run files are never modified.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import tarfile

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location('c1_alex_collect_evidence', HERE / 'candidate_evidence.py')
e = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(e)
SYNTH_REPORTS = ('objects.tcldata', 'summary.tcldata', 'timing_all_ids.xdc', 'timing.rpt',
                'drc.rpt', 'cdc.rpt', 'bus_skew.rpt', 'clock_interaction.rpt',
                'exception_coverage.rpt', 'ignored_exceptions.rpt', 'methodology.rpt',
                'waivers.rpt', 'utilization.rpt', 'REQUIRED_NEXT_STEPS.txt')


def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def encode(value): return (json.dumps(value, indent=2, sort_keys=True) + '\n').encode()
def record_bytes(raw): return {'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}
def write(path, raw):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())

def regular(path):
    info = path.stat()
    e.require(stat.S_ISREG(info.st_mode), 'Not a regular source file: ' + str(path))
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def original_file(root, name, *, installed=False):
    """Only the pinned installed HDK shell alias may be dereferenced."""
    path = root / e.relative(name)
    alias = root / 'hdk/common/shell_stable'
    for part in (path, *path.parents):
        if part.is_symlink():
            e.require(installed and part == alias, 'Symlink in source path: ' + str(part))
    resolved = path.resolve(strict=True)
    e.require(resolved.is_relative_to(root), 'Installed alias escapes HDK root')
    regular(resolved)
    return resolved


def file_names(root, *, prune=()):
    """Inspect one bounded required tree; never traverse links or caches elsewhere."""
    e.no_symlink(root); e.require(root.is_dir(), 'Missing source directory: ' + str(root))
    result = set()
    for directory, dirs, files in os.walk(root, followlinks=False):
        if Path(directory) == root:
            dirs[:] = [name for name in dirs if name not in prune]
        for name in dirs + files:
            path = Path(directory) / name
            mode = path.lstat().st_mode
            e.require(stat.S_ISREG(mode) or stat.S_ISDIR(mode), 'Symlink/special source object: ' + str(path))
            e.relative(path.relative_to(root).as_posix())
        result.update((Path(directory) / name).relative_to(root).as_posix() for name in files)
        e.require(len(result) <= e.MAX_FILES, 'Source tree exceeds file bound')
    return result


class GitSource:
    def __init__(self, path, revision):
        self.path, self.revision = Path(e.absolute(path)), revision
        e.no_symlink(self.path)
        e.require(isinstance(revision, str) and re.fullmatch('[0-9a-f]{40}', revision), 'Full approved Git revision required')
        resolved = self.command('rev-parse', '--verify', revision + '^{commit}').decode().strip()
        e.require(resolved == revision, 'Git source revision did not identify the approved commit')
        self.tree = {}
        raw = self.command('ls-tree', '-rz', '--full-tree', revision)
        for row in raw.split(b'\0'):
            if not row: continue
            fields, name = row.split(b'\t', 1)
            mode, kind, oid = fields.decode().split()
            key = name.decode()
            e.require(key not in self.tree, 'Duplicate Git tree path')
            self.tree[key] = (mode, kind, oid)
        e.require(len(self.tree) <= e.MAX_FILES, 'Git source tree exceeds file bound')

    def command(self, *args):
        # Parent Git overrides must not redirect immutable source selection or
        # inject per-command configuration. Keep only this collector's controls.
        env = {key: value for key, value in os.environ.items() if not key.startswith('GIT_')}
        env.update(GIT_CONFIG_NOSYSTEM='1', GIT_CONFIG_GLOBAL=os.devnull,
                   GIT_CONFIG_SYSTEM=os.devnull, GIT_NO_LAZY_FETCH='1', GIT_TERMINAL_PROMPT='0',
                   GIT_OPTIONAL_LOCKS='0', GIT_NO_REPLACE_OBJECTS='1')
        return subprocess.check_output(['git', '--no-replace-objects', '-c', 'core.hooksPath=/dev/null',
                                       '-C', str(self.path), *args], env=env, timeout=60)

    def read(self, name):
        e.relative(name)
        mode, kind, oid = self.tree.get(name, (None, None, None))
        e.require(mode in ('100644', '100755') and kind == 'blob' and re.fullmatch('[0-9a-f]{40}', oid or ''),
                  'Missing/nonregular approved Git source: ' + name)
        size = int(self.command('cat-file', '-s', oid))
        e.require(0 <= size <= e.MAX_MANIFEST, 'Oversized Git source blob')
        raw = self.command('cat-file', 'blob', oid)
        e.require(len(raw) == size, 'Git blob size changed')
        # cat-file uses immutable object IDs; additionally verify object contents.
        e.require(hashlib.sha1(b'blob ' + str(size).encode() + b'\0' + raw).hexdigest() == oid,
                  'Git blob contents differ from approved tree')
        return raw


class Plan:
    def __init__(self, spec):
        self.spec = spec
        self.roots = {role: Path(path) for role, path in spec['roots'].items()}
        self.files, self.source_bytes, self.tree_checks = {}, {}, []
        self.git = {}
        self.manifest = {'schema': spec['schema'], 'kind': 'c1-alex-portable-candidate-evidence', 'instance_id': spec['instance_id'],
                         'hardware': {}, 'roots': {}, 'source_snapshots': {}, 'receipts': {}, 'files': {}}
        for role, path in self.roots.items():
            prefix = 'runs/simulation/private-ip' if role == 'simulation_private' else 'runs/' + role
            self.manifest['roots'][role] = {'original_absolute_path': str(path), 'bundle_relative_path': prefix}
        for role, row in spec['source_checkouts'].items():
            self.manifest['source_snapshots'][role] = {'revision': row['revision'], 'bundle_relative_path': 'sources/' + role}

    def add(self, role, name, wanted=None):
        e.relative(name)
        path = original_file(self.roots[role], name, installed=role == 'installed')
        stamp = regular(path)
        e.require(stamp[2] <= e.MAX_FILE, 'Source file exceeds size bound')
        row = {'sha256': e.sha(path), 'bytes': stamp[2]}
        e.require(regular(path) == stamp, 'Source changed during inventory: ' + str(path))
        if wanted is not None:
            e.metadata(wanted); e.require(row == wanted, 'Receipt-bound source differs: ' + role + '/' + name)
        key = self.manifest['roots'][role]['bundle_relative_path'] + '/' + name
        previous = self.files.get(key)
        e.require(previous is None or previous == (role, name, path, stamp, row), 'Conflicting selected source binding')
        self.files[key] = (role, name, path, stamp, row)
        self.manifest['files'][key] = row
        return row

    def read(self, role, name, sha256=None):
        row = self.add(role, name)
        e.require(sha256 is None or row['sha256'] == e.digest(sha256), 'Externally trusted receipt/input mismatch: ' + role + '/' + name)
        e.require(row['bytes'] <= e.MAX_MANIFEST, 'JSON evidence exceeds bound')
        raw = original_file(self.roots[role], name, installed=role == 'installed').read_bytes()
        e.require(record_bytes(raw) == row, 'JSON source changed while reading')
        return e.decode(raw)

    def receipt(self, role):
        name = e.schema_receipts(self.spec['schema'])[role]; digest = self.spec['receipt_sha256'][role]
        result = self.read(role, name, digest)
        self.manifest['receipts'][role] = {'relative_path': self.manifest['roots'][role]['bundle_relative_path'] + '/' + name,
                                         'sha256': digest}
        return result

    def inventory(self, role, records):
        e.require(isinstance(records, dict) and records, 'Missing source inventory: ' + role)
        for name, row in records.items(): self.add(role, name, row)

    def exact_tree(self, role, name, expected, *, suffixes=None, prune=()):
        root = self.roots[role] / e.relative(name) if name else self.roots[role]
        def names():
            values = {name + '/' + v if name else v for v in file_names(root, prune=prune)}
            return {v for v in values if suffixes is None or Path(v).suffix.lower() in suffixes}
        actual = names()
        e.require(actual == set(expected), 'Original tree has missing/extra bound files: ' + role + '/' + name)
        self.tree_checks.append((names, actual))

    def source(self, role, revision, records):
        info = self.spec['source_checkouts'][role]
        e.require(info['revision'] == revision, 'Approved source revision differs from original receipt: ' + role)
        key = (info['path'], revision)
        if key not in self.git: self.git[key] = GitSource(*key)
        source = self.git[key]
        for name, wanted in records.items():
            raw = source.read(name); e.metadata(wanted)
            e.require(record_bytes(raw) == wanted, 'Approved Git source differs from recorded input: ' + role + '/' + name)
            dest = 'sources/' + role + '/' + e.relative(name)
            e.require(dest not in self.source_bytes or self.source_bytes[dest] == raw, 'Conflicting source snapshot')
            self.source_bytes[dest] = raw; self.manifest['files'][dest] = wanted
        return source

    def recheck(self):
        for role, name, path, stamp, row in self.files.values():
            current = original_file(self.roots[role], name, installed=role == 'installed')
            e.require(current == path and regular(path) == stamp and e.sha(path) == row['sha256']
                      and regular(path) == stamp, 'Original input changed during collection: ' + str(path))
        for read_names, before in self.tree_checks:
            e.require(read_names() == before, 'Original tree changed during collection')


def checked_spec(spec):
    e.require(isinstance(spec, dict) and set(spec) == {'schema', 'kind', 'instance_id', 'roots', 'receipt_sha256', 'source_checkouts'},
              'Invalid collection specification fields')
    e.require(type(spec['schema']) is int and spec['schema'] in (1, 2) and spec['kind'] == 'c1-alex-candidate-collection-inputs'
              and isinstance(spec['instance_id'], str) and spec['instance_id'] in e.BUILDER_INSTANCES,
              'Wrong collection schema/dedicated instance')
    e.require(set(spec['roots']) == set(e.schema_roles(spec['schema'])) and set(spec['receipt_sha256']) == set(e.schema_receipts(spec['schema']))
              and set(spec['source_checkouts']) == set(e.schema_sources(spec['schema'])), 'Incomplete collection roles/trusted receipt hashes')
    for role, path in spec['roots'].items():
        e.absolute(path); e.no_symlink(Path(path)); e.require(Path(path).is_dir(), 'Missing collection input root: ' + role)
    for value in spec['receipt_sha256'].values(): e.digest(value)
    for row in spec['source_checkouts'].values():
        e.require(set(row) == {'path', 'revision'} and isinstance(row['revision'], str)
                  and re.fullmatch('[0-9a-f]{40}', row['revision']), 'Malformed approved source revision')
        e.absolute(row['path']); e.no_symlink(Path(row['path']))
    e.require(spec['roots']['simulation_private'] == spec['roots']['simulation'] + '/private-ip', 'Wrong simulation private ancestry')


def make_plan(spec):
    p = Plan(spec); b, sim = e.local_module('build'), e.local_module('simulate')
    final = None
    if spec['schema'] == 1:
        final = p.receipt('qualify')
        e.require(final.get('status') == 'pending-review' and final.get('route_qualified') is False and final.get('returncode') == 0,
                  'Raw C1 collection requires the preserved pending-review receipt')
    ip, synth, simulation = (p.receipt(role) for role in ('ip', 'synth', 'simulation'))
    ipi = p.read('ip', 'inputs.json', e.digest(ip['inputs_sha256']))
    syni = p.read('synth', 'inputs.json', e.digest(synth['inputs_sha256']))
    si = p.read('simulation', 'inputs.json', e.digest(simulation['inputs_sha256']))
    p.manifest['hardware'] = {key: syni[key] for key in ('source_revision', 'build_id')}
    for role, value, source_role in (('ip', ipi, 'ip'), ('synth', syni, 'hardware')):
        p.inventory(role, value['prepared_files'])
        source = p.source(source_role, value['source_revision'], value['source_inputs'])
        manifest_name = 'boards/c1_f2_alex/support/manifest.json'
        raw = source.read(manifest_name)
        e.require(hashlib.sha256(raw).hexdigest() == value['support_manifest_sha256'], 'Approved frozen support manifest differs')
        support = e.decode(raw)
        records = {manifest_name: record_bytes(raw)}
        for name, digest in support['sources'].items():
            dest = 'boards/c1_f2_alex/support/' + e.relative(name)
            row = record_bytes(source.read(dest)); e.require(row['sha256'] == e.digest(digest), 'Frozen support source differs')
            records[dest] = row
        p.source(source_role, value['source_revision'], records)
    p.inventory('installed', syni['installed_inputs'])
    p.inventory('ip', ip['products'])
    for stage in ('ip', 'pcis', 'clocks'):
        for name in ('ip_done.txt', 'exit.json', 'command.json', 'console.log'): p.add('ip', stage + '/' + name)
        p.exact_tree('ip', stage + '/ip_project', {name for name in ip['products'] if name.startswith(stage + '/ip_project/')}, suffixes=b.PRODUCT_SUFFIXES)
    for tree in e.DEPENDENCY_TREES:
        names = {tree + '/' + name for name in file_names(p.roots['ip'] / tree)}
        e.require(names, 'Empty generated dependency tree')
        for name in names: p.add('ip', name)
        p.exact_tree('ip', tree, names)
    for name in ('ip-binding.json', 'synth/post_synth.dcp', 'synth/exit.json', 'synth/synth_done.txt', 'synth/command.json', 'synth/console.log'):
        p.add('synth', name)
    for name in SYNTH_REPORTS: p.add('synth', 'synth/synth_audit/' + name)
    if (p.roots['synth'] / 'synth/pre_audit.NOT_QUALIFIED.dcp').exists():
        p.add('synth', 'synth/pre_audit.NOT_QUALIFIED.dcp')
    for role in ('ip', 'synth'):
        if (p.roots[role] / 'vivado_version.txt').exists(): p.add(role, 'vivado_version.txt')
    for stage in (('link', 'implement', 'refine', 'qualify') if spec['schema'] == 1
                  else ('link', 'implement', 'refine')):
        receipt = final if stage == 'qualify' else p.receipt(stage)
        inputs = p.read(stage, 'stage-inputs.json', e.digest(receipt['inputs_sha256']))
        p.inventory(stage, receipt['outputs'])
        p.source(stage, inputs['executor_revision'], inputs['executor_sources'])
        p.exact_tree(stage, '', set(receipt['outputs']) | {'stage-inputs.json', 'stage-result.json'})
    if spec['schema'] == 2:
        for role in e.RAW_ROLES:
            receipt = p.receipt(role)
            p.inventory(role, receipt['files'])
            source_records = {'boards/c1_f2_alex/' + name: receipt['files'][name]
                              for name in e.RAW_HELPERS[role]}
            p.source(role, receipt['source_revision'], source_records)
            # These independent captures deliberately exclude live console and
            # tool backup files. Copy only the complete frozen receipt inventory,
            # then recheck every selected byte; never promote omitted logs to
            # observations made by the collector.
    p.inventory('simulation', simulation['evidence'])
    p.inventory('simulation_private', si['simulation_ip_files'])
    for name in ('inputs.json', 'ip-complete.json'): p.add('simulation_private', name)
    for stage in ('ip', 'pcis', 'clocks'):
        for name in ('build.tcl', 'ip_done.txt'): p.add('simulation_private', stage + '/' + name)
        p.exact_tree('simulation_private', stage + '/ip_project',
                     {name for name in si['simulation_ip_files'] if name.startswith(stage + '/ip_project/')})
    source_records = {'boards/c1_f2_alex/simulate.py': {'sha256': si['runner_sha256'], 'bytes': (HERE / 'simulate.py').stat().st_size},
                      'boards/c1_f2_alex/build.py': {'sha256': si['builder_sha256'], 'bytes': (HERE / 'build.py').stat().st_size}}
    source_records.update({path.relative_to(e.ROOT).as_posix(): {'sha256': si['sources'][name], 'bytes': path.stat().st_size}
                           for name, path in sim.SOURCES.items()})
    p.source('simulation', spec['source_checkouts']['simulation']['revision'], source_records)
    raw = encode(p.manifest); e.manifest_data(raw, hashlib.sha256(raw).hexdigest())
    return p


def copy_file(path, dest, expected):
    dest.parent.mkdir(parents=True, exist_ok=True)
    digest, size = hashlib.sha256(), 0
    with path.open('rb') as source, dest.open('xb') as target:
        for block in iter(lambda: source.read(8 << 20), b''):
            size += len(block); e.require(size <= expected['bytes'], 'Source grew during copy')
            digest.update(block); target.write(block)
        target.flush(); os.fsync(target.fileno())
    e.require({'sha256': digest.hexdigest(), 'bytes': size} == expected, 'Copied file differs from bound source')


def archive_bundle(bundle, destination, manifest_sha256):
    m = e.manifest_data((bundle / e.MANIFEST).read_bytes(), manifest_sha256)
    names = sorted(set(m['files']) | {e.MANIFEST})
    with destination.open('xb') as output, tarfile.open(fileobj=output, mode='w|', format=tarfile.PAX_FORMAT) as archive:
        for name in names:
            path = bundle / e.relative(name); e.no_symlink(path)
            row = record_bytes(path.read_bytes()) if name == e.MANIFEST else m['files'][name]
            info = tarfile.TarInfo(name); info.size = row['bytes']; info.mode = 0o644
            info.uid = info.gid = info.mtime = 0; info.uname = info.gname = ''
            with path.open('rb') as stream: archive.addfile(info, stream)
    return e.inspect_archive(destination, manifest_sha256)


def collect(spec, out, *, spec_raw=None):
    """Trusted caller supplies approved receipt hashes; never mutates originals."""
    checked_spec(spec)
    out = Path(os.path.abspath(out)); e.no_symlink(out)
    inputs = [Path(path) for path in spec['roots'].values()] + [Path(row['path']) for row in spec['source_checkouts'].values()]
    e.require(not out.exists() and all(not out.is_relative_to(path) and not path.is_relative_to(out) for path in inputs),
              'Output must be fresh and outside every input/source checkout')
    out.mkdir(parents=True, exist_ok=False)
    raw = encode(spec) if spec_raw is None else spec_raw
    e.require(e.decode(raw) == spec, 'Collection input bytes differ from specification')
    write(out / 'collection-inputs.json', raw)
    started = now()
    try:
        plan = make_plan(spec)
        total = sum(row['bytes'] for row in plan.manifest['files'].values())
        required = 2 * total + (len(plan.manifest['files']) + 1) * 2048 + (1 << 30)
        e.require(shutil.disk_usage(out).free >= required, f'Insufficient free disk for bundle + tar + reserve: need {required} bytes')
        print(f'Collecting {len(plan.manifest["files"])} bound files ({total} bytes)', flush=True)
        bundle = out / 'bundle'; bundle.mkdir()
        for name, (_, _, path, _, row) in sorted(plan.files.items()): copy_file(path, bundle / name, row)
        for name, value in sorted(plan.source_bytes.items()): write(bundle / name, value)
        plan.recheck()
        manifest_raw = encode(plan.manifest); manifest_sha = hashlib.sha256(manifest_raw).hexdigest()
        write(bundle / e.MANIFEST, manifest_raw)
        verified = e.verify_bundle(bundle, manifest_sha)
        print('Pending candidate receipt chain verified; creating deterministic archive', flush=True)
        archive = out / 'evidence.tar'
        archived = archive_bundle(bundle, archive, manifest_sha)
        plan.recheck()
        e.require(e.verify_bundle(bundle, manifest_sha) == verified, 'Bundle changed during archive creation')
        receipt = {'schema': spec['schema'], 'kind': 'c1-alex-candidate-evidence-collection', 'state': 'complete',
                   'instance_id': plan.manifest['instance_id'], 'started_utc': started, 'finished_utc': now(),
                   'inputs_sha256': hashlib.sha256(raw).hexdigest(), 'manifest_sha256': manifest_sha,
                   'collector_sha256': e.sha(Path(__file__)), 'verifier_sha256': e.sha(HERE / 'candidate_evidence.py'),
                   'archive': {'relative_path': 'evidence.tar', 'sha256': archived['archive_sha256'], 'bytes': archive.stat().st_size},
                   'bundle_relative_path': 'bundle', 'verified': verified, 'hardware_actions': False}
        write(out / 'collection.json', encode(receipt))
        return {**receipt, 'receipt_sha256': e.sha(out / 'collection.json')}
    except BaseException as error:
        write(out / 'failure.json', encode({'schema': 1, 'kind': 'c1-alex-candidate-evidence-collection-failure',
              'state': 'failed', 'started_utc': started, 'failed_utc': now(), 'error': f'{type(error).__name__}: {error}',
              'hardware_actions': False, 'instruction': 'Preserve this attempt; fix the cause and use a new output directory.'}))
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', required=True, type=Path)
    parser.add_argument('--spec-sha256', required=True, help='Independently trusted collection specification SHA256')
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args(argv)
    e.no_symlink(Path(os.path.abspath(args.spec)))
    e.require(args.spec.stat().st_size <= e.MAX_MANIFEST, 'Oversized collection specification')
    raw = args.spec.read_bytes()
    e.require(hashlib.sha256(raw).hexdigest() == e.digest(args.spec_sha256), 'Collection spec differs from independently trusted SHA256')
    result = collect(e.decode(raw), args.out, spec_raw=raw)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__': main()
