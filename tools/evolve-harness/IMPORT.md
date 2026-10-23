# Source snapshot import

This directory imports the standalone OpenTPU bounded evolution harness from commit
`175c317e743e54bc82a33a8130ebe67c773394a5` into an ordinary branch based on OpenTPU
`f237dab2f196855a59a083b1454786df8f01291f`. The original repository and history remain
unchanged. No unrelated histories were merged.

All 44 original files are byte-for-byte copies under this directory.
`SOURCE_SNAPSHOT.json` records their Git blob IDs, SHA-256 digests and sizes.
Only this import note, the manifest and `IMPORT_VALIDATION.json` are additional
files. Source paths in the historical evidence are provenance, not dependencies
that this import executes.

The original README and `docs/STATUS.md` retain the status at source creation,
including statements that nothing had been published. This import publishes the
source for review; it does not change the recorded test or hardware outcomes.
Read `IMPORT_VALIDATION.json` for the checks performed during import.

## Invocation boundary

Run documented commands from `tools/evolve-harness/`. The package remains
self-contained, with only Python standard-library imports. It is not added to the
root OpenTPU package, its build flow, or any running service.

The 20-method fixture suite and synthetic resume/rollback demonstration have not
been executed for publication. The source retains its builder-only execution
restriction; no old denied access attempt was retried. Live use remains disabled
for every seed and still requires a qualified timing bridge, complete fixed-task
references, a current owner window and normal hardware locks. No deployment,
scheduler, model provider, hardware operation or paid resource was started.

## License review

The original standalone snapshot has no separate LICENSE file. The destination
repository carries Apache-2.0 in its root LICENSE. No downloaded or vendored
OpenEvolve implementation was found; the original documentation cites it as an
architecture reference. Historical OpenTPU source and runtime-output evidence is
retained with its original provenance. This import adds no third-party package,
model checkpoint, weight file, executable or archive.
