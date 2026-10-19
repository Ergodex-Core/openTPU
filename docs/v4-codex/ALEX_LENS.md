# Importing a measured C2 Alex session into Lens

`tools/append_c2_alex_lens.py` verifies a completed, restored physical session offline. It does not access AWS or FPGA hardware, execute evidence-bundle code, infer counter values, or create measurement profiles. With `--apply`, it appends the 30 existing verified hardware profiles and renders the combined viewer. Run without `--apply` first.

The fixed target is `../lens-v4/all-profiles.otpuprof` and its sibling HTML. The reviewed starting bundle contains 128 profiles and has SHA256 `3059bbbfeae93eb65595dfa8a81c3cbd67a1c7f7b465daf1c51eafe259466a58`. A different starting file fails validation. All 128 profile dictionaries and their order survive exactly; the final file contains 158 unique profiles. The file envelope and compression may change.

```sh
python tools/append_c2_alex_lens.py \
  --session /absolute/downloaded/session \
  --session-manifest /absolute/downloaded/session-files.json \
  --session-manifest-sha256 TRUSTED_SESSION_FILES_SHA256 \
  --evidence-dir /absolute/qualified/portable-evidence \
  --evidence-manifest-sha256 TRUSTED_EVIDENCE_MANIFEST_SHA256 \
  --image-record /absolute/available-image.json \
  --image-record-sha256 TRUSTED_AVAILABLE_IMAGE_SHA256 \
  --viewer opentpu/lens_app.html \
  --viewer-sha256 REVIEWED_VIEWER_SHA256
```

Use independently received hashes from the actual collection/publication records. The default command validates without writing. After reviewing the report, repeat the same command with `--apply`. It creates `all-profiles.pre-alex-hardware.otpuprof`, `all-profiles.pre-alex-hardware.html`, and `alex-hardware-profile-sources.json` beside the combined bundle. Existing backups or journal prevent a second application. A failed replacement restores both files from the backups. This is an offline, single-writer operation; a host crash between the two replacements can require restoring the retained pair manually.

The transfer manifest must sit outside the downloaded session directory. Its exact schema is:

```json
{
  "schema": 1,
  "kind": "c2-alex-physical-session-files",
  "instance_id": "i-0c78de2253f66519d",
  "source_revision": "40-character physical-session executor Git revision",
  "session": "/home/ubuntu/rachit/codex-c2/original-session-directory",
  "session_sha256": "SHA256 of session.json",
  "plan_sha256": "SHA256 of plan.json",
  "files": {
    "session.json": {"sha256": "file SHA256", "bytes": 123},
    "plan.json": {"sha256": "file SHA256", "bytes": 456}
  }
}
```

The `files` map must enumerate every actual regular file beneath the session, including all fixtures, benchmark results, profiles, provenance and raw logs. The abbreviated example is not a usable capture. Symlinks, hardlinks, special files, unsafe paths, altered files and extra files are rejected. The trusted capture hash binds transferred bytes; it does not grant route qualification.

The importer separately replays the portable route/simulation evidence and available deployment binding, validates the original image's restoration and candidate identity, and recomputes every fixture's canonical fp32/int48 results. It checks the five shapes `(31,1), (257,9), (31,54), (31,255), (3,9)`, both modes, three repeats, each raw result, all counters and exact exported profile contents. The physical executor revision may differ from the hardware source revision. Its source inventory must match a clean recorded Git commit available locally, and the local arithmetic/exporter bytes must match the measured version. No historical fixed CDC-count shortcut is used.

The merged viewer retains the earlier `mbc/mbr/mbs/mbo` blocked-cycle split. C2 aggregate profiles show a per-PC table for returned traffic, outstanding credits, native command backpressure, pending requests without a response, and full receive reservations. Window fractions describe each counter separately: they overlap each other and compute, and do not sum to a critical path. Operand-fill occupancy also overlaps successful transfer and compute. The native-interface lower bound is not measured HBM efficiency; absent unit utilization and instruction counts remain unmeasured. No timeline or full-token speedup is inferred from these counters.

Offline tests use clearly marked synthetic parser records only in pytest temporary directories. Storage tests use `kind=unit-test` display objects with a mocked semantic gate and temporary Lens paths; they never write the actual combined bundle.
