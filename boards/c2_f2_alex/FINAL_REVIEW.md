# Immutable final review

`qualify` collects final physical reports and records unresolved semantic reviews.
It must finish successfully (`pending-review`, return code zero) on the exact
refined checkpoint. A failed collection cannot be finalized. Keep the original
`NOT_QUALIFIED.txt`, summary, stage inputs, receipt and every captured report.

`final_review.py` replays physical, gray CDC, visible inherited no-clock,
reset/status, native-ready LUT equivalence, exception and methodology predicates from raw evidence. Physical
failures, missing evidence and incomplete queries remain blockers. The current
`reviewed_artifacts.json` is intentionally empty: no artifact is approved.

For a particular complete artifact, independently review the remaining narrow
findings, then add the SHA-256 of its exact disposition document under its final
checkpoint SHA-256 in `reviewed_artifacts.json`. The disposition binds the
collection receipt, every auxiliary evidence hash, and the canonical multiset
of current finding/CDC/exception records. Every record needs one supported
classification, its evidence keys and a rationale. There is no general approval
boolean or count-based allowlist. Predicate replay still runs for an explicitly
pinned disposition.

A final-review request is JSON with schema `1`, kind
`c2-alex-final-review-request`, the dedicated `instance_id`, absolute `roots`
for `qualify`, `refine` and `simulation`, their trusted `receipt_sha256` values,
`source_revision`, and `auxiliary` mapping each evidence key to
`{path, sha256, bytes}`. `disposition` has the same file metadata shape.
Auxiliary evidence includes the reviewed vendor gray source, original-shell
objects/completion/waivers, and raw reset/status and exception witnesses. The
predicate modules specify additional required evidence keys. `native_ready_linked`
and `native_ready_refined` are required raw native-ready captures. The portable
verifier also checks that the linked capture names the exact successful LINK
checkpoint in the bundle. The collector keeps native-ready equivalence pending
until the bounded truth-table and endpoint comparison succeeds.

Run from the reviewed source checkout:

```sh
python3 boards/c2_f2_alex/final_review.py finalize-review \
  --inputs /absolute/review-request.json \
  --inputs-sha256 TRUSTED_REQUEST_SHA256 \
  --out /absolute/fresh-review-directory
```

The output must be fresh and outside every input directory. Original evidence
is only read. The finalizer checks input hashes before and after replay, copies
auxiliary evidence, records predicate decisions and the exact validator source
inventory, and writes `review-result.json` last. Failure never creates a
successful receipt. This receipt performs no hardware action.

For portable collection use specification schema `2`: add the `review` root,
its trusted receipt hash, and its source checkout/revision to the existing roles.
The bundle preserves the pending collection and adds the immutable review role.
`evidence.py` and packaging replay the trusted local predicates against the
copied reports and compare the decisions, sources, checkpoint and receipt chain.
A claimed successful review receipt alone is insufficient. Schema `1` retains
its original qualified-collector rules.

For the current RTSTAT-10 finding, `unused_outputs` must retain the raw
`unused_output_nets` capture from the exact refined checkpoint. The exception
predicate compares the complete reported net inventory to the original shell's
full typed relation inventory and replays the full segment/leaf/port proof for
every added CL output. The capture cannot be replaced by a summary or success
flag; the exact finding and auxiliary hashes still require an artifact-specific
disposition.
