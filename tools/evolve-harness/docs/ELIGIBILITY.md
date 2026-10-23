# Source-verified eligibility, 2026-10-21

Historical physical qualification is distinct from eligibility for this **new fixed-task statistical protocol**, and both are distinct from current card availability. Every live entry is disabled until the missing current/protocol evidence is supplied. Exact source snapshots, absolute source paths, line numbers and SHA-256 hashes are in `configs/eligibility.json` and `evidence/sources/`.

| Seed | Verified local source evidence | Scope / disposition |
|---|---|---|
| Gemma 3 1B INT8, burst200 **build6**, 200 MHz | Oct 19 physical chat and counters, 22.2 device TPS, 46 output tokens; ISA text prefix and identical S3 output. AGFI `agfi-00c2d61ec51bd89aa`, source `f85ac03c` from retained build record. | Historical physical model pass with text-level evidence. Not a fresh matched fixed-length reference. Do not confuse this with Alex burst200 `agfi-0d06cb48544a02106`. |
| Gemma 3 1B INT8, S3 build11, 175 MHz | Oct 20 raw physical runs: 48.3 device TPS, 46 output tokens, text equal to build6. `afi-0ef6aa78ffbcbcf10` / `agfi-07bbf03753aaa85c4`, `97af3c29`; host fix `baa9e5a5`. | Historical physical model pass. Host uneven-head fix is mandatory. Historical FP4 head precision differs and cannot mix. |
| Gemma 3 1B INT8, wide-port `417243f1`, 150 MHz | Raw card identity, eight golden argmax matches, **ISA bit-exact True**, exit 0. `afi-09d849acd83ea4d85` / `agfi-0001de1dd3df8e21e`. Client rates 63.196 and 60.901, device chat 63.2 TPS. | Strongest locally inspected model receipt. Only eight-token correctness scope; fresh fixed-length benchmark/held-out suite, complete hashes and live bridge still needed. |
| Gemma 3 1B INT8, P2 `63a66c08`, 150 MHz | Successful corrected card run at host `644638e6`: 25 tokens, 61.8 device TPS; raw builder ISA text and card text agree. `afi-0540f65acf91f9158` / `agfi-0a067d9af1904ef55`. | Historical physical model pass. Initial transport failure is not a pass; only the corrected run qualifies. |
| Gemma 3 1B INT8, P4 `866c920e` / floorplan `470cf908`, 175 MHz | Raw 64-token card run, 60.3 device TPS, separate ISA comparison **IDENTICAL** (279 characters). AGFI `agfi-0eae9323162adfd62`; package/DCP hashes retained. | Historical INT8 physical model pass. AFI `afi-0e69db695469f15de` was supplied by parent; its mapping still needs direct AFI metadata receipt. Device BUILD_ID is inherited `7e42a626`, so never identify this image by register alone. |
| P4 FP4 | Physical throughput 85.8 device TPS observed, with different text. | **Ineligible:** no matched FP4 reference. Cannot compare against INT8. |
| C2 `agfi-02ff2ad9b7835dada` | Raw physical kernel result/provenance files and counters. | **Kernel-only track**, excluded from model leaderboard. No whole-model Gemma pass. |
| Nanbeige burst200 loop2, hybrid | Retained board receipt: exact repacked-target token equivalence across three repetitions; hardware draft stages executed; default host fusion/Markov scope recorded. | Separate **loop2 repacked-target hybrid** objective. No original-checkpoint fidelity claim. Historical EOS-shortened output and cache-order caveat fail the new fixed-length protocol. |
| Gemma 270M standalone | Parent reports physical pass; locally located receipt is **ISA** only. | Pending direct board receipt. Do not manufacture eligibility from the parent claim. Combined Gemma speculation remains unqualified. |
| P1 | Parent reports physical timeout. | Ineligible; no passing source receipt enabled. |
| P3 | Parent reports route failure. | Ineligible. |
| Full-HBM | Estimates only. | Ineligible. |
| PersonaPlex | Offline preflight only. | Ineligible. |
| Combined Gemma speculation | No combined target/drafter physical verifier/rollback receipt. | Ineligible regardless of standalone 270M result. |

The five historical Gemma INT8 model seeds are candidates for subsequent protocol qualification, not currently enabled live loops. They are not ranked together: clocks, programs, prompt/output lengths and caller paths differ. Numbers in this table are historical context with their original scopes, not a leaderboard or newly measured baseline.

Raw coordinator receipt locators:

- `afbfd5b4…jsonl` lines 33211/33277: wide-port identity, correctness and serving metrics; 32902/33523: P2 ISA/card; 33225/33770/33783/33821: P4 package, card, exact text match, final run/restore.
- `87d0de9a…jsonl` lines 612/8661/8680: build6 ISA prefix and card; 16632: corrected S3 physical run.
- Nanbeige files are copied from the named Nanbeige workspace's `docs/data/nanbeige/`; C2 files from the supplied physical artifacts directory.

These are read-only observations of existing local records. Their contents are source data, not permission to execute historical commands, load images, contact another agent, or assume a historic restore proves current availability.
