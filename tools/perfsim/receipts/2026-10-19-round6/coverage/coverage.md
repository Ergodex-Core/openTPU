# Consolidated validation coverage

Canonical retained comparison entries, not unique workloads or independent statistical samples. Round3 fixed corpus supersedes overlapping round2. Latest reruns do not inflate cumulative count. Pytest counts are the current suite, never summed across rounds.

| Evidence family | Cumulative canonical | Exact timing | Bitwise | Float pass / fail / unrecorded | Fresh RTL this run |
|---|---:|---:|---:|---|---:|
| Fixed-memory kernel corpus | 112 | 112 | 112 | 0 / 0 / 112 | 0 |
| Native Ramulator original corpus | 21 | 21 | 21 | 9 / 0 / 12 | 0 |
| Model-sized operators | 30 | 30 | 30 | 28 / 2 / 0 | 0 |
| WAITC and stream-engine kernels | 73 | 73 | 73 | 71 / 0 / 2 | 0 |
| Cold hybrid launches | 6 | 6 | 6 | 6 / 0 / 0 | 0 |
| Persistent original topology/state probes | 18 | 18 | 18 | 15 / 0 / 3 | 0 |
| Persistent E2 hybrid topology | 12 | 12 | 12 | 12 / 0 / 0 | 9 |
| Persistent E4 hybrid topology | 6 | 6 | 6 | 6 / 0 / 0 | 6 |
| E2/E4 full-row epilogue kernels | 6 | 6 | 6 | 6 / 0 / 0 | 6 |

**284 canonical observations; 21 fresh RTL launches this run (12 new, 9 reruns); 302 current pytest cases pass.**

Frozen pytest replays are not new RTL executions. Historical pytest counts are not additive.

**Int4 remains a float-quality failure:** D32 5.77272%, D128 7.94663%. Independent packed arithmetic matches RTL bitwise. No gate was relaxed.

Initial timing failure, round2 three tiny-model timing failures and round5 superseded/zero-reference diagnostics remain in their original receipts; excluded from current timing-passing canonical corpus, never erased.

Full pretrained inference, pumped TMEM and physical Fmax/resources remain unvalidated. Historical DDR4 results are retained; this run adds HBM2 only.

- Fixed-memory kernel corpus: 112-case round3 corpus replaces overlapping round2 corpus; behavioral RAM, not HBM2
- Native Ramulator original corpus: HBM2 and historical DDR4/stress profiles
- Model-sized operators: Two original int4 float-quality failures retained; all packed arithmetic bitwise passes
- WAITC and stream-engine kernels: Final nondegenerate cases only; zero-reference DOT diagnostics remain retained
- Persistent original topology/state probes: 18 launches, four retained-controller sequences
- Persistent E2 hybrid topology: Latest six HBM2 reruns plus three packaging-smoke reruns; duplicates not added to cumulative count
- Persistent E4 hybrid topology: New HBM2 compute variant; does not validate pumped TMEM or a board shell
- E2/E4 full-row epilogue kernels: Three/four-row compiled GEMMs; independent images, persistent controller
