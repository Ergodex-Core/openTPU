# PersonaPlex on the common Inferlab FPGA profile

The deployment target for every model is the same image
`agfi-0001de1dd3df8e21e`, profile `f2-hbm:p4rl8a64c150`. PersonaPlex must switch
weights/programs under coordinated ownership; it must not reload a different
image. The current result qualifies the depth-chain software on the exact
pinned Gemma compiler/ISA simulator, not physical execution or full speech.

## Exact source and configuration

Read-only process/source inspection identified the deployed Inferlab release
`f8b9ae1` and its pinned runtime revision
`18fb978033799f0f21258f0e6e085c0360322d96`. Its source target is preserved byte
for byte in [shared_gemma150.toml](../tools/data/personaplex/shared_gemma150.toml),
SHA-256 `7acb13c3b7dc51dbeb72031df1baaf7b488717232a20a10a962e161585972b3c`.
The [contract](../tools/data/personaplex/shared_gemma150_contract.json) records
source hashes, target fields, required hardware checks and qualification state.

The named variant is essential: the file's base/default describes200MHz and
MCOLS4. The selected variant is150MHz, MCOLS8, PORTB4, ROWLANES=true and ALINE64.
It has platform5, S1/D128/LANES8, ACT_ROWS16/ACT_BLOCKS128, 65,536 TMEM words,
32,768 instruction words and4,294,967,296 mapped DRAM bytes. Its computed
architecture hash is `dd6b92a6`. T2 is available in this source profile but this
PersonaPlex path uses INT8. Inherited free-text descriptions are not the selected
variant's clock configuration.

This development branch's older compiler lacks the new configuration fields.
Changing only clock or MCOLS would therefore be an invalid qualification. The
isolated EC2 check copies the pinned runtime's whole `opentpu/` and `targets/`,
then overlays only `opentpu/llm/personaplex*.py` and the checker/tests. It preserves
pinned ISA, simulator, compiler, quantizer, kernels and host implementation,
verified by source hashes. No live runtime file is edited. Existing Python/native
Moshi dependencies and checkpoint cache are reused; no install or download.

The checker now takes explicit `--target`, records the resolved variant and raw
TOML hash separately, and attests its active ISA/numeric/compiler sources.
Run it from that isolated runtime overlay, not by pretending the old compiler
supports the new target:

```sh
python tools/personaplex_depth_chain_check.py \
  --model-dir /existing/pinned/model --token-dir /existing/token-isa-001 \
  --conditioning-dir /existing/depth-input-isa-002 \
  --stack-dir /existing/depth-stack-isa-001 \
  --output-dir /new/isolated/shared-profile-check \
  --frame-index 0 --target f2-hbm:p4rl8a64c150
```

## Fresh software evidence

The first compatibility run (`121097da-2c70-4a9b-b4de-7e14c77e77c8`) passed44
software cases. One legacy board-protocol fixture could not import
`opentpu.host.isaboard`, which is absent from the pinned runtime. The failure is
preserved; it is a test-support/host-path qualification gap, not a demonstrated
missing FPGA instruction. No model operation or hardware probe ran in that case.

The supported software-only chain and native-reference suite then passed14
cases in3.81s (`e3cbc934-986a-4b27-8037-600e730962a5`). Actual selected-profile
compilation passed with programs208/24,088/200 words. Named occupied DRAM is
1,615,491,144 bytes; the highest required aligned extent is4,273,471,488 bytes;
B has21,495,808 bytes spare. These match the depth-chain allocation on burst200,
without substituting its compiler or profile.

Real-checkpoint job `c37fc3a9-1fa9-4cf5-9069-15b1a3a0f9b1` completed in83.25s
process /80.65s checker CPU wall time. All16 generated IDs match original native
FP32 greedy generation and the ISA's own saved logits. Projected/conditioned/depth
vectors, logits, IDs and all six raw KV arrays have the same byte hashes as the
saved burst200 frame0 result. Full reset replay is bit-exact, including all16 IDs
and all six caches. The receipt records24 packed immutable-span hashes.

Depth relative L2 remains0.02337168964330375 and logit relative L2
0.020083488708821217 against the original dequantized-source FP32 reference.
Native16-step cache behavior, including the final slot0 exclusion, is preserved.
CPU wall time is not FPGA latency or audio frames/s. Quality remains unaccepted;
no PCM or complete model inference is produced. All tests use existing EC2,
one CPU affinity, one numerical/preparation worker, nice19, offline mode and a
physical-open guard recording0. No Mac model imports/math/tests occurred.

## Remaining physical and full-model gates

Actual card capabilities were not read here. Prior burst200 physical receipts
identify another image and do not qualify this one. Before any physical attempt,
the coordinated owner must verify exact live image/build/CFG_HASH/geometry,
CAP_GEN for RLD/ARGMAX, MX/STREAM and FP32 semantics, capacity and error/idle/
drained state. The depth engine deliberately refuses all physical backends.
A qualified shared runner/adapter and its transport test remain work; do not
silently use CPU/GPU serving fallback or reuse an old-image adapter.

The next full-model integration is temporal fixed KV and paged A weights,
resident temporal hidden/text connections, prompt/cache/delay scheduling, and
Mimi/RVQ/state/PCM execution. No new synthesis or missing ISA feature has been
established by the successful software test. Any demonstrated gap needs an
explicit design decision, not a per-model reload. Full speech stays disabled
until complete same-image physical correctness, quality and latency acceptance.

No card handle, lock change, image load, service restart, production pause or
new resource occurred. Serving interruption was zero seconds, preserving the
user's30-second maximum. This software evidence authorizes no serving readiness
change and makes no realtime claim.
