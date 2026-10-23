# Device depth input conditioning

`DepthInputEngine` implements the input-preparation portion of native
`LMModel.forward_depformer`: select a4096→1024 projection for the codebook,
then add the selected previous-token embedding. Codebook0 uses the text table;
codebooks1–15 use the previous-audio table at codebook−1. Both selections execute
on device in one compiled program. The six depth-transformer layers, their
streaming state and audio generation are outside this component.

## Execution and precision boundary

`step(hidden, codebook, previous_token)` accepts a diagnostic temporal output
vector and two integer IDs. It uploads only those values and returns nothing.
The projected and conditioned1024-wide vectors remain on the selected backend;
`read_projected()` and `read_output()` are explicit diagnostics. A future shared
allocator must connect the resident temporal output/token slots directly to
these input descriptors, replacing diagnostic uploads. No complete chained
runtime or strict FPGA-only speech endpoint is implemented here.

The device selects the INT8 projection bank using RLD and projects the temporal
output directly, without extra normalization. FP32 source embeddings retain
exact values. A ragged concatenation avoids padding15 audio tables to text
vocabulary size. Each table has one leading positive-zero row. With text row
countT=text_vocab+1 and audio row countA=audio_vocab+1, the selected row is:

```text
max(codebook-1,0)*A + min(codebook,1)*T + previous_token + 1
```

All row arithmetic executes on device. The total row count is bounded below2^24
for exact FP32 integer arithmetic; byte addresses are bounded to unsigned32.
The native−1 sentinel selects the appropriate zero row. Initial/padding IDs
32000 for text and2048 for audio remain valid. Shared-projection mode is covered
by synthetic regressions, while the pinned production checkpoint uses16 banks.

Weights are converted once during offline image preparation. Runtime does not
select or copy a source matrix/table row, quantize weights, compute a projection,
or add an embedding on host. Diagnostic reference math is explicitly separate.
Memory/program capacity precedes lazy payload reads. Directbackendboard is
rejected; a physical factory must qualify the measured profile and CAP_GEN before
upload. Errors latch until reset; output reads require a completed request.

## Production layout and tests

The recorded burst200 ISA profile compiles the production stage to208 instruction
words versus32,768 IMEM words. Required memory is324,923,392 bytes, including the
normal program area; image payload is324,759,552 bytes. Weights comprise:

| Payload | Bytes |
|---|---:|
| 16 INT8 projection matrices and scales | 67,698,688 |
| Text plus15 audio FP32 tables, including zero rows | 257,032,192 |
| Total weights | 324,730,880 |

The matrix data base is D-byte aligned, including small-vocabulary protocol
fixtures. These are measured EC2 compilation/layout results, not FPGA latency.
The pinned projections are MLX affine INT4/group64 tensors in
`depformer.safetensors`, dequantized to FP32 offline before INT8 image conversion.
Step embeddings come from the separate`embeddings.safetensors` file.

EC2 one-thread nice19 aggregate `60aa90d0-da02-44af-a031-4998ad2b4a74` passed
404 tests in22.17 seconds with0 physical transport opens. It includes48 new
component cases, seven new reader cases and six checker cases. Coverage includes
all16 nonsequential/repeated banks, shared projection, independent quantized
matrix references, exact FP32 embedding addition at the actual device projection,
−1 and padding IDs, poisoned host sources, transfer limits, bad input/weights,
capacity/lazy-read ordering, reset/failure handling and actual BoardBackend over
IsaTransport. The latter is CPU protocol simulation, not physical execution.
The first real-checkpoint attempt stopped in metadata preflight because the
reader and small fixture initially placed projections in the wrong file. The
corrected split-file reader, same-name decoy and affine INT4/config regressions
then passed89 affected tests in2.30 seconds, physical opens0, receipt
`d3fde3a2-1603-4221-97ed-1c60d96dce94`. No component simulation ran in the failed
attempt; its log/receipt is preserved. The earlier404 count predates this reader
correction; unaffected suites were not repeated.

The cached-checkpoint checker uses saved four-frame temporal/token outputs:

```sh
python tools/personaplex_depth_input_check.py \
  --model-dir /existing/pinned/model \
  --token-dir /existing/token-isa-001 \
  --output-dir /new/isolated/depth-input-check
```

It verifies baseline output hashes and completed/reset receipt, pinned checkpoint
revision and file metadata, then calls the installed nativeforward_depformer
with the original source projections/embeddings. A capture stops immediately
before the depth transformer, so no unimplemented depth computation is implied.
Sixteen nonsequential cases use actual saved temporal hidden states and bounded
supplied previous-token IDs; four extra cases test zero and padding IDs. It
reports same-input source projection/conditioning error separately from exact
embedding addition and reset checks. Outputs and receipts require a new folder;
source/baseline/checkpoint changes during the check fail validation. Revision
metadata plus size/mtime is not exhaustive checkpoint-payload hashing.

Corrected cached-checkpoint job `94f6f006-fb4a-45b4-946e-ea30c8e02a2c`
completed exit0 in6.71 seconds of process wall time (checker4.80 seconds,
image build4.52 seconds). All20 cases had bit-exact FP32 embedding addition at
the device's actual quantized projection, and the reset replay matched bits.
Aggregate projection relativeL2 versus original dequantized-source FP32 math
was0.014962568896110136 (cosine0.9998882705415946); conditioned-output relativeL2
was0.012506121368913792 (cosine0.9999219205573243). Maximum absolute error for
both was0.02702629566192627. These quantify additional stage error on identical
saved temporal inputs, not full-model error or accepted quality. No depth layers
or audio generation ran. The physical-open guard recorded0; baseline output
hashes and checkpoint/native-source metadata remained unchanged.

## Remaining gates

The later [reusable depth layer](personaplex_depth_layer.md) implements native
per-step weights, RMSNorm/SiLU, attention and persistent device KV. One real
checkpoint layer and reset were initially CPU/ISA-validated; the later six-layer
check now executes all layers with exact output/KV reset and reported source
error. Resident stage integration, stream/prompt/cache/delay scheduling, Mimi/RVQ and PCM
validation remain. Prior temporal32.9% error and181.39ms/frame
are still unaccepted; a correct conditioning stage does not establish quality.

The [unified residency plan](personaplex_residency_plan.md) separates persistent
state/tables from temporal and depth/codec weight overlays. Its unified offsets remain a design; standalone depth banks/programs now have
measured allocation and compilation, while relocation and integration are pending. The prior
three standalone images totaled5,270,028,288 bytes; adding this independent image
would total5,594,951,680 bytes. A unified allocator must share the depth embedding
tables and program/state areas rather than add independent images together.
Host copying of immutable prepacked packets is permitted transport, but its
latency and bytes must be measured. No realtime or complete codec-fit claim is
made. No new RTL/image requirement has been demonstrated; partition/relocation
must be tested before proposing an authorized image/build change.

No live-card access, image reload, synthesis, new paid resources or service restart
occurred. The UI owns its current hardware window; eventual physical validation
requires coordinated availability and an exact qualified image/profile, not an
assumption that the old burst200 image is still loaded.
