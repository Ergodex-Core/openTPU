# Localhost release resident-greedy-v4

URL: http://127.0.0.1:8787/
Source: task-2/f2-console. No Git checkout, push, PR or Vercel deployment was performed by this task. The separate Inferlab deployment owner consumes the immutable snapshot.

## Exact limits and device

Combined rendered chat context increased512→768; output maximum128→256. At output256 the rendered prompt budget is512; at output32 it is736. Runtime counts template/history tokens exactly and rejects overflow before acquiring/using the card. No truncation. A257-output request returns HTTP400 and the256-token limit. HTML/JS/session/server/private-worker handshake agree.

Existing image agfi-0001de1dd3df8e21e, build0x417243f1,150MHz, pinned runtime18fb978033799f0f21258f0e6e085c0360322d96. Hardware contracts assert resident/device-input/generate/chaining/stateless-KV, and reject host service/embedding/PLE. Every observed token is device selected. Both normal locks remain mandatory. The image was not changed for the capacity update.

1024 was rejected by the unchanged strict runtime guard because its default768-entry sliding ring is stateful.768 retains stateless KV through the existing API. `cap768-hardware.jsonl` records short baseline token-ID equivalence,497 rendered prompt +256 generated output twice with identical token IDs, and a clean short-answer reset afterward. Both locks were released after qualification.

## Browser receipts

All requests below used real FPGA inference, except the deliberate overflow which was rejected before hardware work.

| Job | Check | Result |
|---|---|---|
|0bd25f40-d41b-4360-baf6-70ffac6ae4c9|Cold worker, cached artifacts; sky/32 prompt|25 tokens; browserTTFT10.46s,total10.89s,63.21 walltok/s|
|58dd1024-7dc8-43de-8d86-74e4fe28399e|Long prefill beyond512|544 prefill tokens (545 rendered including held final token),8 generated; browserTTFT15.86s,total16.00s|
|849e1e68-a95c-4ffa-b5f4-244102921d17|UI maximum256|256 generated,finish max_new; browserTTFT1.42s,total5.48s,62.95 walltok/s|
|be8a3686-fc57-45bf-bd99-360201127374|Cancel long prefill|Stopped after8 prefill tokens; terminal cancelled,zero generated; warm lease retained|
|6382c1e8-2fb3-48a7-971f-82f02891e4ef|Combined overflow|809 prompt +32 output rejected against768; no hardware event/upload|
|87874aa2-9006-4868-8fc2-6c49854cf8ec|Warm retry of identical sky/32 after cancel/error|25 tokens; browserTTFT0.67s,total1.20s,63.21 walltok/s|
|ced450bc-38a1-4e38-972e-15900b404939|Real Markdown output|101 tokens,63.22 walltok/s; two bullets and fenced Python code rendered; no unsafe DOM nodes|

`browser-v4-hardware-events.json` contains exact contracts, normal-lock/image identity, metrics and terminal events for only these fixed tests. Cold cached model initialization1.526s, packed cache hit0.250s, upload1.262s; first-process imports4.463s and compiler warmup1.396s remain visible. Warm requests upload zero bytes through the load path. The long-prefill test remains15.53s on FPGA; larger context does not make prefill faster. The optional prompt_runs route remains unqualified/unenabled.

Earlier identical cap512 sky/32 measurements: uncached browserTTFT31.25s, cached reload4.44s, warm0.59–0.71s; corresponding receipts remain separate. Reusing residency speeds setup/delivery, not device arithmetic.

## Automated and isolated browser tests

57 automated tests passed:43Node and14Python, with no physical inference in the suite. Backend/transport checks cover repeat requests, cancellation, external busy state, reconnect without replay, stale-child races, context/output limits, strict host-compute gates, CSRF/origin/Host protections and model gates. Markdown/state checks cover incomplete streamed prefixes, literal rawHTML, unsafe URLs, tables/lists/code, long output and cancelled partials.

Clearly labeled fixture browser checks verified rendered lists/tables/fences, inert script/image payloads and blocked javascript links; failed request then retry; cancellation of an incomplete fence; reconnect to the same stream; and server-restart404 recovery preserving the draft and partial output without replay. `browser-restart-recovery.json`, `browser-markdown-safety.json` and `browser-audio-gates.json` retain receipts. Five model slots are visible; Phonon ASR and Qwen TTS controls are disabled. No microphone/upload/audio backend is invoked.

The original user-reported failure had no retained detailed error receipt. A concurrent409 failure mode was reproduced and fixed: active work blocks Send with the draft preserved; stale missing requests now recover instead of remaining busy. This does not prove the precise cause of the original historical message.

## Ownership and handoff

User-requested localhost cleanup found active:null. The temporary competing owner was the deployed gateway PID3675016; it was never killed or cancelled here. Its lease released normally before final browser tests. All local test requests are terminal; normal warm lease expiry or the idle Release FPGA endpoint handles handoff. No unrelated request or process was stopped.

The production gateway/Vercel owner must apply the768/256 session/capability/validation limits and update its pinned worker with --resident --context768. This task only deploys localhost. Keep its existing authentication and access policy. Nanbeige, PersonaPlex, Phonon-2 and Qwen3-TTS remain unavailable until their dedicated full-FPGA paths pass qualification.
