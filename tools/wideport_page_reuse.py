"""Saved-event page epoch audit with independent expected memory bytes.

This does not predict adapter writes or DMA timing. Native request/data/ack
events drive a protocol ledger; captured read data is compared with original
DRAM bytes plus declared patches, never used to generate those expected bytes.
"""
from collections import deque
import hashlib
import json
from pathlib import Path
import numpy as np
from tools.wideport_buffer_epochs import EpochPageLedger
from tools.wideport_credit_budget import classify_byte
from tools.wideport_memory_feedback import read_trace
from tools.wideport_native_memory import logical_beat
from tools.wideport_memory_coupled import validate_memory_coupled


def validate_page_epochs(directory, *, native_events=None):
    root = Path(directory)
    manifest = json.loads((root / "feedback-inputs.json").read_text())
    patch = manifest["epoch2"]
    image = np.fromfile(root / "dram_0.bin", dtype=">u4").astype("<u4").view(np.uint8)[:1 << 20]
    if hashlib.sha256(image.tobytes()).hexdigest() != manifest["dram_sha256"]:
        raise ValueError("initial image SHA mismatch")
    old = image.tobytes()
    fresh = bytearray(old)
    patches = {}
    for kind in ("weight", "scale"):
        dst, src = patch[kind + "_page"], patch[kind + "_stage"]
        data = old[src:src + 64]
        if hashlib.sha256(data).hexdigest() != patch[kind + "_patch_sha256"]:
            raise ValueError("patch SHA mismatch")
        fresh[dst:dst + 64] = data
        patches[dst] = kind, data
    header, recorded = read_trace(root / "native-feedback.jsonl")
    native = recorded if native_events is None else native_events
    _, slices = read_trace(root / "tmfeedback.jsonl")
    starts = [e["cycle"] for e in slices if e["start"] & 2]
    if len(starts) != 3:
        raise ValueError("page epoch fixture requires three MM starts")
    prefix = validate_memory_coupled(root, mm_prefix=2)
    counts = prefix["predicted_initial_page_native_reads"]
    if counts["0"] != counts["1"]:
        raise ValueError("bounded page ledger requires equal independently predicted initial read counts")
    # Old APF counts come from generated adapter requests, not a fixed beat
    # count fitted to the trace. The write/new-cache schedule remains excluded.
    ledger = EpochPageLedger(counts["0"])
    reads = [deque() for _ in range(8)]
    writes = [deque() for _ in range(8)]
    data = [deque() for _ in range(8)]
    read_id = [0] * 8
    write_id = [0] * 8
    ack = [0] * 8
    overwrite = None
    ack_cycles = {}
    checked = changed = 0
    new_reads = {"weight": 0, "scale": 0}
    for event in native:
        cycle = event["cycle"]
        if cycle == starts[0]:
            ledger.read_start(0, 0)
        if cycle == starts[1]:
            ledger.read_start(1, 0)
        if cycle == starts[2]:
            ledger.read_start(0, 1)
        for channel, row in enumerate(event["channels"]):
            delta = row["wdone"] - ack[channel]
            if delta < 0:
                raise ValueError("write completion counter regressed")
            ack[channel] = row["wdone"]
            for _ in range(delta):
                if not writes[channel] or writes[channel][0]["payload"] is None:
                    raise ValueError("write ack without command and accepted data")
                entry = writes[channel].popleft()
                if entry["patch"]:
                    kind, expected = patches[entry["address"]]
                    if entry["payload"] != expected or entry["mask"] != (1 << 64) - 1:
                        raise ValueError("patch write payload/mask differs from independent bytes")
                    ledger.complete_write(0, 1, kind, entry["ticket"])
                    ack_cycles[kind] = cycle
            if row["rv"]:
                if not reads[channel]:
                    raise ValueError("native read response without command")
                address, ticket, classification, epoch = reads[channel].popleft()
                if classification is not None:
                    kind, page = classification
                    expected = (fresh if epoch else old)[address:address + 64]
                    observed = int(row["rdata"], 16).to_bytes(64, "little")
                    if observed != expected:
                        raise ValueError(f"stale/data epoch mismatch at cycle {cycle}, byte {address:#x}, epoch {epoch}")
                    checked += 1
                    changed += int(bool(epoch and address in patches))
                    if not epoch:
                        ledger.respond(page, 0, kind, ticket, response_epoch=0)
            if row["cv"] and row["cr"]:
                address = 64 * logical_beat(row["addr"], channel, chash=bool(header["chash"]))
                if row["we"]:
                    is_patch = address in patches
                    ticket = ("w", channel, write_id[channel])
                    write_id[channel] += 1
                    if is_patch:
                        if overwrite is None:
                            ledger.close_reads(0, 0)
                            ledger.begin_overwrite(0, 0, 1, {"weight": 1, "scale": 1})
                            overwrite = cycle
                        ledger.start_write(0, 1, patches[address][0], ticket)
                    writes[channel].append(dict(address=address, ticket=ticket, patch=is_patch,
                                                payload=None, mask=None))
                else:
                    key = classify_byte(address)
                    ticket = ("r", channel, read_id[channel])
                    read_id[channel] += 1
                    epoch = int(bool(key and key[1] == 0 and overwrite is not None))
                    if key is not None and not epoch:
                        ledger.issue(key[1], 0, key[0], ticket)
                    elif epoch and cycle < starts[2]:
                        raise ValueError("new page read before both patches acknowledged and MM started")
                    elif epoch:
                        new_reads[key[0]] += 1
                    reads[channel].append((address, ticket, key, epoch))
            if row["wv"] and row["wr"]:
                data[channel].append((int(row["wdata"], 16).to_bytes(64, "little"),
                                      int(row["wmask"], 16)))
            for entry in writes[channel]:
                if entry["payload"] is None and data[channel]:
                    entry["payload"], entry["mask"] = data[channel].popleft()
    ledger.close_reads(1, 0)
    if not ledger.page_free(1, 0) or ledger.snapshot(0)["ready_epoch"] != 1:
        raise ValueError("page epoch ledger did not drain or complete patches")
    if changed < 2 or set(ack_cycles) != {"weight", "scale"}:
        raise ValueError("did not observe both changed native lines after patch completion")
    return dict(read_payload_beats_checked=checked, changed_epoch_lines_checked=changed,
                overwrite_cycle=overwrite, patch_ack_cycles=ack_cycles, new_epoch_mm_start=starts[2],
                old_native_epoch_closed_before_overwrite=True,
                page0=ledger.snapshot(0), page1=ledger.snapshot(1),
                initial_read_prefix=prefix,
                new_epoch_read_ownership_closed=False,
                new_epoch_observed_native_reads=new_reads,
                new_epoch_payload_checked=True, new_epoch_cache_schedule_predicted=False,
                write_timing="observed requests/data/acks, no independent DMA/adapter write predictor",
                whole_program_predictor=False, calibrated=False, tok_s=None)
