"""Independent native-boundary predictions checked against saved RTL oracles."""
import json
from pathlib import Path

from tools.wideport_native_memory import NativeCommand, NativeInput, NativeMemory, NativeMemoryConfig


def read_trace(path):
    records = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if (not records or records[0].get("kind") != "config" or records[0].get("schema") != 1
            or records[-1].get("kind") != "end"):
        raise ValueError("missing native config/end or truncated trace")
    events = records[1:-1]
    if not events or records[-1]["cycles"] != len(events):
        raise ValueError("native trace length mismatch")
    return records[0], events


def native_config(header):
    return NativeMemoryConfig(channels=header["channels"], seed=header["seed"], stall=header["stall"],
                              lat=header["lat"], bandwidth=header["bw"], command_depth=header["cqd"],
                              write_depth=header["wqd"], write_done_lag=header["wdl"], dram=header["dram"],
                              deterministic=bool(header["deterministic"]), words=header["words"],
                              chash=bool(header["chash"]))


def require(got, expected, field, cycle):
    if got != expected:
        raise ValueError(f"cycle {cycle} {field}: observed {got}, predicted {expected}")


def validate_native_trace(path):
    """Use requests/reset only; ready/valid/done are comparison targets only.

    The upstream native adapter remains external in this boundary check. This
    does not supply a whole-program memory-arrival or calibrated TPS model.
    """
    header, events = read_trace(path)
    model = NativeMemory(native_config(header))
    counts = dict(cycles_checked=0, commands=0, write_commands=0, read_responses=0,
                  command_stall_cycles=0, write_stall_cycles=0)
    ids = [0] * header["channels"]
    for cycle, event in enumerate(events):
        require(event.get("kind"), "cycle", "record.kind", cycle)
        require(event["cycle"], cycle, "native.cycle", cycle)
        snapshot = model.snapshot()
        inputs = []
        for c, row in enumerate(event["channels"]):
            for key, predicted in (("cr", snapshot.command_ready[c]), ("wr", snapshot.write_ready[c]),
                                   ("rv", snapshot.read_valid[c]), ("wdone", snapshot.write_done[c])):
                require(row[key], int(predicted), f"native.channel{c}.{key}", cycle)
            command = NativeCommand(bool(row["we"]), row["addr"], ids[c]) if row["cv"] else None
            inputs.append(NativeInput(command, bool(row["wv"])))
            counts["read_responses"] += int(snapshot.read_valid[c])
            counts["command_stall_cycles"] += int(bool(row["cv"] and not snapshot.command_ready[c]))
            counts["write_stall_cycles"] += int(bool(row["wv"] and not snapshot.write_ready[c]))
        edge = model.step(inputs, reset=bool(event["rst"]))
        for c, command in enumerate(edge.accepted_commands):
            if command is not None:
                ids[c] += 1
                counts["commands"] += 1
                counts["write_commands"] += int(command.write)
        counts["cycles_checked"] += 1
    return {**counts, "writes_done": sum(model.snapshot().write_done), "mismatches": 0,
            "independent_fields": ["native command-ready", "native write-ready", "native read-valid", "native write-done"],
            "external_inputs": ["pre-edge native command/write-data requests", "native reset"],
            "whole_program_predictor": False, "calibrated": False, "tok_s": None}
