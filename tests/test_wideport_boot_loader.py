"""Small, standalone boot-loader FSM checks; no RTL or FPGA execution."""

import hashlib
import json
import struct

import pytest

from tools.wideport_boot_loader import BootConfig, BootLoader


def test_boot_wait_sidecar_addresses_and_backpressure():
    config = BootConfig(byte_addr=0x1000, instructions=5, seq_wc=True)
    assert config.rows == 2 and config.total_reads == 4
    assert [config.address_of(i) for i in range(4)] == [1024, 1056, 1088, 1088]
    loader = BootLoader(config)
    assert not loader.step(start=True, rd_idle_i=True).request
    assert not loader.step(rd_idle_i=False).request
    assert not loader.step(rd_idle_i=True).request
    first = loader.step(b_gnt=False)
    assert (first.request, first.word_addr, first.request_id) == (True, 1024, ("boot", 0))
    assert loader.step(b_gnt=True).request_id == ("boot", 0)
    assert loader.step(b_gnt=True).word_addr == 1056
    assert loader.step(b_gnt=True).word_addr == 1088
    assert loader.step(b_gnt=True).word_addr == 1088
    assert not loader.step().request
    for ordinal in range(4):
        got = loader.step(b_response_id=("boot", ordinal))
        assert got.instruction_write == (ordinal < 2)
        assert got.sidecar_write == (ordinal >= 2)
        if ordinal >= 2:
            assert (got.sidecar_row, got.sidecar_quarter) == (ordinal - 2, ordinal - 2)
    assert not loader.step().busy
    assert loader.finish()["reads"] == 4


def test_boot_sidecar_repeats_each_chunk_four_times():
    config = BootConfig(0, 17, True)
    assert config.rows == 5
    assert [config.address_of(i) for i in range(10)] == [
        0, 32, 64, 96, 128, 160, 160, 160, 160, 192]


def test_boot_rejects_orphan_and_out_of_order_response():
    config = BootConfig(0, 4, False)
    loader = BootLoader(config)
    with pytest.raises(ValueError, match="orphaned"):
        loader.step(b_response_id=("boot", 0))
    loader.step(start=True)
    loader.step(rd_idle_i=True)
    loader.step(b_gnt=True)
    with pytest.raises(ValueError, match="out of order"):
        loader.step(b_response_id=("boot", 1))


def test_fixture_config_uses_command_program_waits_and_image(tmp_path):
    config = BootConfig(128, 5, True)
    words = list(range(40))
    words[-7:] = [0] * 7
    program = b"".join(struct.pack("<I", word) for word in words)
    waits = [100 + i for i in range(5)]
    wait_bytes = b"".join(struct.pack("<Q", value) for value in waits)
    expected = program + bytes(-len(program) % 128) + wait_bytes
    image = bytearray(128 + len(expected))
    image[128:] = b"".join(expected[i:i + 4][::-1] for i in range(0, len(expected), 4))
    (tmp_path / "dram_0.bin").write_bytes(image)
    (tmp_path / "prog_0.hex").write_text("".join(f"{word:08x}\n" for word in words[:-7]))
    (tmp_path / "waits.hex").write_text("".join(f"{value:016x}\n" for value in waits))
    (tmp_path / "sim-command.json").write_text(json.dumps(["sim", "+boot", "+boot_addr=128", "+boot_n=5"]))
    (tmp_path / "feedback-inputs.json").write_text(json.dumps({
        "parameters": {"D": 128, "SEQ_WC": 1}, "instruction_count": 5,
        "program_sha256": hashlib.sha256(program).hexdigest(),
        "waits_sha256": hashlib.sha256(wait_bytes).hexdigest(),
    }))
    assert BootConfig.from_fixture(tmp_path) == config
    image[128] ^= 1
    (tmp_path / "dram_0.bin").write_bytes(image)
    with pytest.raises(ValueError, match="DRAM boot bytes"):
        BootConfig.from_fixture(tmp_path)
