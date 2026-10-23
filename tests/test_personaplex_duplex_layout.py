"""Address and lease checks for the shared software-only duplex candidate."""

import pytest

from opentpu.llm.personaplex_duplex_layout import (
    ARCH_HASH, CAPACITY, PAGE_LIMIT, PACKET_PLAN, TARGET, TEMPORAL_KV_AT,
    TEMPORAL_KV_END, DuplexLayout, check_spans,
)


@pytest.fixture(scope="module")
def layout():
    return DuplexLayout.production()


def test_exact_profile_and_all_codec_spans_survive_unchanged(layout):
    assert TARGET == "f2-hbm:p4rl8a64c150"
    assert ARCH_HASH == "dd6b92a6"
    rvq = layout.input_rvq
    assert layout.codec_spans == rvq.prefix_spans + rvq.spans
    assert [span for span in layout.spans if span["name"] in
            {old["name"] for old in layout.codec_spans}] == sorted(
                layout.codec_spans, key=lambda span: span["base"])
    assert len([span for span in layout.codec_spans if span["kind"] == "program"]) == 60
    assert len(layout.program_slots) == 74
    assert len(set(layout.program_slots)) == 74
    assert all(0x04000000 <= slot < 0x05000000 for slot in layout.program_slots)
    assert layout.spans == check_spans(layout.spans)
    b = layout.budget()
    assert b["program_slot_count"] == 74
    assert b["program_reserved_bytes"] == 12_124_160
    assert b["program_remaining_bytes"] == 4_653_056
    assert max(span["base"] + span["bytes"] for span in layout.spans) <= CAPACITY


def test_lm_tables_have_one_zero_prefix_and_do_not_claim_depth_steps(layout):
    s = layout.embedding_spec
    assert (s.text_vocab, s.audio_vocab, s.audio_streams, s.voice_frames) == (
        32001, 2049, 16, 4)
    assert layout.embedding.table_bytes == 1_061_715_968
    by_name = {span["name"]: span for span in layout.spans}
    assert by_name["lm_text_embedding"]["bytes"] == (32001 + 1) * 4096 * 4
    assert all(by_name[f"lm_audio_embedding{i}"]["bytes"] ==
               (2049 + 1) * 4096 * 4 for i in range(16))
    assert by_name["voice_frames"]["base"] + by_name["voice_frames"]["bytes"] == 0x44498000
    assert by_name["step_embeddings"]["base"] == 0x44498000
    assert by_name["step_embeddings"]["base"] != by_name["lm_text_embedding"]["base"]
    assert layout.budget()["zero_prefix_rows_per_lm_table"] == 1


def test_state_and_head_reservations_are_disjoint(layout):
    by_name = {span["name"]: span for span in layout.spans}
    assert by_name["temporal_kv"]["base"] == TEMPORAL_KV_AT
    assert by_name["temporal_kv"]["bytes"] == TEMPORAL_KV_END - TEMPORAL_KV_AT
    assert layout.temporal.cap == 128
    assert layout.temporal.kv_bytes == 34_603_008
    assert by_name["text_head_codes"]["base"] == 0x539B8000
    assert by_name["text_head_scales"]["base"] + by_name["text_head_scales"]["bytes"] == 0x5B7D1400
    assert 0x02510000 <= by_name["text_logits"]["base"] < 0x03510000
    assert by_name["text_logits"]["bytes"] == 128_000
    assert by_name["temporal_cos"]["base"] == 0x5D831400
    assert by_name["temporal_sin"]["base"] == 0x5D839400
    assert by_name["temporal_final_norm"]["base"] == 0x5D841400
    assert by_name["temporal_zmask"]["bytes"] == 1_024
    for key in ("ring_tokens", "ring_provided", "incoming_tokens",
                "incoming_present", "target_tokens", "target_provided", "emitted",
                "frame_plan", "current_text_id",
                "temporal_frame", "temporal_page"):
        assert 0 <= layout.io[key] < 0x10000
    assert layout.io["input_queue"] == layout.io["incoming_tokens"]
    assert layout.io["output_queue"] == layout.io["emitted"]
    assert layout.io["text_token"] == layout.io["current_text_id"]
    for key in ("embedding_sum", "temporal_x", "temporal_normalized"):
        assert 0x02510000 <= layout.io[key] < 0x03510000
        assert by_name[key]["bytes"] == 16_384


def test_six_exclusive_packets_fit_only_post_codec_page_window(layout):
    assert layout.input_rvq.weight_end == 1_960_284_160
    assert tuple((p.start_layer, p.end_layer, p.format) for p in layout.packets) == PACKET_PLAN
    assert tuple(p.bytes for p in layout.packets) == (
        831_782_912, 884_473_856, 884_473_856, 884_473_856, 331_677_696, 207_945_728)
    assert {p.base for p in layout.packets} == {layout.input_rvq.weight_end}
    assert layout.temporal.page_bytes == max(p.bytes for p in layout.packets)
    assert layout.budget()["packet_slack_bytes"] == 108_032_000
    assert len([s for s in layout.spans if s["name"] == "temporal_weight_page"]) == 1
    assert all(p.base % 4096 == 0 and p.base + p.bytes <= PAGE_LIMIT
               for p in layout.packets)
    for i, packet in enumerate(layout.packets):
        assert layout.check_refill(i, packet.base, packet.bytes,
                                   halted=True, drained=True) == packet
        stride = packet.bytes // (packet.end_layer - packet.start_layer)
        for local in range(packet.end_layer - packet.start_layer):
            assert layout.check_page_block(i, local, packet.base + local * stride,
                                           stride, halted=True, drained=True) == packet


@pytest.mark.parametrize("halted,drained", [(False, False), (False, True), (True, False)])
def test_packet_write_waits_for_halt_and_drain(layout, halted, drained):
    packet = layout.packets[0]
    with pytest.raises(RuntimeError, match="halted and drained"):
        layout.check_refill(0, packet.base, packet.bytes, halted=halted, drained=drained)


def test_packet_write_rejects_codec_crossing_partial_and_old_packet(layout):
    packet = layout.packets[1]
    for address, size in ((packet.base - 4096, packet.bytes),
                          (packet.base, packet.bytes - 4096),
                          (packet.base, 1_318_649_856),
                          (PAGE_LIMIT, packet.bytes)):
        with pytest.raises(MemoryError):
            layout.check_refill(1, address, size, halted=True, drained=True)
    with pytest.raises(ValueError):
        layout.check_refill(6, packet.base, packet.bytes, halted=True, drained=True)
    with pytest.raises(MemoryError):
        layout.check_page_block(1, 0, packet.base + 128,
                                packet.bytes // 8, halted=True, drained=True)
    with pytest.raises(RuntimeError):
        layout.check_page_block(1, 0, packet.base, packet.bytes // 8,
                                halted=False, drained=True)


def test_span_checker_rejects_overlap_duplicate_alignment_and_capacity():
    a = dict(name="a", base=128, bytes=128, alignment=128)
    assert check_spans([a]) == [a]
    with pytest.raises(MemoryError, match="overlap"):
        check_spans([a, dict(name="b", base=192, bytes=64)])
    with pytest.raises(ValueError, match="duplicate"):
        check_spans([a, dict(name="a", base=256, bytes=128)])
    with pytest.raises(MemoryError, match="invalid"):
        check_spans([dict(name="b", base=129, bytes=128, alignment=128)])
    with pytest.raises(MemoryError, match="invalid"):
        check_spans([dict(name="b", base=CAPACITY - 128, bytes=256)])
