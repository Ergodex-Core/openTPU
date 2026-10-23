"""Exact shared Gemma-image compile contract; never opens or changes the card."""
import dataclasses

AGFI = "agfi-0001de1dd3df8e21e"
PROFILE = "f2-hbm:p4rl8a64c150"
RUNTIME_REVISION = "30100afb075374381601a6e01d9f914a58dbec91"
BUILD_ID = 0x417243f1
CFG_HASH = 0xdd6b92a6


def shared_tts_config():
    from ..isasim import Config
    from ..target import config, arch_hash
    required = {"PORTB", "ROWLANES", "ALINE", "T2"}
    if not required <= set(Config.__dataclass_fields__):
        raise RuntimeError("shared TTS target requires the wide-port runtime, revision 30100afb or compatible")
    cfg = config(PROFILE)
    expected = {"S": 1, "D": 128, "MCOLS": 8, "LANES": 8, "PORTB": 4,
                "ROWLANES": True, "ALINE": 64, "ACT_BLOCKS": 128,
                "ACT_ROWS": 16, "TMEM_WORDS": 65536, "IMEM_WORDS": 32768,
                "DRAM_BYTES": 4294967296, "SEQ_WC": True}
    if any(getattr(cfg, k) != v for k, v in expected.items()) or arch_hash(cfg) != CFG_HASH:
        raise RuntimeError("shared image profile/hash mismatch; do not substitute another image")
    return cfg


def profile_metadata():
    cfg = shared_tts_config()
    return {"agfi": AGFI, "profile": PROFILE, "build_id": f"{BUILD_ID:08x}",
            "cfg_hash": f"{CFG_HASH:08x}", "config": dataclasses.asdict(cfg),
            "evidence": "Pinned 30100afb target plus historical exact-image register receipt; no live card query",
            "physical_tts_accepted": False, "max_serving_interruption_seconds": 30}
