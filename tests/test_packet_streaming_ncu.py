import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "analyze_packet_streaming_ncu.py"
SPEC = importlib.util.spec_from_file_location("packet_streaming_ncu", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_packet_streaming_report_has_pairwise_ratios():
    rows = {}
    for label, time, instructions in (
        ("aggregate_b16", 100.0, 200.0),
        ("online_b16", 90.0, 220.0),
        ("aggregate_b32", 200.0, 400.0),
        ("online_b32", 220.0, 500.0),
        ("aggregate_b64", 400.0, 800.0),
        ("online_b64", 420.0, 840.0),
    ):
        rows[label] = {"time_us": str(time),
                       "warp_instructions": str(instructions),
                       "global_load_sectors": str(instructions / 2),
                       "shared_load_bank_conflicts": str(instructions / 4),
                       "cbu_warp_instructions": str(instructions / 8),
                       "long_scoreboard_stall_pct": str(instructions / 10)}
    text = MODULE.report(rows)
    assert "Batch 16: online/aggregate time=0.900x" in text
    assert "Batch 32: online/aggregate time=1.100x" in text
    assert "Batch 64: online/aggregate time=1.050x" in text
    assert "load sectors=1.100x" in text
    assert "shared-load conflicts=1.250x" in text
