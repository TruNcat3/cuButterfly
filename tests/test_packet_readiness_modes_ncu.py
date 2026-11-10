import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "packet_readiness_modes_ncu",
    ROOT / "scripts" / "analyze_packet_readiness_modes_ncu.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_report_compares_bitmap_to_packet_and_aggregate():
    rows = {}
    for batch in MODULE.BATCHES:
        for mode, scale in (("aggregate", 1.0), ("per_packet", 1.2), ("wave_bitmap", 1.1)):
            rows[f"{mode}_b{batch}"] = {
                "time_us": str(100 * scale),
                "warp_instructions": str(200 * scale),
                "global_load_sectors": str(300 * scale),
                "shared_load_bank_conflicts": str(400 * scale),
                "cbu_warp_instructions": str(500 * scale),
            }
    text = MODULE.report(rows)
    assert "## Batch 64" in text
    assert "Bitmap/per-packet: time=0.917x" in text
    assert "bitmap/aggregate time=1.100x" in text
