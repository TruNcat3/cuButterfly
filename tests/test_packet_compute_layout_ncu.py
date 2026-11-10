import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "packet_compute_layout_ncu",
    ROOT / "scripts" / "analyze_packet_compute_layout_ncu.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_report_attributes_warp_row_delta():
    rows = {}
    for batch in MODULE.BATCHES:
        for mode, scale in (("aggregate", 1.0), ("bitmap_interleaved", 1.2), ("bitmap_warp_rows", 0.9)):
            rows[f"{mode}_b{batch}"] = {
                "time_us": str(100 * scale),
                "warp_instructions": str(200 * scale),
                "global_load_sectors": str(300 * scale),
                "shared_load_bank_conflicts": str(400 * scale),
                "cbu_warp_instructions": str(500 * scale),
                "lsu_warp_instructions": str(600 * scale),
            }
    text = MODULE.report(rows)
    assert "## Batch 64" in text
    assert "Warp-row/interleaved: time=0.750x" in text
    assert "shared-load conflicts=0.750x" in text
