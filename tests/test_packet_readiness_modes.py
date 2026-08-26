import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "packet_readiness_modes",
    ROOT / "scripts" / "summarize_packet_readiness_modes.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_summary_selects_fastest_bitmap_window():
    rows = []
    for batch in (16, 32, 64):
        rows.extend([
            {"label": f"aggregate_b{batch}_t1", "kernel_ms": "1.0"},
            {"label": f"per_packet_b{batch}_rw2_t1", "kernel_ms": "1.1"},
            {"label": f"wave_bitmap_b{batch}_rw1_t1", "kernel_ms": "1.2"},
            {"label": f"wave_bitmap_b{batch}_rw4_t1", "kernel_ms": "1.05"},
        ])
    summary = MODULE.summarize(rows)
    text = MODULE.make_markdown(summary)
    assert "Batch 16: bitmap ready_window=4" in text
    assert "bitmap/per-packet throughput=1.048x" in text
    assert "Bitmap/per-packet geomean throughput: 1.048x" in text
