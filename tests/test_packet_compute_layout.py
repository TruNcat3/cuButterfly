import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "packet_compute_layout",
    ROOT / "scripts" / "summarize_packet_compute_layout.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_report_isolates_layout_gain():
    rows = []
    values = {
        "aggregate": 1.0,
        "per_packet_interleaved_rows": 1.2,
        "per_packet_warp_rows": 1.1,
        "wave_bitmap_interleaved_rows": 1.15,
        "wave_bitmap_warp_rows": 0.95,
    }
    for batch in (16, 32, 64):
        for mode, value in values.items():
            rows.append({"label": f"{mode}_b{batch}_t1", "kernel_ms": str(value)})
    summary = MODULE.summarize(rows)
    text = MODULE.make_markdown(summary)
    assert "warp-row per-packet=1.091x" in text
    assert "warp-row bitmap=1.211x" in text
    assert "best/aggregate throughput=1.053x" in text
