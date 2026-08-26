import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "packet_warp_rows_weights",
    ROOT / "scripts" / "summarize_packet_warp_rows_weights.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_summary_selects_rebalanced_service_ratio():
    rows = []
    for batch in (16, 32, 64):
        rows.extend([
            {"label": f"aggregate_b{batch}_t1", "kernel_ms": "1.0"},
            {"label": f"interleaved_b{batch}_p5_t1", "kernel_ms": "1.1"},
            {"label": f"warp_rows_b{batch}_p5_t1", "kernel_ms": "1.05"},
            {"label": f"warp_rows_b{batch}_p8_t1", "kernel_ms": "0.95"},
        ])
    summary = MODULE.summarize(rows)
    text = MODULE.make_markdown(summary)
    assert "Batch 16: best=8:12" in text
    assert "warp-row/interleaved throughput=1.158x" in text
    assert "warp-row/aggregate=1.053x" in text
