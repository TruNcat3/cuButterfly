import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "packet_folded_barriers",
    ROOT / "scripts" / "summarize_packet_folded_barriers.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_summary_reports_barrier_folding_gain():
    rows = []
    values = {
        "aggregate": 1.0,
        "interleaved": 1.1,
        "interleaved_folded": 1.05,
        "warp_rows": 1.02,
        "warp_rows_folded": 0.95,
    }
    for batch in (16, 32, 64):
        for mode, value in values.items():
            rows.append({"label": f"{mode}_b{batch}_t1", "kernel_ms": str(value)})
    summary = MODULE.summarize(rows)
    text = MODULE.make_markdown(summary)
    assert "interleaved folding=1.048x" in text
    assert "warp-row folding=1.074x" in text
    assert "best/aggregate throughput=1.053x" in text
