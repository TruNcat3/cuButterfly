import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "packet_polling_backoff",
    ROOT / "scripts" / "summarize_packet_polling_backoff.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_polling_summary_selects_median_candidate():
    rows = []
    for trial, value in enumerate((2.0, 3.0, 2.5), 1):
        rows.append({"label": f"aggregate_b16_t{trial}", "kernel_ms": str(value)})
    for ready_window, values in {2: (2.6, 2.4, 2.5), 8: (2.2, 2.1, 2.3)}.items():
        for trial, value in enumerate(values, 1):
            rows.append({
                "label": f"online_b16_rw{ready_window}_t{trial}",
                "kernel_ms": str(value),
            })
    summary = MODULE.summarize(rows)
    assert summary[0]["aggregate_ms"] == 2.5
    assert summary[0]["poll_sleep_cycles"] == 64
    assert summary[1]["online_ms"] == 2.2
    assert "ready_window=8" in MODULE.make_markdown(summary)
