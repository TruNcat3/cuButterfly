import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "packet_streaming_v06",
    ROOT / "scripts" / "summarize_packet_streaming_v06_comparison.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_summary_uses_trial_medians():
    rows = []
    for implementation, values in {
        "v06": (4.0, 2.0, 3.0),
        "aggregate": (2.0, 1.0, 1.5),
        "online": (2.2, 1.2, 1.7),
    }.items():
        for trial, value in enumerate(values, 1):
            rows.append({
                "label": f"{implementation}_b16_t{trial}",
                "kernel_ms": str(value),
            })
    summary = MODULE.summarize(rows)
    assert summary[0]["v06_ms"] == 3.0
    assert summary[0]["aggregate_ms"] == 1.5
    assert summary[0]["online_ms"] == 1.7
    assert summary[0]["aggregate_speedup_v06"] == 2.0
    assert "v0.7 aggregate beats v0.6" in MODULE.make_markdown(summary)
