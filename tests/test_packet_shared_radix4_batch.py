import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "summarize_packet_shared_radix4_batch.py"
SPEC = importlib.util.spec_from_file_location("packet_batch", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_selects_packet_weights_per_batch():
    rows = [
        {"label": "v06_b1", "kernel_ms": "1.0"},
        {"label": "lane32_b1", "kernel_ms": "0.9"},
        {"label": "packet128_b1_w8_12", "kernel_ms": "0.8"},
        {"label": "packet128_b1_w9_11", "kernel_ms": "0.7"},
        {"label": "v06_b16", "kernel_ms": "4.0"},
        {"label": "lane32_b16", "kernel_ms": "3.5"},
        {"label": "packet128_b16_w8_12", "kernel_ms": "3.2"},
        {"label": "packet128_b16_w9_11", "kernel_ms": "3.0"},
    ]
    summary = MODULE.summarize(rows)
    assert summary[0]["packet_weights"] == "9:11"
    assert summary[0]["packet_over_v06"] == 1.0 / 0.7
    assert summary[0]["best_control"] == "lane32"
    assert summary[0]["packet_over_best_control"] == 0.9 / 0.7
    assert not summary[0]["weight_search_boundary"]
    markdown = MODULE.make_markdown(summary)
    assert "v0.6 at 2/2 points" in markdown
    assert "best legacy core at 2/2 points" in markdown


def test_reports_open_lower_weight_boundary():
    rows = [
        {"label": "v06_b32", "kernel_ms": "4.0"},
        {"label": "lane32_b32", "kernel_ms": "2.0"},
        {"label": "packet128_b32_w4_16", "kernel_ms": "2.2"},
        {"label": "packet128_b32_w5_15", "kernel_ms": "2.3"},
    ]
    summary = MODULE.summarize(rows)
    assert summary[0]["weight_search_boundary"]
    markdown = MODULE.make_markdown(summary)
    assert "lower search boundary at batch 32" in markdown
    assert "lane32 crossover remains batch-dependent" in markdown
