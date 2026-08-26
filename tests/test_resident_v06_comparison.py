import importlib.util
from pathlib import Path


SCRIPT = (Path(__file__).parents[1] / "scripts" /
          "summarize_resident_v06_comparison.py")
SPEC = importlib.util.spec_from_file_location("resident_v06", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def row(variant, bits, logn, batch, kernel_ms):
    return {"variant": variant, "word_bits": str(bits), "logN": str(logn),
            "batch": str(batch), "kernel_ms": str(kernel_ms)}


def test_summary_selects_best_versions_and_attributes_layers():
    rows = [
        row("v06_hybrid2d", 32, 20, 1, 1.2),
        row("v06_resident", 32, 20, 1, 1.0),
        row("v07_full", 32, 20, 1, 4.0),
        row("v07_resident_generic", 32, 20, 1, 2.0),
        row("v07_resident_core", 32, 20, 1, 0.8),
        row("v07_resident_core", 32, 20, 1, 1.0),
    ]
    records = MODULE.summarize(rows)
    assert len(records) == 1
    record = records[0]
    assert record["v06_selected"] == "v06_resident"
    assert record["mg_selected"] == "v07_resident_core"
    assert record["lowering_speedup"] == 2.0
    assert record["core_speedup"] == 2.0 / 0.9
    assert record["mg_best_vs_v06"] == 1.0 / 0.9
    assert record["v07_core_vs_v06_resident"] == 1.0 / 0.9
    rendered = MODULE.render(records)
    assert "physically equivalent" in rendered
    assert "1/1 points" in rendered


def test_summary_accepts_lengths_without_resident_v06_candidate():
    rows = [
        row("v06_hybrid2d", 64, 16, 4, 1.0),
        row("v07_full", 64, 16, 4, 3.0),
        row("v07_resident_generic", 64, 16, 4, 2.0),
        row("v07_resident_core", 64, 16, 4, 1.5),
    ]
    record = MODULE.summarize(rows)[0]
    assert record["v06_selected"] == "v06_hybrid2d"
    assert record["v06_resident_ms"] == ""
    assert record["v07_core_vs_v06_resident"] == ""
    assert record["mg_best_vs_v06"] == 2.0 / 3.0
