import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "summarize_three_level_partitions.py"
SPEC = importlib.util.spec_from_file_location("three_level_summary", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_summary_ranks_each_batch_independently():
    rows = [
        {"label": "p6-7-7_b1_t1", "kernel_ms": "2.0", "correct": "1"},
        {"label": "p6-7-7_b1_t2", "kernel_ms": "2.2", "correct": "1"},
        {"label": "p7-7-6_b1_t1", "kernel_ms": "1.0", "correct": "1"},
        {"label": "p7-7-6_b1_t2", "kernel_ms": "1.2", "correct": "1"},
    ]
    records = MODULE.summarize(rows)
    assert records[0]["partition"] == "7+7+6"
    assert records[0]["kernel_ms"] == 1.1
    assert records[0]["correct"] == 1
    assert "1.909x" in MODULE.render(records)


def test_summary_rejects_invalid_label():
    try:
        MODULE.summarize([
            {"label": "bad", "kernel_ms": "1", "correct": "1"}
        ])
    except ValueError as error:
        assert "invalid three-level label" in str(error)
    else:
        raise AssertionError("invalid label accepted")
