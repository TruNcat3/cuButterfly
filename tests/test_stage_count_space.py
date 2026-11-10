import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "summarize_stage_count_space.py"
SPEC = importlib.util.spec_from_file_location("stage_count_summary", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_stage_count_summary_preserves_m_dimension():
    rows = [
        {"label": "m2_p10-10_b4_t1", "kernel_ms": "1.0"},
        {"label": "m2_p10-10_b4_t2", "kernel_ms": "1.2"},
        {"label": "m4_p5-5-5-5_b4_t1", "kernel_ms": "2.0"},
        {"label": "m4_p5-5-5-5_b4_t2", "kernel_ms": "2.2"},
    ]
    records = MODULE.summarize(rows)
    assert records[0]["segment_count"] == 2
    assert records[1]["segment_count"] == 4
    assert records[0]["kernel_ms"] == 1.1
    assert "1.909x" in MODULE.render(records)


def test_stage_count_summary_rejects_inconsistent_label():
    try:
        MODULE.summarize([
            {"label": "m3_p10-10_b1_t1", "kernel_ms": "1.0"}
        ])
    except ValueError as error:
        assert "disagrees" in str(error)
    else:
        raise AssertionError("inconsistent stage-count label accepted")
