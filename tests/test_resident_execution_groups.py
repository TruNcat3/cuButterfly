import importlib.util
from pathlib import Path


SCRIPT = (Path(__file__).parents[1] / "scripts" /
          "summarize_resident_execution_groups.py")
SPEC = importlib.util.spec_from_file_location("resident_groups", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_resident_group_summary_keeps_m_and_g_independent():
    rows = [
        {"label": "m4_g4_p5-5-5-5_e5-5-5-5_b1_t1", "kernel_ms": "4.0"},
        {"label": "m4_g2_p5-5-5-5_e10-10_b1_t1", "kernel_ms": "1.0"},
        {"label": "m4_g2_p5-5-5-5_e10-10_b1_t2", "kernel_ms": "1.2"},
        {"label": "m3_g3_p5-5-10_e5-5-10_b1_t1", "kernel_ms": "3.0"},
        {"label": "m3_g2_p5-5-10_e10-10_b1_t1", "kernel_ms": "1.1"},
    ]
    records = MODULE.summarize(rows)
    m4g2 = next(row for row in records
                if row["logical_subgraphs"] == 4 and
                row["execution_groups"] == 2)
    assert m4g2["materialized_boundaries"] == 1
    assert m4g2["kernel_ms"] == 1.1
    rendered = MODULE.render(records)
    assert "M is the logical" in rendered
    assert "3.636x" in rendered
    assert "Physical-Equivalence Check" in rendered
