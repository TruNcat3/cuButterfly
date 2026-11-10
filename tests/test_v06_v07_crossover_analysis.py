import importlib.util
from pathlib import Path


SCRIPT = (Path(__file__).parents[1] / "scripts" /
          "analyze_v06_v07_crossover.py")
SPEC = importlib.util.spec_from_file_location("crossover", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def make(case_id, variant, value):
    return {"case_id": case_id, "variant": variant,
            "kernel_ms": str(value)}


def test_crossover_summary_attributes_wave_and_updated_control():
    rows = []
    for batch, h2d, td6, td12 in ((160, 3.0, 2.0, 2.1),
                                  (256, 3.0, 4.0, 4.1)):
        rows.extend([
            make("n12_u32", f"hybrid2d_b{batch}", h2d),
            make("n12_u32", f"hybrid_td6_b{batch}", td6),
            make("n12_u32", f"hybrid_td12_b{batch}", td12),
        ])
    rows.extend([
        make("n20_u32_b16", "hybrid2d", 1.1),
        make("n20_u32_b16", "resident_9_11", 1.0),
        make("n20_u32_b16", "packet_7_13", 1.2),
        make("n20_u32_b16", "lane32", 1.3),
    ])
    records = MODULE.summarize(rows)
    n160 = next(row for row in records if row["batch"] == 160)
    n256 = next(row for row in records if row["batch"] == 256)
    assert n160["cta_waves"] == 1
    assert n256["cta_waves"] == 2
    assert n256["last_wave_fill"] == 0.6
    n20 = next(row for row in records if row["case_id"] == "n20_u32_b16")
    assert n20["v07_vs_v06"] == 1.0 / 1.2
    rendered = MODULE.render(records)
    assert "candidate set strictly contains" in rendered
    assert "older 17:13" in rendered
