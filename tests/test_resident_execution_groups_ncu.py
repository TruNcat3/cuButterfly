import importlib.util
from pathlib import Path


SCRIPT = (Path(__file__).parents[1] / "scripts" /
          "analyze_resident_execution_groups_ncu.py")
SPEC = importlib.util.spec_from_file_location("resident_groups_ncu", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_controlled_pair_attribution_and_equivalence():
    rows = []
    values = {
        "m4_g4_generic": (40, 400, 200, 800, 80, 20),
        "m4_g2_generic": (20, 200, 100, 400, 80, 20),
        "m4_g2_dataflow": (10, 100, 50, 200, 40, 40),
        "m2_g2_dataflow": (10, 100, 50, 200, 40, 40),
    }
    for batch in (1, 16):
        for suffix, (time, loads, stores, warp, registers, active) in values.items():
            rows.append({
                "label": f"b{batch}_{suffix}",
                "time_us": str(time),
                "global_load_sectors": str(loads),
                "global_store_sectors": str(stores),
                "warp_instructions": str(warp),
                "integer_thread_instructions": str(warp * 4),
                "active_warps_pct": str(active),
                "registers_per_thread": str(registers),
                "long_scoreboard_stall_pct": "20",
            })
    records = MODULE.analyze(rows)
    boundary = next(row for row in records
                    if row["batch"] == 1 and
                    row["comparison"] == "boundary-lowering")
    assert boundary["ratio_global_load_sectors"] == 2.0
    core = next(row for row in records
                if row["batch"] == 1 and
                row["comparison"] == "physical-core")
    assert core["base_registers_per_thread"] == 80
    assert core["candidate_registers_per_thread"] == 40
    rendered = MODULE.render(records)
    assert "maximum sector/instruction delta is 0.00%" in rendered
