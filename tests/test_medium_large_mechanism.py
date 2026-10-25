import copy
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "paper/tools"))
import prepare_medium_large_mechanism_20261012 as medium
from test_principle_gap_campaign import accepted_journal


def config():
    return json.loads(medium.CONFIG.read_text())


@pytest.mark.parametrize("operator", ["fft", "fwht", "ntt"])
def test_single_axis_parallel_contract(operator):
    study = medium.make_study(config(), operator)
    case = study["cases"][0]
    assert case["workload"]["logN"] == 20 and case["workload"]["batch"] == 4
    mappings = [row["mapping"] for row in case["variants"]]
    assert [key for key in mappings[0] if mappings[0][key] != mappings[1][key]] == case["comparisons"][0]["changed_axes"]
    assert all(mapping["stage_partition"] == [10, 10] for mapping in mappings)
    resources = medium.resources(study, config())
    assert resources["ctas_per_group"] == 4096
    assert resources["threads_per_cta"] == 256
    assert resources["dynamic_shared_bytes"] == (8192 if operator == "fwht" else 16384)
    points = medium.module_points(study)
    assert [key for key in points[0] if points[0][key] != points[1][key]] == ["writer_aligned"]
    assert medium.base.audit_journal(accepted_journal(study), study)["status"] == "accepted"
    bad = accepted_journal(study)
    bad["cases"][0]["variants"][1]["preflight"]["resolved_mapping"]["compute_unit"] = "radix4"
    assert medium.base.audit_journal(bad, study)["status"] == "rejected"


def test_no_cartesian_or_weak_baseline_expansion():
    bad = config()
    bad["first_round"]["stage_partition"] = [1] * 20
    with pytest.raises(ValueError, match="finite"):
        medium.make_study(bad, "fft")
    value = config()
    assert value["candidate_matrix"]["status"] == "planned_not_queued"
    assert len(value["candidate_matrix"]["points"]) < 4 * 3
    assert all("launch_argv" not in task for task in value["separate_tasks"])
    assert value["separate_tasks"][1]["status"] == "blocked_public_interface"
    assert value["separate_tasks"][2]["status"] == "blocked_implementation"


def test_resource_bound_and_cpu_no_compile(tmp_path, monkeypatch):
    value = config()
    value["resource_limits"]["dynamic_shared_ceiling_bytes"] = 1
    with pytest.raises(ValueError, match="resource"):
        medium.resources(medium.make_study(value, "fft"), value)
    broker_dir = tmp_path / "broker"
    spec = dict(environment=dict(CUBUTTERFLY_PREPARE_BROKER=str(broker_dir)), jit_identity=dict(config={}))
    monkeypatch.setattr(medium.broker, "_document", lambda path: dict(id="fixture", config={}))
    def forbidden(*args, **kwargs):
        raise AssertionError("CPU-only non-compilation preparation cannot start any subprocess")
    monkeypatch.setattr(medium.base.controlled.subprocess, "run", forbidden)
    dependencies = medium.compile_dependencies(medium.module_points(medium.make_study(config(), "fft")), spec, tmp_path)
    assert len(dependencies) == 2
    assert all(row["status"] == "not_prepared" for row in dependencies)


def test_rejected_audit_exit_and_float_policy(tmp_path, monkeypatch):
    study = medium.make_study(config(), "fft")
    journal = accepted_journal(study)
    journal["cases"][0]["variants"][0]["trials"].pop()
    path = tmp_path / "journal.json"
    path.write_text(json.dumps(journal))
    captured = []
    monkeypatch.setattr(medium.base.controlled, "_atomic_write", lambda path, value: captured.append(copy.deepcopy(value)))
    assert medium.main(["--operator", "fft", "--audit-journal", str(path)]) == 1
    assert "unassessed" in captured[0]["numerical_gate"]
