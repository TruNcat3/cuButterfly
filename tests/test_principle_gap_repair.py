import copy
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "paper/tools"))
import prepare_principle_gap_repair as repair
from test_principle_gap_campaign import accepted_journal


def test_repair_exactly_three_missing_pairs():
    config = json.loads(repair.CONFIG.read_text())
    study = repair.repair_study(config)
    assert len(study["cases"]) == 3
    assert sum(len(case["variants"]) for case in study["cases"]) == 6
    assert config["retained_v1_case"] not in [case["id"] for case in study["cases"]]
    bad = copy.deepcopy(config)
    bad["cases"].append(config["retained_v1_case"])
    with pytest.raises(ValueError, match="three incomplete pairs"):
        repair.repair_study(bad)


def test_float_sanity_is_not_user_acceptance():
    config = json.loads(repair.CONFIG.read_text())
    study = repair.repair_study(config)
    result = repair.audit(accepted_journal(study), study, config["numeric_policy"])
    assert result["status"] == "accepted"
    assert [case["numerical_gate"] for case in result["cases"]] == [
        "unassessed-user-acceptance", "unassessed-user-acceptance", "exact-full-batch"]


def test_real_v1_archive_remains_partial_and_ready_cpu_only(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("archive must not invoke GPU or subprocess")
    monkeypatch.setattr(repair.base.controlled.subprocess, "run", forbidden)
    evidence = repair.v1_evidence()
    assert evidence["status"] == "partial-not-whole-run-accepted"
    assert len(evidence["preparation_gate"]["events"]) == 3
    assert len(evidence["ready_now"]) == 2
    rows = evidence["cases"]
    assert [row["mapping_audit"]["status"] for row in rows] == ["rejected"] * 3 + ["accepted"]
    assert rows[-1]["variants"][0]["trial_us"] == pytest.approx([15.34, 15.36, 15.37])
    assert rows[-1]["variants"][1]["trial_us"] == pytest.approx([16.138, 16.108, 16.097])
    assert rows[-1]["baseline_over_treatment_median"] == pytest.approx(0.9535634467)
    assert "baseline_over_treatment_median" not in rows[2]
    assert rows[-1]["mapping_audit"]["actual_changed_axes"] == ["dataflow_layout"]


def test_rejected_audit_exits_nonzero(tmp_path, monkeypatch):
    study = repair.repair_study(json.loads(repair.CONFIG.read_text()))
    bad = accepted_journal(study)
    bad["cases"][0]["variants"][0]["trials"].pop()
    journal = tmp_path / "journal.json"
    journal.write_text(json.dumps(bad))
    monkeypatch.setattr(repair.base.controlled, "_atomic_write", lambda *args: None)
    assert repair.main(["--audit-journal", str(journal)]) == 1
