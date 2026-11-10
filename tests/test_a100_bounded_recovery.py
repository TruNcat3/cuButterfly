"""CPU-only recovery checks; original frozen runners remain untouched."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "results/a100_closure_20261102/mechanism/bounded_search_recovery.py"
SPEC = importlib.util.spec_from_file_location("bounded_recovery_test", PATH)
recovery = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(recovery)
SOURCE = ROOT / "results/a100_closure_20261102/tail/bounded/attempts/20261102_130412_145861/screen"


def test_normalization_changes_only_proven_empty_partition_defaults():
    raw = {"stage_partition": [], "factor_partition": [], "boundaries": [],
           "local_stage_partitions": [], "unknown": 0, "nested": {"stage_partition": []}}
    normalized, audit = recovery.normalize(raw)
    assert normalized == {"boundaries": [], "local_stage_partitions": [], "unknown": 0,
                          "nested": {"stage_partition": []}}
    assert {row["field"] for row in audit} == {"stage_partition", "factor_partition"}
    assert raw["stage_partition"] == []
    for value in ([0], [-1], "", None, False):
        assert recovery.normalize({"factor_partition": value}) == ({"factor_partition": value}, [])
    with pytest.raises(ValueError, match="legacy alias"):
        recovery.normalize({"stage_partition": [], "stages_per_decomposition": [10, 10]})


def test_real_failed_three_inventory_prepare_and_audit(tmp_path):
    if not SOURCE.exists():
        pytest.skip("the captured A10080 failed inventory is not present")
    rows = recovery.prepare(SOURCE, tmp_path)
    assert len(rows) == 3
    assert all(row["status"] == "prepared_for_screen" for row in rows)
    for row in rows:
        directory = tmp_path / row["id"]
        raw = {recovery.base.digest({k: v for k, v in mapping.items()
                                   if k not in recovery.base.controlled.MAPPING_METADATA_FIELDS}): mapping
               for line in (directory / "inventory.jsonl").read_text().splitlines()
               if (mapping := json.loads(line))}
        study = json.loads((directory / "screen_study.json").read_text())
        audit = json.loads((directory / "normalization_audit.json").read_text())
        variants = study["cases"][0]["variants"]
        assert len(variants) <= 32
        for side in ("conventional", "hybrid"):
            assert 0 < sum(v["id"].startswith(side + "-") for v in variants) <= 16
        for variant, item in zip(variants, audit["variants"]):
            key = variant["id"].split("-", 1)[1]
            original = next(mapping for digest, mapping in raw.items() if digest.startswith(key))
            assert recovery.normalize(original)[0] == variant["mapping"]
            assert item["original_mapping_sha256"] == recovery.base.digest(original)
        prepared = json.loads((directory / "screen_prepared.json").read_text())
        assert prepared["status"] == "prepared"
        assert all(v["status"] == "pending" for v in prepared["cases"][0]["variants"])
        assert all("--mapping-json" in v["command"] for v in prepared["cases"][0]["variants"])
    manifest = json.loads((tmp_path / "recovery.json").read_text())
    assert manifest["gpu_queried_by_recovery"] is False
    assert manifest["experiment_complete"] is False
