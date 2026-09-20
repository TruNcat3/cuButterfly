#!/usr/bin/env python3
"""CPU-only checks for the portable research campaign manifests."""

import copy
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "prepare_portable_campaign", ROOT / "scripts" / "prepare_portable_campaign.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def prepare(tmp_path):
    # Exercise the published semantic inputs. Historical result directories are
    # local research artifacts and are not prerequisites of a fresh checkout.
    campaign = ROOT / "config/research_campaign"
    return MODULE.prepare(
        core=campaign / "core.json",
        batch_precision=campaign / "batch_precision.json",
        crypto=campaign / "crypto.json",
        composed=campaign / "composed.json",
        stage_points=campaign / "stage_points.json",
        mapping_seeds=campaign / "mapping_seeds.json",
        output_dir=tmp_path / "campaign",
    )


def load(tmp_path, name):
    return json.loads((tmp_path / "campaign" / name).read_text())


def test_manifest_counts_and_hashes_are_reproducible(tmpdir):
    tmp_path = Path(str(tmpdir))
    manifest = prepare(tmp_path)
    assert manifest["counts"] == {
        "core": 74,
        "batch_precision": 284,
        "crypto": 208,
        "composed": 138,
        "stage_points": 28,
        "mapping_seeds": 79,
        "plan_workloads": 16,
        "plan_deferred": 11,
        "operator_extensions": 18,
    }
    body = copy.deepcopy(manifest)
    digest = body.pop("manifest_sha256")
    assert digest == MODULE._sha256_text(MODULE._canonical(body))
    for name, record in manifest["files"].items():
        text = (tmp_path / "campaign" / name).read_text()
        assert record["sha256"] == MODULE._sha256_text(text)
        assert record["bytes"] == len(text.encode())


def test_outputs_are_portable_and_do_not_contain_timing_history(tmpdir):
    tmp_path = Path(str(tmpdir))
    prepare(tmp_path)
    forbidden = ("/home/", "/root/", "kernel_ms", "h2d_ms", "d2h_ms", "measurements")
    for path in (tmp_path / "campaign").glob("*.json"):
        text = path.read_text()
        assert not any(token in text for token in forbidden), path


def test_stage_points_and_mapping_seeds_drop_provenance(tmpdir):
    tmp_path = Path(str(tmpdir))
    prepare(tmp_path)
    stage = load(tmp_path, "stage_points.json")
    assert len(stage["points"]) == 28
    assert all(point["mapping"]["backend"] == "shared-iterative" for point in stage["points"])
    assert all("mapping_json" not in point and "provenance" not in point for point in stage["points"])

    seeds = load(tmp_path, "mapping_seeds.json")
    assert len(seeds["seeds"]) == 79
    assert all(set(seed) == {"mapping", "semantics"} for seed in seeds["seeds"])
    assert all("provenance" not in seed for seed in seeds["seeds"])


def test_plan_points_use_only_complete_plan_bench_verification_contract(tmpdir):
    tmp_path = Path(str(tmpdir))
    prepare(tmp_path)
    document = load(tmp_path, "plan_workloads.json")
    assert len(document["workloads"]) == 16
    assert len(document["deferred"]) == 11
    unsupported = {"--element-stride", "--batch-stride", "--matrix"}
    for point in document["workloads"]:
        args = point["args"]
        assert point["operator"] == "fft"
        assert point["verification"]["required"] is True
        assert "--verify" in args and "--compare-cufft" in args
        assert not unsupported.intersection(args)
        assert "--inverse" not in args and "--in-place" not in args
        assert args[args.index("--length-mode") + 1] == "standard"
    assert all(point["status"] == "deferred-harness" for point in document["deferred"])
    assert {point["operator"] for point in document["deferred"]} >= {
        "fwht", "subset-zeta", "superset-zeta", "structured-2x2"
    }


def test_operator_extension_preserves_semantic_contracts_and_avoids_source_duplicates(tmpdir):
    tmp_path = Path(str(tmpdir))
    prepare(tmp_path)
    document = load(tmp_path, "operator_extensions.json")
    assert document["schema"] == "cubutterfly-install-search-v1"
    assert 12 <= len(document["workloads"]) <= 20
    assert {point["operator"] for point in document["workloads"]} == {
        "fwht", "subset-zeta", "superset-zeta", "structured-2x2"
    }
    assert any(point["direction"] == "inverse" for point in document["workloads"])
    assert any(point.get("element_stride") == 2 for point in document["workloads"])
    assert all("kernel_ms" not in point for point in document["workloads"])


def test_custom_source_and_target_arguments_are_supported(tmpdir):
    tmp_path = Path(str(tmpdir))
    arguments = ["--output-dir", str(tmp_path / "custom")]
    for name in ("core", "batch_precision", "crypto", "composed", "stage_points", "mapping_seeds"):
        source = tmp_path / f"custom-{name}.json"
        source.write_text((ROOT / "config/research_campaign" / f"{name}.json").read_text())
        arguments.extend([f"--{name.replace('_', '-')}", str(source)])
    assert MODULE.main(arguments) == 0
    manifest = json.loads((tmp_path / "custom/manifest.json").read_text())
    assert manifest["portable"] is True
    assert manifest["source_inputs"]["core"] == "custom-core.json"
