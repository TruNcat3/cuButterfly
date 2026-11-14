import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_manifest():
    return json.loads((ROOT / "experiments/workloads/hadacore_local_core.json").read_text())


def test_manifest_keeps_complete_abi_out_of_local_inventory():
    manifest = load_manifest()
    provider = manifest["provider"]
    assert manifest["status"] == "local-abi-with-target-calibration-gate"
    assert provider["abi"] == "packed-local-tile-v1"
    assert provider["local_tile_executable"] is True
    assert "paired same-protocol" in provider["selection_gate"]
    assert provider["requires_explicit_arithmetic_contract"] is True


def test_population_and_contract_are_explicit():
    manifest = load_manifest()
    assert len(manifest["cases"]) == 90
    assert {case["tile_length"] for case in manifest["cases"]} == {2**k for k in range(1, 16)}
    assert {case["batch"] for case in manifest["cases"]} == {1, 2, 4, 16, 64, 256}
    assert len(manifest["quick_cases"]) == 9
    assert {case["tile_length"] for case in manifest["quick_cases"]} == {256, 4096, 32768}
    assert {case["batch"] for case in manifest["quick_cases"]} == {1, 16, 256}
    contract = manifest["semantic_contract"]
    assert contract["operator"] == "fwht"
    assert contract["storage_precision"] == "fp16"
    assert contract["accumulation_precision"] == "fp16-mma"
    assert contract["input_order"] == contract["output_order"] == "natural"


def test_unknown_compiler_resources_cannot_be_presented_as_measured():
    resources = load_manifest()["resource_contract"]
    assert resources["registers_per_thread"].startswith("cudaFuncGetAttributes")
    assert resources["workspace_bytes"] == 0
    acceptance = load_manifest()["acceptance"]
    assert acceptance["requires_gpu_correctness"]
    assert acceptance["requires_resource_report"]
    assert acceptance["requires_fixed_core_organization_comparison"]


def test_core_strategy_population_separates_controls_and_uncompiled_candidates():
    manifest = load_manifest()
    strategies = {item["id"]: item for item in manifest["core_strategies"]}
    assert {"upstream-mma-fp16", "native-fp16-stage-rounded-radix2",
            "native-fp16-stage-rounded-radix4", "fp16-register-half2",
            "fp16-mma-tile", "fp32-register-shuffle"} <= set(strategies)
    assert strategies["upstream-mma-fp16"]["requires_tile_abi"]
    assert strategies["fp16-mma-tile"]["same_contract_as_native_fp16_stage_rounding"] is False
    assert strategies["native-fp16-stage-rounded-radix2"]["same_contract_as_native_fp16_stage_rounding"]
    assert strategies["fp16-register-half2"]["status"] == "uncompiled"


def test_upstream_launch_population_is_exact_and_candidates_are_not_claimed_measured():
    launch = load_manifest()["resource_contract"]["launch_configs_from_upstream"]
    assert len(launch["log_n_1_to_8"]) == 2
    assert len(launch["log_n_9_to_15"]) == 7
    assert launch["log_n_9_to_15"][-1] == {
        "chunks_per_warp": 8, "warps_per_block": 16, "blocks_per_sm": 1}
    assert "matching validated paired calibration" in load_manifest()["selection_policy"]["never_default"]
