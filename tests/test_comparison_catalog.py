import json
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = ROOT / "scripts" / "validate_comparison_catalog.py"
SPEC = importlib.util.spec_from_file_location("cubutterfly_comparison_catalog_validator", VALIDATOR)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
validate = MODULE.validate


def test_external_comparison_catalog_is_contract_checked():
    document = json.loads((ROOT / "config" / "external_comparison_catalog.json").read_text())
    entries = validate(document)
    assert {entry["id"] for entry in entries} >= {
        "cufft", "vkfft", "dao-fht", "gpu-ntt", "cufftdx", "turbofft",
        "tcfft", "fftx-spectralpack", "fftw", "spiral", "flashfftconv",
        "tensorir", "tvm", "ansor", "roller", "welder", "timeloop", "quipsharp",
    }


def test_fft_precision_classes_do_not_promote_provider_support_to_measurements():
    document = json.loads((ROOT / "config" / "external_comparison_catalog.json").read_text())
    entries = {entry["id"]: entry for entry in validate(document)}
    for name in ("cufft", "vkfft", "cufftdx", "fftw", "tcfft", "flashfftconv"):
        entry = entries[name]
        support = entry["precision_support"]
        assert set(support["provider"]) == set(entry["compute_types"])
        for kind, measured in support["local_measured"].items():
            assert set(measured) <= set(support["provider"][kind])
        if entry["status"] == "source-only":
            assert all(not values for values in support["local_measured"].values())
            assert entry["runner"] is None
    assert entries["cufft"]["precision_support"]["local_measured"]["fft-l"] == []
    assert entries["vkfft"]["precision_support"]["local_measured"]["fft-l"] == []
    assert entries["fftw"]["compute_types"] == ["fft-h"]
    assert entries["fftw"]["platform"] == "cpu"
    assert entries["tcfft"]["compute_types"] == ["fft-l"]
    assert entries["tcfft"]["status"] == "prepared"
    assert entries["tcfft"]["local_adapter_status"] == "cpu-compiled-graph-unverified"
    assert entries["tcfft"]["precision_support"]["local_measured"]["fft-l"] == []
    assert entries["tcfft"]["upstream_commit"] == "3e87a7083577741815e21b2a3c2bed22520786ce"
    assert entries["tcfft"]["license"] == "GPL-3.0"
    assert entries["flashfftconv"]["comparison_class"] == "end-to-end-context"
    assert entries["cufftdx"]["comparison_class"] == "processing-unit"
    assert entries["turbofft"]["compute_types"] == ["fft-h"]
    assert entries["turbofft"]["compute_types_unknown"] == ["fft-l"]
    assert "not_verified_in_this_update" in document["precision_evidence_scope"]
    for name in ("cufft", "vkfft", "cufftdx"):
        assert entries[name]["precision_support"]["local_measurement_status"]["fft-l"] == "not_verified_in_this_update"
    for name in ("cufft", "vkfft"):
        assert entries[name]["precision_support"]["low_precision_adapter_status"] == "missing-exact-fp16-bf16-binding"
