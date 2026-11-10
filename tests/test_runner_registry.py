from pathlib import Path

from benchmark.runner_registry import normalize, resolve, validate_references


ROOT = Path(__file__).resolve().parents[1]


def test_legacy_binaries_and_runners_have_one_resolution_path():
    legacy = {"binaries": {"ntt": "build/cuntt_bench"}}
    assert normalize(legacy)["ntt"]["kind"] == "ntt"
    modern = {"runners": {"ntt": {"kind": "cuda-ntt", "executable": "build/cuntt_bench"}},
              "cases": [{"runner": "ntt"}]}
    assert validate_references(modern)["ntt"]["kind"] == "cuda-ntt"
    assert resolve(modern, "ntt", ROOT)["path"] == (ROOT / "build/cuntt_bench").resolve()


def test_a100_comprehensive_manifest_uses_runner_registry():
    import json

    document = json.loads((ROOT / "config" / "a100_comprehensive_suite.json").read_text())
    registry = validate_references(document)
    assert registry["butterfly"]["kind"] == "comprehensive-cuda"
    assert registry["ntt"]["executable"].endswith("cuntt_bench")
