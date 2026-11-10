import copy
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
from hardware_registry import canonical_semantics, promote, locked_registry


def candidate(**sample):
    mapping = dict(schema_version=1, kind="butterfly", backend="shared-iterative",
                   stage_partition=[3, 4, 5], boundaries=[{"layout": "direct-strided"}])
    return dict(name="test", status="measured", correct=True, median_kernel_ms=0.2,
                samples=[dict(operator="fft", precision="fp32", logN=12, batch=2,
                              placement="out-of-place", mapping_json=json.dumps(mapping), runtime_fingerprint="test-v1", **sample)])


def test_registry_is_incremental_and_timings_do_not_change_mapping_identity(tmp_path):
    path = tmp_path / "registry.json"
    gpu40 = dict(device="A100", compute_capability="8.0", global_memory_bytes=40 << 30)
    gpu80 = dict(gpu40, global_memory_bytes=80 << 30)
    first = candidate(kernel_ms=0.2)
    promote(path, gpu40, [first])
    second = copy.deepcopy(first)
    second["median_kernel_ms"] = 0.3
    second["samples"][0]["kernel_ms"] = 0.3
    promote(path, gpu40, [second, candidate(direction="inverse")])
    promote(path, gpu80, [first])
    targets = json.loads(path.read_text())["targets"]
    assert len(targets) == 2 and [len(t["records"]) for t in targets] == [2, 1]
    assert {r["kernel_ms"] for r in targets[0]["records"]} == {0.2, 0.3}
    assert targets[1]["records"][0]["mapping"]["stage_partition"] == [3, 4, 5]


def test_semantics_distinguish_ntt_inverse_precision_modulus_and_order():
    row = dict(word_bits="32", logN="12", batch=4, inverse="1", modulus=2013265921,
               input_order="appt-static", output_order="natural")
    key = canonical_semantics(row)
    assert key["operator"] == "ntt" and key["precision"] == "word32" and key["direction"] == "inverse"
    for field, value in (("word_bits", "64"), ("inverse", "0"), ("modulus", 998244353), ("output_order", "appt-static")):
        assert key != canonical_semantics(dict(row, **{field: value}))
    assert canonical_semantics(dict(logN=8, batch_stride=0)) == canonical_semantics(dict(logN=8, batch_stride=256))


def test_incomplete_and_external_records_cannot_be_automatically_promoted(tmp_path):
    path = tmp_path / "registry.json"
    profile = dict(device="A100", compute_capability="8.0", global_memory_bytes=40 << 30)
    legacy = candidate()
    del legacy["samples"][0]["mapping_json"]
    external = candidate(backend="cufft")
    failed = dict(candidate(), correct=False)
    promote(path, profile, [legacy, external, failed])
    rows = json.loads(path.read_text())["targets"][0]["records"]
    assert len(rows) == 1 and rows[0]["status"] == "needs-revalidation"
    saved = path.read_bytes()
    try:
        with locked_registry(path) as data:
            data["targets"].clear()
            raise RuntimeError("interrupted calibration")
    except RuntimeError:
        pass
    assert path.read_bytes() == saved
