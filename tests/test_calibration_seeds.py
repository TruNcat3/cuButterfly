import json
import pathlib
import sys

import pytest


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import calibration_seeds
from hardware_registry import SCHEMA as REGISTRY_SCHEMA, canonical_semantics, promote


def _semantics(*, operator="fft", precision="fp32", log_n=18,
               placement="out-of-place", batch=1):
    return canonical_semantics({
        "operator": operator,
        "precision": precision,
        "logN": log_n,
        "placement": placement,
        "batch": batch,
        "normalization": "none",
        "direction": "forward",
    })


def _mapping(partition=(9, 9), *, threads=256, backend="online-reorder"):
    return {
        "schema_version": 1,
        "kind": "butterfly",
        "backend": backend,
        "fft_core": "register-tile" if backend == "online-reorder" else "scalar",
        "stage_partition": list(partition),
        "boundaries": [{"layout": "writer-aligned", "residency": "global"}],
        "prefix_threads": threads,
        "prefix_ept": 16,
        "suffix_threads": threads,
        "suffix_ept": 16,
    }


def _record(key, mapping, latency, fingerprint, *, record_id, status="measured"):
    return {
        "record_id": record_id,
        "key": key,
        "mapping": mapping,
        "runtime_fingerprint": fingerprint,
        "status": status,
        "kernel_ms": latency,
    }


def _registry(path, targets):
    path.write_text(json.dumps({"schema": REGISTRY_SCHEMA, "targets": targets}) + "\n")


def _target(device, memory, records):
    return {
        "hardware": {
            "device": device,
            "compute_capability": "8.0",
            "global_memory_bytes": memory,
        },
        "records": records,
    }


def _candidate_mapping(candidate):
    return json.loads(candidate["point"]["mapping_json"])


def _assert_no_latency(value):
    if isinstance(value, dict):
        assert not {"kernel_ms", "median_kernel_ms", "latency_ms"}.intersection(value)
        for child in value.values():
            _assert_no_latency(child)
    elif isinstance(value, list):
        for child in value:
            _assert_no_latency(child)


@pytest.mark.parametrize("core", ["register-tile", "cufftdx-block"])
def test_imported_online_column_alias_is_canonical_and_deduplicated(core):
    old = dict(_mapping((11, 11)), fft_core=core, reorder_columns=4,
               prefix_threads=256, prefix_ept=32, suffix_threads=512,
               local_stage_partitions=[[6, 5], []], exchange_chunks=[0, 8])
    resolved = dict(old, reorder_columns=1)
    semantics = _semantics(log_n=22)
    snapshot = {"seeds": [
        {"semantics": semantics, "mapping": mapping,
         "provenance": {"kind": "explicit", "source": label}}
        for label, mapping in (("legacy", old), ("canonical", resolved))]}
    candidates = calibration_seeds.candidates_for_workload(snapshot, semantics)
    assert len(candidates) == 1
    assert _candidate_mapping(candidates[0]) == resolved
    assert len(candidates[0]["sources"]) == 2
    assert old["reorder_columns"] == 4  # Never mutate the frozen source.
    imported = calibration_seeds._seed(semantics, old, {"source": "legacy"})
    assert imported["mapping"] == resolved
    calibration_seeds.validate_mapping(imported["mapping"], resolved)
    with pytest.raises(ValueError, match="resolved mapping differs"):
        calibration_seeds.validate_mapping(imported["mapping"], dict(resolved, suffix_ept=8))


@pytest.mark.parametrize("backend,core", [("online-reorder", "scalar"),
                                          ("factor-streamed", "register-tile")])
def test_real_column_axes_are_not_canonicalized(backend, core):
    mapping = dict(_mapping(), backend=backend, fft_core=core, reorder_columns=4)
    assert calibration_seeds.canonical_mapping(mapping) == mapping


def test_registry_latency_selects_winners_only_within_original_context(tmpdir, monkeypatch):
    tmp_path = pathlib.Path(str(tmpdir))
    registry = tmp_path / "registry.json"
    monkeypatch.setenv("CUBUTTERFLY_REGISTRY", str(registry))
    key = _semantics(log_n=18, placement="out-of-place", batch=1)
    mapping_a = _mapping((8, 10), threads=128)
    mapping_b = _mapping((9, 9), threads=256)
    mapping_c = _mapping((10, 8), threads=512)
    mapping_external = dict(mapping_a, backend="cufft")
    _registry(registry, [
        _target("A100-40GB", 40 << 30, [
            _record(key, mapping_a, 2.0, "fp-a", record_id="a-slow"),
            _record(key, mapping_b, 1.0, "fp-a", record_id="a-fast"),
            # A different build/runtime context has its own winner.
            _record(key, mapping_a, 0.1, "fp-b", record_id="a-other-build"),
            _record(key, mapping_external, 0.001, "fp-a", record_id="cufft"),
        ]),
        _target("H100", 80 << 30, [
            # Hardware history is also a separate context, even when the
            # runtime fingerprint happens to have the same spelling.
            _record(key, mapping_c, 0.05, "fp-a", record_id="h-fast"),
        ]),
    ])

    snapshot = calibration_seeds.seed_snapshot(
        type("Args", (), {"mapping_seeds": [], "resume_search": False})(),
        tmp_path / "snapshot.json",
    )
    workload = {"operator": "fft", "precision": "fp32", "logN": 18,
                "placement": "in-place", "batch": 64,
                "normalization": "none"}
    candidates = calibration_seeds.candidates_for_workload(snapshot, workload)

    mappings = [_candidate_mapping(candidate) for candidate in candidates]
    assert {json.dumps(mapping, sort_keys=True) for mapping in mappings} == {
        json.dumps(mapping, sort_keys=True)
        for mapping in (mapping_a, mapping_b, mapping_c)
    }
    # The source's placement and batch are not portable execution semantics.
    assert all(candidate["point"]["placement"] == "in-place" for candidate in candidates)
    assert all(candidate["point"]["batch"] == 64 for candidate in candidates)
    assert all(candidate["point"]["operator"] == "fft" for candidate in candidates)
    assert all(candidate["point"]["precision"] == "fp32" for candidate in candidates)
    assert all(candidate["point"]["logN"] == 18 for candidate in candidates)
    _assert_no_latency(candidates)


def test_candidates_filter_operator_precision_and_log_n_but_transfer_other_semantics():
    shared = _mapping((8, 10))
    snapshot = {"seeds": [
        {"semantics": _semantics(operator="fft", precision="fp32", log_n=18,
                                  placement="out-of-place", batch=1),
         "mapping": shared,
         "provenance": {"kind": "registry-winner", "source": "old-a"}},
        {"semantics": _semantics(operator="ntt", precision="word32", log_n=18),
         "mapping": _mapping((6, 12), backend="shared-iterative"),
         "provenance": {"kind": "registry-winner", "source": "ntt"}},
        {"semantics": _semantics(operator="fft", precision="fp64", log_n=18),
         "mapping": _mapping((9, 9), threads=512),
         "provenance": {"kind": "registry-winner", "source": "fp64"}},
        {"semantics": _semantics(operator="fft", precision="fp32", log_n=19),
         "mapping": _mapping((10, 9), threads=512),
         "provenance": {"kind": "registry-winner", "source": "logn19"}},
    ]}

    target = {"operator": "fft", "precision": "fp32", "logN": 18,
              "placement": "in-place", "batch": 128,
              "normalization": "none"}
    candidates = calibration_seeds.candidates_for_workload(snapshot, target)

    assert len(candidates) == 1
    assert _candidate_mapping(candidates[0]) == shared
    assert candidates[0]["point"]["placement"] == "in-place"
    assert candidates[0]["point"]["batch"] == 128


def test_explicit_file_is_prioritized_and_duplicate_mappings_keep_all_sources(tmpdir, monkeypatch):
    tmp_path = pathlib.Path(str(tmpdir))
    registry = tmp_path / "registry.json"
    explicit = tmp_path / "explicit.json"
    monkeypatch.setenv("CUBUTTERFLY_REGISTRY", str(registry))
    key = _semantics(log_n=18)
    mapping = _mapping((8, 10), threads=256)
    explicit.write_text(json.dumps({
        "schema": calibration_seeds.SCHEMA,
        "seeds": [{"semantics": key, "mapping": mapping,
                   "provenance": {"label": "checked-explicit"}}],
    }) + "\n")
    _registry(registry, [_target("A100", 80 << 30, [
        _record(key, mapping, 0.01, "fp-a", record_id="registry-copy"),
    ])])

    snapshot = calibration_seeds.seed_snapshot(
        type("Args", (), {"mapping_seeds": [explicit], "resume_search": False})(),
        tmp_path / "snapshot.json",
    )
    candidates = calibration_seeds.candidates_for_workload(snapshot, {
        "operator": "fft", "precision": "fp32", "logN": 18,
        "placement": "in-place", "batch": 32,
    })

    assert len(candidates) == 1
    sources = candidates[0]["sources"]
    assert [source["kind"] for source in sources] == ["explicit", "registry-winner"]
    assert sources[0]["origin"] == {"label": "checked-explicit"}
    _assert_no_latency(candidates)


def test_resume_snapshot_is_frozen_against_registry_promotion(tmpdir, monkeypatch):
    tmp_path = pathlib.Path(str(tmpdir))
    registry = tmp_path / "registry.json"
    snapshot_path = tmp_path / "mapping_seed_snapshot.json"
    monkeypatch.setenv("CUBUTTERFLY_REGISTRY", str(registry))
    key = _semantics(log_n=18)
    old_mapping = _mapping((8, 10), threads=128)
    new_mapping = _mapping((9, 9), threads=512)
    _registry(registry, [_target("A100", 80 << 30, [
        _record(key, old_mapping, 1.0, "fp-a", record_id="old"),
    ])])
    args = type("Args", (), {"mapping_seeds": [], "resume_search": False})()
    initial = calibration_seeds.seed_snapshot(args, snapshot_path)

    promote(registry, {
        "device": "A100", "compute_capability": "8.0",
        "global_memory_bytes": 80 << 30,
    }, {"candidates": [{
        "name": "new-winner", "status": "measured", "correct": True,
        "median_kernel_ms": 0.01,
        "samples": [{
            **key, "mapping_json": json.dumps(new_mapping),
            "runtime_fingerprint": "fp-new", "kernel_ms": 0.01,
        }],
    }]})
    assert json.loads(registry.read_text())["targets"][0]["records"]

    resumed_args = type("Args", (), {
        "mapping_seeds": [], "resume_search": True,
    })()
    resumed = calibration_seeds.seed_snapshot(resumed_args, snapshot_path)
    assert resumed == initial
    candidates = calibration_seeds.candidates_for_workload(resumed, {
        "operator": "fft", "precision": "fp32", "logN": 18,
        "placement": "in-place", "batch": 16,
    })
    assert [_candidate_mapping(candidate) for candidate in candidates] == [old_mapping]


def test_resume_rejects_changed_explicit_seed_hash(tmpdir, monkeypatch):
    tmp_path = pathlib.Path(str(tmpdir))
    registry = tmp_path / "registry.json"
    explicit = tmp_path / "explicit.json"
    monkeypatch.setenv("CUBUTTERFLY_REGISTRY", str(registry))
    explicit.write_text(json.dumps({
        "schema": calibration_seeds.SCHEMA,
        "seeds": [{"semantics": _semantics(log_n=18),
                   "mapping": _mapping((8, 10)), "provenance": {}}],
    }) + "\n")
    snapshot_path = tmp_path / "snapshot.json"
    args = type("Args", (), {"mapping_seeds": [explicit], "resume_search": False})()
    calibration_seeds.seed_snapshot(args, snapshot_path)

    explicit.write_text(json.dumps({
        "schema": calibration_seeds.SCHEMA,
        "seeds": [{"semantics": _semantics(log_n=18),
                   "mapping": _mapping((9, 9)), "provenance": {}}],
    }) + "\n")
    with pytest.raises(ValueError, match="inputs changed"):
        calibration_seeds.seed_snapshot(type("Args", (), {
            "mapping_seeds": [explicit], "resume_search": True,
        })(), snapshot_path)


@pytest.mark.parametrize("bad_mapping", [
    {"schema_version": 2, "backend": "shared-iterative"},
    {"schema_version": 1, "backend": "cufft"},
    {"schema_version": 1, "backend": "shared-iterative", "kernel_ms": 0.1},
])
def test_public_seed_mapping_schema_and_measurements_are_rejected(tmpdir, monkeypatch, bad_mapping):
    tmp_path = pathlib.Path(str(tmpdir))
    explicit = tmp_path / "bad-seed.json"
    monkeypatch.setenv("CUBUTTERFLY_REGISTRY", str(tmp_path / "registry.json"))
    explicit.write_text(json.dumps({
        "schema": calibration_seeds.SCHEMA,
        "seeds": [{"semantics": _semantics(), "mapping": bad_mapping,
                   "provenance": {}}],
    }) + "\n")
    with pytest.raises(ValueError):
        calibration_seeds.seed_snapshot(type("Args", (), {
            "mapping_seeds": [explicit], "resume_search": False,
        })(), tmp_path / "snapshot.json")


def test_validate_mapping_allows_decoder_defaults_but_rejects_axis_changes():
    requested = _mapping((8, 10), threads=256)
    resolved = {**requested, "shared_bytes": 16384, "registers_per_thread": 32}
    calibration_seeds.validate_mapping(requested, resolved)

    changed_threads = {**resolved, "prefix_threads": 128}
    with pytest.raises(ValueError, match="differs"):
        calibration_seeds.validate_mapping(requested, changed_threads)

    changed_partition = {**resolved, "stage_partition": [9, 9]}
    with pytest.raises(ValueError, match="differs"):
        calibration_seeds.validate_mapping(requested, changed_partition)

    changed_schema = {**resolved, "schema_version": 2}
    with pytest.raises(ValueError, match="differs"):
        calibration_seeds.validate_mapping(requested, changed_schema)
