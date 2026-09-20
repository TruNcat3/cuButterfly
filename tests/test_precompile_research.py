import json
import os
from pathlib import Path
import sys
from types import ModuleType

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import precompile_research


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


def _request(log_n):
    return {
        "backend": "shared-iterative",
        "operator": "fft",
        "precision": "fp32",
        "logN": log_n,
        "stage_partition": [log_n],
        "threads": 128,
    }


def _manifest(output, requests, complete=True):
    output = Path(output)
    modules = []
    for index, request in enumerate(requests):
        identifier = precompile_research._module_id(request)
        modules.append({
            "id": identifier,
            "request": request,
            "representative_point": {"logN": request["logN"], "batch": index + 1},
            "representative_cell": "cell-test",
            "cells": ["cell-test"],
            "policy": "research",
        })
    return {
        "schema": precompile_research.SCHEMA,
        "status": "ready" if complete else "incomplete",
        "complete": complete,
        "policy": "research",
        "profile": {"compute_capability": "8.0", "sm": 80},
        "module_count": len(modules),
        "modules_path": str(output / precompile_research.MODULES_NAME),
        "compilation_path": str(output / precompile_research.COMPILATION_NAME),
        "modules": modules,
    }


class _ImmediateFuture:
    def __init__(self, function, args):
        self._error = None
        try:
            self._result = function(*args)
        except BaseException as error:
            self._error = error

    def result(self):
        if self._error is not None:
            raise self._error
        return self._result

    def cancel(self):
        return False


def _inline_executor(monkeypatch, worker_counts):
    class ImmediateExecutor:
        def __init__(self, max_workers):
            worker_counts.append(max_workers)

        def submit(self, function, *args):
            return _ImmediateFuture(function, args)

        def shutdown(self, wait=True, cancel_futures=False):
            return None

    monkeypatch.setattr(precompile_research, "ProcessPoolExecutor", ImmediateExecutor)
    monkeypatch.setattr(precompile_research, "as_completed",
                        lambda futures: list(futures))


def test_compile_only_reads_manifest_hides_cuda_and_obeys_jobs(tmpdir, monkeypatch):
    output = Path(str(tmpdir)) / "compile"
    output.mkdir()
    _write_json(output / precompile_research.MODULES_NAME,
                _manifest(output, [_request(8)]))
    worker_counts = []
    _inline_executor(monkeypatch, worker_counts)
    seen = []

    def compile_one(task):
        seen.append(os.environ.get("CUDA_VISIBLE_DEVICES"))
        return {"id": task["id"], "path": "/cpu/mock.so", "status": "compiled",
                "elapsed_seconds": 0.0, "error": None}

    monkeypatch.setattr(precompile_research, "_compile_one", compile_one)
    monkeypatch.setattr(precompile_research, "runtime_candidates",
                        lambda *args: pytest.fail("compile mode enumerated candidates"))
    monkeypatch.setattr(precompile_research, "_load_workloads",
                        lambda *args: pytest.fail("compile mode read workloads"))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7")

    code = precompile_research.main([
        "--mode", "compile", "--output-dir", str(output), "--jobs", "2",
    ])

    assert code == 0
    assert worker_counts == [2]
    assert seen == [""]
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "7"
    journal = json.loads((output / precompile_research.COMPILATION_NAME).read_text())
    assert journal["complete"] is True
    assert journal["successful_modules"] == 1


def test_export_uses_policy_runtime_environment_and_deduplicates_modules(tmpdir, monkeypatch):
    root = Path(str(tmpdir))
    output = root / "export"
    build = root / "build"
    build.mkdir()
    (build / "cubutterfly_bench").write_bytes(b"mock benchmark")
    workloads = root / "workloads.json"
    _write_json(workloads, {
        "schema": "cubutterfly-install-search-v1",
        "workloads": [{"operator": "fft", "precision": "fp32", "logN": 8,
                       "batch": 1, "placement": "out-of-place", "normalization": "none"}],
    })
    profile = root / "profile.json"
    _write_json(profile, {"device": "Test GPU", "compute_capability": "8.0",
                          "global_memory_bytes": 4096})
    points = [
        {"operator": "fft", "precision": "fp32", "logN": 8, "batch": 1, "kind": "module-a"},
        {"operator": "fft", "precision": "fp32", "logN": 8, "batch": 2, "kind": "module-a"},
        {"operator": "fft", "precision": "fp32", "logN": 8, "batch": 4, "kind": "linked"},
        {"operator": "fft", "precision": "fp32", "logN": 8, "batch": 8, "kind": "gap"},
    ]
    runtime_policies = []
    projection_policies = []

    def runtime(binary, workload, runner):
        runtime_policies.append(os.environ.get("CUBUTTERFLY_COMPILE_MODE"))
        assert workload["operator"] == "fft"
        return points

    projector = ModuleType("research_compile_requests")

    def compile_request(point, policy="research"):
        projection_policies.append(policy)
        if point["kind"] == "linked":
            return None
        if point["kind"] == "gap":
            raise ValueError("unresolved stage_partition")
        return _request(8)

    projector.compile_request = compile_request
    monkeypatch.setattr(precompile_research, "runtime_candidates", runtime)
    monkeypatch.setattr(precompile_research, "_device_identity",
                        lambda *args: {"device": "Test GPU", "compute_capability": "8.0",
                                       "global_memory_bytes": 4096, "source": "mock"})
    monkeypatch.setitem(sys.modules, "research_compile_requests", projector)
    monkeypatch.setenv("CUBUTTERFLY_COMPILE_MODE", "auto")

    code = precompile_research.main([
        "--mode", "export", "--policy", "research", "--output-dir", str(output),
        "--build-dir", str(build), "--workloads", str(workloads), "--profile", str(profile),
    ])

    assert code == 1
    assert runtime_policies == ["research"]
    assert projection_policies == ["research"] * 4
    assert os.environ["CUBUTTERFLY_COMPILE_MODE"] == "auto"
    manifest = json.loads((output / precompile_research.MODULES_NAME).read_text())
    assert manifest["policy"] == "research"
    assert manifest["profile"]["sm"] == 80
    assert manifest["device_identity"]["global_memory_bytes"] == 4096
    assert manifest["complete"] is False
    assert manifest["candidate_source"] == "runtime"
    assert manifest["coverage"]["full_space"] is False
    assert manifest["coverage"]["full_runtime_inventory"] is True
    assert manifest["module_count"] == 1
    cell = manifest["cells"][0]
    assert cell["candidate_count"] == 4
    assert cell["projected_count"] == 2
    assert cell["module_count"] == 1
    assert cell["no_jit_count"] == 1
    assert cell["cannot_project_count"] == 1
    assert cell["cannot_project_reasons"] == {"unresolved stage_partition": 1}
    assert len(manifest["modules"]) == 1
    assert manifest["modules"][0]["policy"] == "research"
    assert manifest["modules"][0]["candidate_source"] == "runtime"


def test_export_seed_source_skips_runtime_and_reports_bounded_coverage(tmpdir, monkeypatch):
    root = Path(str(tmpdir))
    output = root / "export"
    build = root / "build"
    build.mkdir()
    (build / "cubutterfly_bench").write_bytes(b"mock benchmark")
    workloads = root / "workloads.json"
    _write_json(workloads, {
        "schema": "cubutterfly-install-search-v1",
        "workloads": [{"operator": "fft", "precision": "fp32", "logN": 8,
                       "batch": 1, "placement": "out-of-place", "normalization": "none"}],
    })
    profile = root / "profile.json"
    _write_json(profile, {"device": "Test GPU", "compute_capability": "8.0",
                          "global_memory_bytes": 4096})
    seed_file = root / "mapping-seeds.json"
    _write_json(seed_file, {"schema": "cubutterfly-mapping-seeds-v1", "seeds": []})
    snapshot_path = output / precompile_research.SEED_SNAPSHOT_NAME
    _write_json(snapshot_path, {"schema": "cubutterfly-mapping-seeds-v1",
                                "inputs": {}, "seeds": []})
    seed_point = {"operator": "fft", "precision": "fp32", "logN": 8,
                  "batch": 1, "kind": "historical-seed"}
    seed_module = ModuleType("calibration_seeds")
    seed_module.candidates_for_workload = lambda snapshot, workload: [{"point": seed_point}]
    monkeypatch.setitem(sys.modules, "calibration_seeds", seed_module)

    def no_runtime(*args):
        raise AssertionError("seed candidate source must not enumerate runtime candidates")

    projector = ModuleType("research_compile_requests")
    projector.compile_request = lambda point, policy="research": _request(8)
    monkeypatch.setattr(precompile_research, "runtime_candidates", no_runtime)
    monkeypatch.setattr(precompile_research, "_device_identity",
                        lambda *args: {"device": "Test GPU", "compute_capability": "8.0",
                                       "global_memory_bytes": 4096, "source": "mock"})
    monkeypatch.setattr(precompile_research, "_seed_snapshot",
                        lambda args, output_dir: ({"seeds": [{"point": seed_point}]},
                                                   snapshot_path))
    monkeypatch.setitem(sys.modules, "research_compile_requests", projector)
    worker_counts = []
    _inline_executor(monkeypatch, worker_counts)
    monkeypatch.setattr(precompile_research, "_compile_one",
                        lambda task: {"id": task["id"], "path": "/cpu/mock.so",
                                      "status": "compiled", "elapsed_seconds": 0.0,
                                      "error": None})

    code = precompile_research.main([
        "--mode", "all", "--candidate-source", "seeds", "--policy", "research",
        "--output-dir", str(output), "--build-dir", str(build),
        "--workloads", str(workloads), "--profile", str(profile),
        "--mapping-seeds", str(seed_file), "--jobs", "1",
    ])

    assert code == 0
    manifest = json.loads((output / precompile_research.MODULES_NAME).read_text())
    assert manifest["candidate_source"] == "seeds"
    assert manifest["coverage"] == {
        "source": "seeds",
        "scope": "mapping-seeds-snapshot",
        "full_space": False,
        "full_runtime_inventory": False,
        "limitation": "seed candidates are a bounded subset; runtime enumeration is omitted",
    }
    assert manifest["cells"][0]["candidate_count"] == 1
    module = manifest["modules"][0]
    assert module["candidate_source"] == "seeds"
    assert set(module) >= {
        "id", "request", "representative_point", "representative_cell",
        "cells", "policy", "candidate_source", "coverage",
    }
    journal = json.loads((output / precompile_research.COMPILATION_NAME).read_text())
    assert journal["candidate_source"] == "seeds"
    assert journal["coverage"]["full_space"] is False
    assert journal["coverage"]["full_runtime_inventory"] is False
    assert journal["complete"] is True
    assert worker_counts == [1]


def test_seed_source_with_no_matching_candidates_is_incomplete(tmpdir, monkeypatch):
    root = Path(str(tmpdir))
    output = root / "export"
    build = root / "build"
    build.mkdir()
    (build / "cubutterfly_bench").write_bytes(b"mock benchmark")
    workloads = root / "workloads.json"
    _write_json(workloads, {
        "schema": "cubutterfly-install-search-v1",
        "workloads": [{"operator": "fft", "precision": "fp32", "logN": 12,
                       "batch": 1, "placement": "out-of-place", "normalization": "none"}],
    })
    profile = root / "profile.json"
    _write_json(profile, {"device": "Test GPU", "compute_capability": "8.0",
                          "global_memory_bytes": 4096})
    seed_file = root / "mapping-seeds.json"
    _write_json(seed_file, {"schema": "cubutterfly-mapping-seeds-v1", "seeds": []})
    snapshot_path = output / precompile_research.SEED_SNAPSHOT_NAME
    _write_json(snapshot_path, {"schema": "cubutterfly-mapping-seeds-v1",
                                "inputs": {}, "seeds": []})

    monkeypatch.setattr(precompile_research, "runtime_candidates",
                        lambda *args: pytest.fail("seed source enumerated runtime candidates"))
    monkeypatch.setattr(precompile_research, "_device_identity",
                        lambda *args: {"device": "Test GPU", "compute_capability": "8.0",
                                       "global_memory_bytes": 4096, "source": "mock"})
    monkeypatch.setattr(precompile_research, "_seed_snapshot",
                        lambda args, output_dir: ({"seeds": []}, snapshot_path))

    code = precompile_research.main([
        "--mode", "export", "--candidate-source", "seeds", "--policy", "research",
        "--output-dir", str(output), "--build-dir", str(build),
        "--workloads", str(workloads), "--profile", str(profile),
        "--mapping-seeds", str(seed_file),
    ])

    assert code == 1
    manifest = json.loads((output / precompile_research.MODULES_NAME).read_text())
    assert manifest["status"] == "incomplete"
    assert manifest["complete"] is False
    assert manifest["module_count"] == 0
    assert manifest["cells"][0]["candidate_count"] == 0
    assert "produced no candidates" in manifest["cells"][0]["enumeration_error"]
    assert manifest["coverage"]["full_runtime_inventory"] is False


def test_seed_source_requires_explicit_mapping_seed_file(tmpdir):
    output = Path(str(tmpdir)) / "export"
    with pytest.raises(ValueError, match="requires at least one explicit --mapping-seeds file"):
        precompile_research.main([
            "--mode", "export", "--candidate-source", "seeds",
            "--output-dir", str(output), "--build-dir", str(output / "build"),
            "--workloads", str(output / "workloads.json"),
            "--profile", str(output / "profile.json"),
        ])


def test_compile_failure_is_retryable_and_old_journal_does_not_skip_modules(tmpdir, monkeypatch):
    output = Path(str(tmpdir)) / "retry"
    output.mkdir()
    requests = [_request(8), _request(9)]
    _write_json(output / precompile_research.MODULES_NAME,
                _manifest(output, requests))
    identifiers = [precompile_research._module_id(request) for request in requests]
    failing = {identifiers[1]}
    calls = []
    worker_counts = []
    _inline_executor(monkeypatch, worker_counts)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")

    def compile_one(task):
        calls.append((task["id"], os.environ.get("CUDA_VISIBLE_DEVICES")))
        if task["id"] in failing:
            return {"id": task["id"], "path": None, "status": "failed",
                    "elapsed_seconds": 0.0, "error": "mock compiler failure"}
        return {"id": task["id"], "path": "/cpu/mock.so", "status": "compiled",
                "elapsed_seconds": 0.0, "error": None}

    monkeypatch.setattr(precompile_research, "_compile_one", compile_one)
    first = precompile_research.main([
        "--mode", "compile", "--output-dir", str(output), "--jobs", "3",
    ])
    journal = json.loads((output / precompile_research.COMPILATION_NAME).read_text())
    assert first == 1
    assert journal["status"] == "failed"
    assert journal["complete"] is False
    assert journal["successful_modules"] == 1
    assert journal["failed_modules"] == 1
    assert {item[0] for item in calls} == set(identifiers)
    assert all(item[1] == "" for item in calls)
    assert worker_counts == [3]
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0"

    calls[:] = []
    failing.clear()
    second = precompile_research.main([
        "--mode", "compile", "--output-dir", str(output), "--jobs", "3",
    ])
    journal = json.loads((output / precompile_research.COMPILATION_NAME).read_text())
    assert second == 0
    assert journal["status"] == "complete"
    assert journal["complete"] is True
    assert journal["successful_modules"] == 2
    assert {item[0] for item in calls} == set(identifiers)
    assert worker_counts == [3, 3]


def test_compile_interrupt_writes_partial_journal(tmpdir, monkeypatch):
    output = Path(str(tmpdir)) / "interrupt"
    output.mkdir()
    requests = [_request(8), _request(9)]
    _write_json(output / precompile_research.MODULES_NAME,
                _manifest(output, requests))
    identifiers = [precompile_research._module_id(request) for request in requests]
    worker_counts = []
    _inline_executor(monkeypatch, worker_counts)

    def compile_one(task):
        if task["id"] == identifiers[1]:
            raise KeyboardInterrupt()
        return {"id": task["id"], "path": "/cpu/mock.so", "status": "compiled",
                "elapsed_seconds": 0.0, "error": None}

    monkeypatch.setattr(precompile_research, "_compile_one", compile_one)
    code = precompile_research.main([
        "--mode", "compile", "--output-dir", str(output), "--jobs", "2",
    ])

    journal = json.loads((output / precompile_research.COMPILATION_NAME).read_text())
    assert code == 130
    assert journal["status"] == "partial"
    assert journal["interrupted"] is True
    assert journal["complete"] is False
    assert journal["completed_modules"] == 1
    assert worker_counts == [2]


def test_policy_and_cache_defaults_follow_environment(tmpdir, monkeypatch):
    output = Path(str(tmpdir)) / "manifest"
    xdg = Path(str(tmpdir)) / "xdg"
    monkeypatch.delenv("CUBUTTERFLY_JIT_CACHE", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(xdg))
    monkeypatch.setenv("CUBUTTERFLY_COMPILE_MODE", "auto")
    args = precompile_research.parse_args(["--mode", "compile", "--output-dir", str(output)])
    assert args.policy == "auto"
    assert args.cache == (xdg / "cubutterfly" / "modules").resolve()

    jit = Path(str(tmpdir)) / "jit"
    monkeypatch.setenv("CUBUTTERFLY_JIT_CACHE", str(jit))
    args = precompile_research.parse_args(["--mode", "compile", "--output-dir", str(output)])
    assert args.cache == jit.resolve()
