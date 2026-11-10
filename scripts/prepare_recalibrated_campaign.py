#!/usr/bin/env python3
"""Prepare the bounded 2026-10-06 retest campaign without running CUDA.

The campaign is deliberately a plan, not a result importer.  It snapshots the
finite gap matrix, creates a fresh rank-1 comparison manifest, and records the
current source/build identities.  No benchmark, CUDA runtime, or device query
is started by this script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "results" / "recalibrated_campaign_20261006"
GAP_INPUT = ROOT / "results" / "operator_coverage_gap_20261006" / "manifest.json"
CURRENT_BUILD = ROOT / "build-e13-cufftdx"
BASELINE_BUILD = ROOT / "build-a100-vkfft"
HISTORICAL_CONFIG = ROOT / "config" / "a100_comprehensive_suite.json"
EXTERNAL_CATALOG = ROOT / "config" / "external_comparison_catalog.json"
EXTERNAL_SUITE_INPUT = ROOT / "config" / "a100_c0_external_rebenchmark.json"
TARGET_GPU_UUID = os.environ.get("CUBUTTERFLY_GPU_UUID", "")
TARGET_HARDWARE_LABEL = (
    "NVIDIA A100-PCIE-80GB / sm_80 / "
    f"target UUID {TARGET_GPU_UUID} (requires runtime confirmation)"
)
MATRIX_INPUTS = (
    ("gap_matrix", GAP_INPUT, "finite gap rows and explicit unsupported boundaries"),
    ("xd_coverage_axes", ROOT / "results" / "all_operators_xd_20261006_gpu1_coverage_v4" / "manifest.json", "shape/batch/operator axes only; timing rows forbidden"),
    ("xd_scale_axes", ROOT / "results" / "all_operators_xd_20261006_gpu1_scale_v4" / "manifest.json", "shape/batch/operator axes only; timing rows forbidden"),
    ("historical_comprehensive_37", ROOT / "results" / "comprehensive_current_20261006_cufftdx" / "manifest.json", "provenance only; explicitly not full breadth and no timing reuse"),
    ("historical_comprehensive_config", HISTORICAL_CONFIG, "historical case semantics only; fresh suite is generated below"),
    ("external_comparison_catalog", EXTERNAL_CATALOG, "runner contracts and availability labels; no result rows"),
    ("historical_external_suite", EXTERNAL_SUITE_INPUT, "finite matching-protocol workload matrix only; fresh raw CSV required"),
)

FAMILIES: tuple[dict[str, Any], ...] = (
    {
        "name": "fft",
        "public_operator": "fft",
        "precisions": ("fp32", "fp64", "fp16", "bf16"),
        "low_precision_accumulations": ("native", "fp32"),
        "external_references": ("cufft", "vkfft"),
    },
    {
        "name": "ntt",
        "public_operator": "ntt",
        "precisions": ("uint32", "uint64"),
        "low_precision_accumulations": (),
        "external_references": (),
    },
    {
        "name": "fwht",
        "public_operator": "fwht",
        "precisions": ("fp32", "fp64", "fp16", "bf16"),
        "low_precision_accumulations": ("native", "fp32"),
        "external_references": (),
    },
    {
        "name": "subset-zeta",
        "public_operator": "subset-zeta",
        "precisions": ("uint32",),
        "low_precision_accumulations": (),
        "external_references": (),
    },
    {
        "name": "superset-zeta",
        "public_operator": "superset-zeta",
        "precisions": ("uint32",),
        "low_precision_accumulations": (),
        "external_references": (),
    },
    {
        "name": "xor-zeta",
        "public_operator": "xor-zeta",
        "alias_of": "subset-zeta",
        "precisions": ("uint32",),
        "low_precision_accumulations": (),
        "external_references": (),
    },
    {
        "name": "structured-2x2",
        "public_operator": "structured-2x2",
        "precisions": ("fp32", "fp64", "fp16", "bf16"),
        "low_precision_accumulations": ("native", "fp32"),
        "external_references": (),
    },
)

PROFILE_OPERATOR_PAIRS: tuple[dict[str, Any], ...] = (
    {"name": "fft", "precisions": ("fp32", "fp64")},
    {"name": "fwht", "precisions": ("fp32", "fp64")},
    {"name": "ntt", "precisions": ("uint32", "uint64")},
    {"name": "subset-zeta", "precisions": ("uint32",)},
    {"name": "superset-zeta", "precisions": ("uint32",)},
    {"name": "structured-2x2", "precisions": ("fp32", "fp64")},
)

PROFILE_SHAPES = {
    "coverage": ((8, 8), (32, 32), (4, 4, 4), (8, 8, 8)),
    "scale": ((128, 128), (16, 16, 16), (32, 32, 32), (64, 64, 64)),
}
PROFILE_BATCHES = {"coverage": (1, 4, 16), "scale": (1, 4, 16)}

SOURCE_PATTERNS = (
    "CMakeLists.txt",
    "cmake/**/*.cmake",
    "include/**/*.h",
    "include/**/*.hpp",
    "src/**/*.c",
    "src/**/*.cc",
    "src/**/*.cpp",
    "src/**/*.cu",
    "src/**/*.cuh",
    "src/**/*.h",
    "src/**/*.hpp",
    "src/*.in",
    "apps/cubutterfly_bench.cpp",
    "apps/cubutterfly_plan_bench.cpp",
    "apps/cuntt_bench.cpp",
    "tests/check_xd.py",
    "scripts/check_operator_coverage_gaps.py",
    "scripts/check_gpu_operator_matrix.py",
    "tests/test_butterfly.cpp",
    "tests/test_c_api.c",
    "tests/test_ntt.cpp",
    "tests/test_xd_fft_layout.cu",
    "scripts/benchmark_all_operators_xd.py",
    "scripts/benchmark_xd.py",
    "scripts/benchmark_xd_real.py",
    "scripts/run_external_baseline_suite.py",
    "scripts/benchmark_external_fht.py",
    "scripts/benchmark_semantics.py",
    "scripts/summarize_external_baseline_suite.py",
    "scripts/prepare_recalibrated_campaign.py",
    "scripts/run_recalibrated_campaign.py",
    "scripts/summarize_recalibrated_campaign.py",
    "scripts/run_isolated_gpu.py",
    "scripts/run_comprehensive_suite.py",
    "scripts/summarize_all_operators_xd.py",
    "scripts/summarize_comprehensive_suite.py",
    "scripts/reviewed_comparison.py",
    "scripts/calibrat*.py",
    "scripts/stage_*.py",
    "scripts/compile_module.py",
    "scripts/resident_mapping.py",
    "scripts/local_selector_data.py",
    "scripts/initialize_hardware_profile.py",
    "scripts/check_hardware_profile.py",
    "scripts/finalize_recalibration.py",
    "scripts/hardware_registry.py",
    "scripts/run_research_acceptance.py",
    "scripts/verify_local_selector.py",
    "scripts/fit_local_cost_model.py",
    "scripts/schedule_cost_model.py",
    "scripts/pipeline_schedule_model.py",
    "scripts/evolutionary_search.py",
    "config/install_search_workloads.json",
    "config/platform_tuning/*.json",
)

BINARY_PATHS = {
    "current_butterfly": CURRENT_BUILD / "cubutterfly_bench",
    "current_plan_bench": CURRENT_BUILD / "cubutterfly_plan_bench",
    "current_ntt": CURRENT_BUILD / "cuntt_bench",
    "current_library": CURRENT_BUILD / "libcuntt.so",
    "current_c_api_tests": CURRENT_BUILD / "cubutterfly_c_api_tests",
    "current_butterfly_tests": CURRENT_BUILD / "cubutterfly_tests",
    "current_ntt_tests": CURRENT_BUILD / "cuntt_tests",
    "current_xd_layout_tests": CURRENT_BUILD / "cubutterfly_xd_fft_layout_tests",
    "current_rank3_tests": CURRENT_BUILD / "cubutterfly_rank3_permute_tests",
    "current_planner_tests": CURRENT_BUILD / "cubutterfly_public_planner_tests",
    "current_mapping_tests": CURRENT_BUILD / "cubutterfly_mapping_tests",
    "baseline_fft": BASELINE_BUILD / "vkfft_bench",
}

FHT_PYTHON_CANDIDATES = (Path(sys.executable),)
GPU_NTT_BINARY_CANDIDATES = (
    Path("/tmp/gpuntt_merge_gap_bench_a100"),
    Path("/tmp/gpuntt_merge_gap_bench"),
    ROOT / "build-a100-gpuntt" / "gpuntt_merge_gap_bench",
    ROOT / "build-v100-sm70" / "gpuntt_merge_gap_bench",
)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def _source_files() -> list[Path]:
    paths: set[Path] = set()
    for pattern in SOURCE_PATTERNS:
        candidate = ROOT / pattern
        if candidate.is_file():
            paths.add(candidate)
        else:
            paths.update(path for path in ROOT.glob(pattern) if path.is_file())
    return sorted(paths)


def _source_freeze() -> dict[str, Any]:
    source_paths = _source_files()
    inputs = []
    digest = hashlib.sha256()
    for path in source_paths:
        relative = _relative(path)
        file_hash = _sha256_file(path)
        if file_hash is None:
            continue
        inputs.append({"path": relative, "sha256": file_hash})
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--short", "--", *[_relative(path) for path in source_paths]],
            cwd=ROOT, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        commit = None
        status = f"identity query failed: {exc}"
    return {
        "git_commit": commit,
        "working_tree_dirty": bool(status),
        "working_tree_status_scope": "source inputs only",
        "working_tree_status_sha256": _sha256_bytes(status.encode()),
        "source_sha256": digest.hexdigest(),
        "input_count": len(inputs),
        "inputs": inputs,
    }


def _parse_cache(path: Path) -> dict[str, str | None]:
    wanted = {
        "CMAKE_BUILD_TYPE",
        "CMAKE_CUDA_ARCHITECTURES",
        "CUBUTTERFLY_ENABLE_CUFFTDX",
        "CUBUTTERFLY_ENABLE_XD_CUFFTDX",
        "CMAKE_CUDA_COMPILER_VERSION",
    }
    values: dict[str, str | None] = {key: None for key in wanted}
    if path.is_file():
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if "=" not in line or line.startswith("//"):
                continue
            key, value = line.split("=", 1)
            key = key.split(":", 1)[0]
            if key in wanted:
                values[key] = value
    return values


def _artifact(path: Path, *, required: bool) -> dict[str, Any]:
    return {
        "path": _relative(path),
        "required": required,
        "exists": path.is_file(),
        "sha256": _sha256_file(path),
        "size_bytes": path.stat().st_size if path.is_file() else None,
    }


def _matrix_input_freeze() -> list[dict[str, Any]]:
    records = []
    for name, path, role in MATRIX_INPUTS:
        records.append({
            "name": name,
            "path": _relative(path),
            "role": role,
            "exists": path.is_file(),
            "sha256": _sha256_file(path),
            "size_bytes": path.stat().st_size if path.is_file() else None,
        })
    return records


def _first_executable(candidates: Iterable[Path]) -> Path | None:
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate.resolve()
    return None


def _module_paths(python: Path, module: str) -> list[Path]:
    """Find a module by path only; importing it could initialize CUDA."""
    prefixes = [python.resolve().parent.parent]
    paths: list[Path] = []
    for prefix in prefixes:
        for site in prefix.glob("lib/python*/site-packages"):
            paths.append((site / module / "__init__.py").resolve())
            paths.extend(sorted(path.resolve() for path in site.glob(f"{module}_*.so")))
    return paths


def _external_tool_inventory() -> dict[str, Any]:
    fht_candidates = []
    if os.environ.get("FHT_PYTHON"):
        fht_candidates.append(Path(os.environ["FHT_PYTHON"]))
    fht_candidates.append(Path(sys.executable))
    fht_candidates.extend(FHT_PYTHON_CANDIDATES)
    fht_python = None
    fht_modules: list[Path] = []
    fallback_python = None
    fallback_modules: list[Path] = []
    for candidate in fht_candidates:
        if not candidate.is_file() or not os.access(candidate, os.X_OK):
            continue
        resolved = candidate.resolve()
        modules = _module_paths(resolved, "fast_hadamard_transform")
        native = any(path.suffix == ".so" and path.is_file() for path in modules)
        if fallback_python is None:
            fallback_python, fallback_modules = resolved, modules
        if any(path.name == "__init__.py" and path.is_file() for path in modules) and native:
            fht_python, fht_modules = resolved, modules
            break
    if fht_python is None:
        fht_python, fht_modules = fallback_python, fallback_modules
    fht_module = next((path for path in fht_modules if path.is_file()), None)
    fht_native = next((path for path in fht_modules if path.suffix == ".so" and path.is_file()), None)

    gpuntt_candidates = []
    if os.environ.get("GPUNTT_BIN"):
        gpuntt_candidates.append(Path(os.environ["GPUNTT_BIN"]))
    gpuntt_candidates.extend(GPU_NTT_BINARY_CANDIDATES)
    gpuntt_binary = _first_executable(gpuntt_candidates)

    return {
        "dao-fht": {
            "catalog_runner": "scripts/benchmark_external_fht.py",
            "runner_exists": (ROOT / "scripts" / "benchmark_external_fht.py").is_file(),
            "python": str(fht_python) if fht_python else None,
            "module": str(fht_module) if fht_module else None,
            "module_sha256": _sha256_file(fht_module) if fht_module else None,
            "native_module": str(fht_native) if fht_native else None,
            "native_module_sha256": _sha256_file(fht_native) if fht_native else None,
            "available": bool(fht_python and fht_module and fht_native and
                               (ROOT / "scripts" / "benchmark_external_fht.py").is_file()),
            "verification": "path/module metadata only; no CUDA import or benchmark",
        },
        "gpu-ntt": {
            "catalog_runner": "scripts/run_external_baseline_suite.py",
            "runner_exists": (ROOT / "scripts" / "run_external_baseline_suite.py").is_file(),
            "binary": str(gpuntt_binary) if gpuntt_binary else None,
            "binary_sha256": _sha256_file(gpuntt_binary) if gpuntt_binary else None,
            "available": bool(gpuntt_binary and
                               (ROOT / "scripts" / "run_external_baseline_suite.py").is_file()),
            "verification": "executable metadata only; --help was not invoked because the harness initializes CUDA",
        },
    }


def _load_external_suite() -> dict[str, Any]:
    document = json.loads(EXTERNAL_SUITE_INPUT.read_text(encoding="utf-8"))
    if document.get("schema_version") != 1 or not isinstance(document.get("workloads"), list):
        raise ValueError("A100 external comparison suite has an unsupported schema")
    for workload in document["workloads"]:
        if workload.get("operator") != "ntt":
            continue
        workload["input_order"] = workload.get("input_order") or "natural"
        workload["output_order"] = workload.get("output_order") or "natural"
        for implementation in workload.get("implementations", []):
            if implementation.get("runner") != "ntt":
                continue
            implementation["name"] = "cuNTT-auto-select"
            implementation["args"] = _ntt_auto_select_args(
                precision=workload.get("precision"),
                word_bits=workload.get("word_bits"),
                direction=workload.get("direction", "forward"),
                input_order=workload["input_order"],
                output_order=workload["output_order"],
            )
            implementation["mapping_policy"] = "auto-select-requires-observed-mapping"
    return document


def _build_identity() -> dict[str, Any]:
    return {
        "current": {
            "build_dir": _relative(CURRENT_BUILD),
            "cmake_cache": _parse_cache(CURRENT_BUILD / "CMakeCache.txt"),
            "artifacts": {
                name: _artifact(path, required=True)
                for name, path in BINARY_PATHS.items()
                if name.startswith("current_")
            },
        },
        "baseline": {
            "build_dir": _relative(BASELINE_BUILD),
            "cmake_cache": _parse_cache(BASELINE_BUILD / "CMakeCache.txt"),
            "artifacts": {"baseline_fft": _artifact(BINARY_PATHS["baseline_fft"], required=True)},
        },
    }


def _load_gap() -> dict[str, Any]:
    if not GAP_INPUT.is_file():
        raise FileNotFoundError(GAP_INPUT)
    document = json.loads(GAP_INPUT.read_text(encoding="utf-8"))
    if not isinstance(document.get("rows"), list) or len(document["rows"]) != 113:
        raise ValueError("operator coverage gap manifest must contain its 113-row finite matrix")
    return document


def _shell(command: Iterable[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in command)


def _current(path: str) -> str:
    return str(Path(path))


def _gap_case(row: dict[str, Any]) -> dict[str, Any]:
    section = row["section"]
    if row["coverage_state"] == "unsupported":
        initial = "unsupported"
    else:
        initial = "pending"
    if section == "low_precision_public_api":
        runner = "core-low-precision-contract"
        evidence = "low-level-only; public C-ABI row has no current execution evidence"
    elif section == "unsupported_real_low_precision":
        runner = "public-cabi-negative-test"
        evidence = "public C-ABI rejects this storage/transform combination"
    else:
        runner = "tests/check_xd.py"
        evidence = "current-build public C-ABI or semantic anchor"
    reference = "missing-external-reference"
    if row["operator"] == "fft" and row.get("fft_type") in ("c2c", "r2c", "c2r"):
        reference = "cufft-available"
    if row["operator"] == "fft" and row.get("fft_type") == "c2c" and row.get("rank") == 1:
        reference = "cufft-vkfft-for-fp32-c2c" if row.get("precision") == "fp32" else "cufft-available"
    return {
        "id": row["id"],
        "family": row["operator"],
        "precision": row["precision"],
        "direction": row["direction"],
        "rank": row["rank"],
        "shape": row.get("shape", row.get("logical_shape")),
        "batch": row["batch"],
        "section": section,
        "coverage_state": row["coverage_state"],
        "expected_execution": row["expected_execution"],
        "initial_status": initial,
        "runner": runner,
        "evidence_scope": evidence,
        "comparison_reference": reference,
        "source_gap_row": row,
    }


def _alias_rows(profile: str, configurations: Iterable[str]) -> int:
    shape_count = len(PROFILE_SHAPES[profile])
    batch_count = len(PROFILE_BATCHES[profile])
    return shape_count * batch_count * 2 * len(tuple(configurations))


def _profile_definition(profile: str, *, skip_resident: bool) -> dict[str, Any]:
    configurations = ("selected",) if skip_resident else ("selected", "resident-t128")
    arithmetic_rows = sum(len(item["precisions"]) for item in PROFILE_OPERATOR_PAIRS)
    arithmetic_rows *= len(PROFILE_SHAPES[profile]) * len(PROFILE_BATCHES[profile]) * 2 * len(configurations)
    alias_rows = _alias_rows(profile, configurations)
    return {
        "profile": profile,
        "runner": "scripts/benchmark_all_operators_xd.py",
        "operator_pairs": [
            {"name": item["name"], "precisions": list(item["precisions"])}
            for item in PROFILE_OPERATOR_PAIRS
        ],
        "shapes": [list(shape) for shape in PROFILE_SHAPES[profile]],
        "batches": list(PROFILE_BATCHES[profile]),
        "directions": ["forward", "inverse"],
        "configurations": list(configurations),
        "arithmetic_rows": arithmetic_rows,
        "alias_rows_inherited_from_subset_zeta": alias_rows,
        "semantic_rows_including_alias": arithmetic_rows + alias_rows,
        "timing_inputs": "none; output directory must be new",
    }


def _butterfly_args(operator: str, precision: str, inverse: bool,
                    accumulation: str | None = None, *, log_n: int | None = None) -> list[str]:
    """Build a legal public benchmark contract for one butterfly case.

    The public ``fp16-fp32`` precision is the legacy WMMA D8 contract, not a
    general FP16 storage/FP32 accumulation spelling.  General low-precision
    rows therefore pass storage precision plus ``--accumulation fp32``.  A
    temporal tile is only legal through logN=10; larger low-precision rows use
    the capability-advertised scalar shared-iterative lowering.
    """
    legacy_wmma = precision == "fp16-fp32" and operator == "fft" and (log_n is None or log_n == 3)
    if legacy_wmma:
        args = ["--operator", operator, "--precision", precision,
                "--backend", "temporal-tile", "--fft-core", "wmma-dft8",
                "--compute-unit", "radix8", "--tile-threads", "256"]
    else:
        storage_precision = precision[:-len("-fp32")] if precision.endswith("-fp32") else precision
        low_precision = storage_precision in {"fp16", "bf16"}
        args = ([] if low_precision else ["--auto-select"])
        args += ["--operator", operator, "--precision", storage_precision]
        if low_precision:
            if log_n is not None and log_n > 10:
                args += ["--backend", "shared-iterative", "--compute-unit", "radix2",
                         "--local-stages", "6", "--tile-threads", "128"]
            else:
                args += ["--backend", "temporal-tile", "--compute-unit", "radix2", "--tile-threads", "128"]
    if accumulation is not None:
        # Fp16Fp32 is already the WMMA core's fixed numeric contract; the
        # benchmark's parser rejects an additional accumulation override.
        if not legacy_wmma:
            args += ["--accumulation", accumulation]
    if operator == "structured-2x2":
        args += ["--stage-matrix", "1,0.25,-0.5,1"]
    if operator in {"fft", "fwht", "structured-2x2"}:
        args += ["--placement", "in-place"]
        args += ["--normalization", "inverse" if inverse else "none"]
    if inverse:
        args.append("--inverse")
    return args


def _ntt_auto_select_args(*, precision: str | None = None, word_bits: int | str | None = None,
                          direction: str = "forward", input_order: str = "natural",
                          output_order: str = "natural") -> list[str]:
    """Build the public NTT AUTO contract without pinning a design point."""
    if word_bits is None:
        word_bits = 32 if str(precision) in {"uint32", "word32"} else 64
    args = ["--auto-select", "--word-bits", str(word_bits),
            "--input-order", input_order or "natural", "--output-order", output_order or "natural"]
    if direction == "inverse":
        args.append("--inverse")
    return args


def _suite_case(case_id: str, group: str, operator: str, precision: str,
                log_n: int, batch: int, inverse: bool, *, tier: str,
                accumulation: str | None = None, implementation: str | None = None,
                reference: bool = False, runner: str = "butterfly",
                args: list[str] | None = None, mapping_policy: str = "auto-select",
                element_stride: int | None = None, batch_padding: int | None = None,
                modulus_bits: int | None = None, source_case_id: str | None = None,
                source_config: str | None = None, cohort: str | None = None) -> dict[str, Any]:
    if args is None:
        if runner == "ntt":
            args = _ntt_auto_select_args(
                precision=precision,
                direction="inverse" if inverse else "forward",
            )
        else:
            args = _butterfly_args(operator, precision, inverse, accumulation, log_n=log_n)
    result = {
        "id": case_id,
        "tier": tier,
        "group": group,
        "runner": runner,
        "implementation": implementation or "cuButterfly-auto-select",
        "reference": reference,
        "operator": operator,
        "precision": precision,
        "logN": log_n,
        "batch": batch,
        "accumulation": accumulation or "native",
        "mapping_policy": mapping_policy,
        "args": args,
    }
    for key, value in (
        ("element_stride", element_stride),
        ("batch_padding", batch_padding),
        ("modulus_bits", modulus_bits),
        ("source_case_id", source_case_id),
        ("source_config", source_config),
        ("cohort", cohort),
    ):
        if value is not None:
            result[key] = value
    return result


def _historical_suite_cases() -> list[dict[str, Any]]:
    """Re-express every old case as a fresh current-run case.

    The old JSON is read for semantic axes only.  Its result references and
    timing files are never loaded.  Current butterfly and NTT cases use their
    public AUTO selection contracts; fixed backend/design arguments are not
    copied from the historical manifest.
    """
    document = json.loads(HISTORICAL_CONFIG.read_text(encoding="utf-8"))
    if document.get("schema_version") != 1 or len(document.get("cases", [])) != 37:
        raise ValueError("historical A100 suite must provide exactly 37 case semantics")
    cases: list[dict[str, Any]] = []
    for source in document["cases"]:
        source_args = [str(value) for value in source["args"]]
        log_n = int(source["logN"])
        tier = str(source["tier"])
        protocol = document["protocols"][tier]
        batch = int(source.get("batch", max(1, int(protocol["target_points"]) // (1 << log_n))))
        operator = str(source["operator"])
        precision = str(source["precision"])
        inverse = "--inverse" in source_args or source.get("direction") == "inverse"
        common = {
            "case_id": f"historical_{source['id']}",
            "group": f"historical_{source['group']}",
            "operator": operator,
            "precision": precision,
            "log_n": log_n,
            "batch": batch,
            "inverse": inverse,
            "tier": tier,
            "source_case_id": str(source["id"]),
            "source_config": _relative(HISTORICAL_CONFIG),
            "cohort": "historical-37-case",
            "element_stride": int(source.get("element_stride", 1)),
            "batch_padding": int(source.get("batch_padding", 0)),
        }
        if source["runner"] == "fft-library":
            # These remain true external reference rows.  The paired current
            # row is regenerated below/alongside them, and no old timing is
            # copied into either result set.
            cases.append(_suite_case(
                implementation=str(source["implementation"]),
                reference=bool(source.get("reference", False)),
                runner="fft-library", args=source_args,
                mapping_policy="external-reference",
                **common,
            ))
            continue
        if (source["runner"] == "butterfly" and
                _suite_arg(source_args, "--backend", "") == "cufft"):
            # cuFFT also has a ButterflyPlan adapter for inverse, FP64 and
            # strided cases. It remains a reference, regardless of CLI runner.
            cases.append(_suite_case(
                implementation="cuFFT", reference=bool(source.get("reference", False)),
                runner="butterfly", args=source_args,
                mapping_policy="external-reference", **common))
            continue
        if source["runner"] == "ntt":
            current_args = _ntt_auto_select_args(
                word_bits=_suite_arg(source_args, "--word-bits", "64"),
                direction="inverse" if inverse else "forward",
                input_order=_suite_arg(source_args, "--input-order", "natural"),
                output_order=_suite_arg(source_args, "--output-order", "natural"),
            )
            cases.append(_suite_case(
                implementation="cuNTT-auto-select",
                runner="ntt", args=current_args,
                mapping_policy="auto-select-requires-observed-mapping",
                modulus_bits=int(source.get("modulus_bits", 60)),
                **common,
            ))
            continue

        accumulation = "native" if precision == "fp16-fp32" else ("fp32" if precision.endswith("-fp32") else "native")
        current_args = _butterfly_args(operator, precision, inverse, accumulation, log_n=log_n)
        cases.append(_suite_case(
            implementation="cuButterfly-auto-select",
            runner="butterfly", args=current_args,
            accumulation=accumulation,
            mapping_policy=("alias-of-subset-zeta" if operator == "xor-zeta" else
                            "explicit-low-precision-backend" if precision.startswith(("fp16", "bf16")) else
                            "auto-select-requires-observed-mapping"),
            **common,
        ))
    return cases


def _make_comparison_suite() -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    rank1_points = ((8, 1, "quick"), (12, 16, "full"))
    for family in FAMILIES:
        operator = family["public_operator"]
        if "alias_of" in family:
            precisions = ("uint32",)
        else:
            precisions = family["precisions"]
        for precision in precisions:
            accumulations = family["low_precision_accumulations"] if precision in {"fp16", "bf16"} else ("native",)
            for accumulation in accumulations:
                # Keep storage precision and accumulation as separate public
                # fields.  The ``fp16-fp32`` spelling is reserved for the
                # historical WMMA D8 design point.
                cli_precision = precision
                for log_n, batch, tier in rank1_points:
                    for inverse in (False, True):
                        direction = "inv" if inverse else "fwd"
                        suffix = f"_{accumulation}" if precision in {"fp16", "bf16"} else ""
                        case_id = f"{operator}_{precision}{suffix}_log{log_n}_b{batch}_{direction}"
                        group = f"{operator}_{precision}{suffix}_log{log_n}_b{batch}_{direction}"
                        runner = "ntt" if operator == "ntt" else "butterfly"
                        cases.append(_suite_case(
                            case_id, group, operator, cli_precision, log_n, batch, inverse,
                            tier=tier, accumulation=accumulation,
                            mapping_policy=("alias-of-subset-zeta" if "alias_of" in family else
                                            "auto-select-requires-observed-mapping" if operator == "ntt" else
                                            "explicit-low-precision-backend" if precision in {"fp16", "bf16"}
                                            else "auto-select-requires-observed-mapping"),
                            runner=runner,
                            implementation="cuNTT-auto-select" if operator == "ntt" else None,
                            modulus_bits=(32 if precision == "uint32" else 60) if operator == "ntt" else None,
                        ))
                        if operator == "fft" and precision == "fp32" and accumulation == "native" and not inverse:
                            cases.append(_suite_case(
                                f"{operator}_fp32_log{log_n}_b{batch}_fwd_cufft",
                                group, operator, precision, log_n, batch, False,
                                tier=tier, implementation="cuFFT", reference=True,
                                runner="fft-library", args=["--library", "cufft"],
                                mapping_policy="external-reference",
                            ))
                            cases.append(_suite_case(
                                f"{operator}_fp32_log{log_n}_b{batch}_fwd_vkfft",
                                group, operator, precision, log_n, batch, False,
                                tier=tier, implementation="VkFFT", runner="fft-library",
                                args=["--library", "vkfft"],
                                mapping_policy="external-reference",
                            ))
                        if operator == "fft" and precision == "fp64" and not inverse:
                            cases.append(_suite_case(
                                f"{operator}_fp64_log{log_n}_b{batch}_fwd_cufft",
                                group, operator, precision, log_n, batch, False,
                                tier=tier, implementation="cuFFT", reference=True,
                                args=["--operator", "fft", "--backend", "cufft",
                                      "--precision", "fp64", "--placement", "in-place",
                                      "--normalization", "none"],
                                mapping_policy="external-reference",
                            ))

    standard_case_count = len(cases)
    historical_cases = _historical_suite_cases()
    cases.extend(historical_cases)

    # The gap manifest owns the exact modulus and large-N anchors.  Keep them
    # as explicit commands because run_comprehensive_suite derives a prime
    # from bit width rather than accepting a fixed modulus value.
    return {
        "schema_version": 1,
        "hardware": TARGET_HARDWARE_LABEL,
        "protocols": {
            "quick": {"target_points": 65536, "warmup": 10, "repeat": 50, "trials": 1, "verify_batch": 2, "verify_timed_batches": 2},
            "full": {"target_points": 1048576, "warmup": 10, "repeat": 100, "trials": 3, "verify_batch": 2, "verify_timed_batches": 2},
        },
        "binaries": {
            "butterfly": "build-e13-cufftdx/cubutterfly_bench",
            "ntt": "build-e13-cufftdx/cuntt_bench",
            "fft-library": "build-a100-vkfft/vkfft_bench",
        },
        "cases": cases,
        "cohorts": {
            "standard_rank1": standard_case_count,
            "historical_37_case": len(historical_cases),
            "timing_rows_imported": 0,
        },
        "historical_case_semantics": {
            "source_config": _relative(HISTORICAL_CONFIG),
            "source_case_count": 37,
            "fresh_mapped_case_count": len(historical_cases),
            "timings_read": False,
            "mapping": "current butterfly and cuNTT binaries with observed auto-select; NTT word/order/direction semantics remain explicit; external reference rows retain their runner",
        },
        "external_evidence": [
            {"operator": "fft", "implementation": "cuFFT", "status": "available", "reason": "legacy baseline binary is present; run fresh on target GPU"},
            {"operator": "fft", "implementation": "VkFFT", "status": "available", "reason": "legacy baseline binary is present; run fresh on target GPU"},
            {"operator": "r2c/c2r FFT", "implementation": "PyTorch/cuFFT", "status": "available", "reason": "benchmark_xd_real.py owns the fresh reference"},
            {"operator": "ntt", "implementation": "GPU-NTT", "status": "catalog-runner", "reason": "fresh fixed-prime task is generated from the A100 finite workload matrix; availability is recorded separately"},
            {"operator": "fwht", "implementation": "Dao-AILab FHT", "status": "catalog-runner", "reason": "fresh matching-protocol task is generated from the A100 finite workload matrix; availability is recorded separately"},
            {"operator": "zeta/structured", "implementation": "external", "status": "missing-external-reference", "reason": "no matching catalog runner; same-build selected/resident controls still run"},
        ],
    }


def _suite_arg(arguments: Iterable[str], option: str, default: str) -> str:
    values = list(arguments)
    try:
        index = values.index(option)
    except ValueError:
        return default
    return values[index + 1] if index + 1 < len(values) else default


def _rank1_search_workloads(suite: dict[str, Any]) -> dict[str, Any]:
    """Build future per-cell AUTO inputs without claiming calibration hits."""
    unique: dict[str, dict[str, Any]] = {}
    for case in suite["cases"]:
        if (case.get("runner") != "butterfly" or
                case.get("mapping_policy") != "auto-select-requires-observed-mapping"):
            continue
        arguments = case.get("args", [])
        semantic = {
            "operator": case["operator"],
            "precision": case["precision"],
            "accumulation": case.get("accumulation", "native"),
            "direction": "inverse" if "--inverse" in arguments else "forward",
            "normalization": _suite_arg(arguments, "--normalization", "none"),
            "placement": _suite_arg(arguments, "--placement", "out-of-place"),
            "stage_matrix": _suite_arg(arguments, "--stage-matrix", ""),
            "element_stride": int(case.get("element_stride", 1)),
            "batch_padding": int(case.get("batch_padding", 0)),
        }
        key_fields = {
            "logN": int(case["logN"]),
            "batch": int(case["batch"]),
            "semantic": semantic,
        }
        key = _canonical(key_fields).decode("ascii")
        record = unique.get(key)
        if record is None:
            unique[key] = {
                "id": f"rank1_search_{len(unique):04d}",
                "registry_key": key_fields,
                "exact_hit": False,
                "status": "unmeasured-feasible",
                "case_ids": [case["id"]],
                "cohorts": [case.get("cohort", "standard-rank1")],
                "args": list(arguments),
            }
        else:
            record["case_ids"].append(case["id"])
            cohort = case.get("cohort", "standard-rank1")
            if cohort not in record["cohorts"]:
                record["cohorts"].append(cohort)
    workloads = list(unique.values())
    return {
        "schema_version": 1,
        "kind": "rank1-search-workloads",
        "source": "comparison_suite.json",
        "selection": {
            "runner": "butterfly",
            "mapping_policy": "auto-select-requires-observed-mapping",
            "excluded": ["external-reference", "explicit-low-precision-backend", "alias-of-subset-zeta", "ntt"],
            "deduplication": "operator/precision/accumulation/direction/semantic/logN/batch",
            "registry_matching": "exact logN + batch + semantic; no nearest-neighbor or default-path inference",
        },
        "calibration_status": {
            "exact-hit": 0,
            "unmeasured-feasible": len(workloads),
            "search_executed": False,
        },
        "cases": workloads,
    }


def _external_phase(output: Path, external_suite_path: Path,
                    inventory: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return fresh external tasks and their output contract.

    The existing adapter owns CSV semantics and correctness checks.  This
    planner only freezes its finite workload input and passes current build
    binaries; it never imports CUDA or reads an old result CSV.
    """
    result = _relative(output)
    manifest = _relative(external_suite_path)
    dao_available = bool(inventory["dao-fht"]["available"])
    gpuntt_available = bool(inventory["gpu-ntt"]["available"])
    commands: list[dict[str, Any]] = []
    artifacts: list[str] = []
    task_status: dict[str, str] = {}
    dao_python = inventory["dao-fht"].get("python")
    gpuntt_binary = inventory["gpu-ntt"].get("binary")

    if dao_available and gpuntt_available:
        raw = f"{result}/comparison/external_raw.csv"
        summary = f"{result}/comparison/external_summary.csv"
        markdown = f"{result}/comparison/external_summary.md"
        commands.append({
            "id": "external_matching_protocol_comparison",
            "purpose": "fresh Dao-FHT and GPU-NTT comparisons with current butterfly/cuNTT binaries",
            "status_policy": "planned",
            "run": (
                f"CUDA_VISIBLE_DEVICES=${{GPU_UUID}} python3 scripts/run_external_baseline_suite.py "
                f"--manifest {manifest} --output {raw} --fht-python {shlex.quote(str(dao_python))} "
                f"--gpuntt-binary {shlex.quote(str(gpuntt_binary))} "
                f"--butterfly-binary build-e13-cufftdx/cubutterfly_bench "
                f"--ntt-binary build-e13-cufftdx/cuntt_bench --require-exclusive-gpu --resume"
            ),
        })
        commands.append({
            "id": "summarize_external_matching_protocol",
            "purpose": "summarize only fresh external raw CSV rows",
            "status_policy": "planned-after-external-run",
            "run": f"python3 scripts/summarize_external_baseline_suite.py {raw} --output {summary} --markdown {markdown}",
        })
        artifacts.extend(["comparison/external_raw.csv", "comparison/external_summary.csv",
                          "comparison/external_summary.md"])
        task_status = {"dao-fht": "planned", "gpu-ntt": "planned"}
    else:
        # Keep an executable task for each available adapter, while making a
        # missing dependency explicit instead of fabricating a `none` baseline.
        for name, token, only in (
            ("dao-fht", "dao", "fwht"),
            ("gpu-ntt", "gpuntt", "ntt"),
        ):
            available = bool(inventory[name]["available"])
            task_status[name] = "planned" if available else "unavailable"
            if not available:
                continue
            raw = f"{result}/comparison/external_{token}_raw.csv"
            summary = f"{result}/comparison/external_{token}_summary.csv"
            markdown = f"{result}/comparison/external_{token}_summary.md"
            tool_args = (
                f"--fht-python {shlex.quote(str(dao_python))}" if name == "dao-fht" else
                f"--gpuntt-binary {shlex.quote(str(gpuntt_binary))}"
            )
            commands.append({
                "id": f"external_{token}_comparison",
                "purpose": f"fresh {name} comparison with the current source binary",
                "status_policy": "planned",
                "run": (
                    f"CUDA_VISIBLE_DEVICES=${{GPU_UUID}} python3 scripts/run_external_baseline_suite.py "
                    f"--manifest {manifest} --only {only} --output {raw} {tool_args} "
                    f"--butterfly-binary build-e13-cufftdx/cubutterfly_bench "
                    f"--ntt-binary build-e13-cufftdx/cuntt_bench --require-exclusive-gpu --resume"
                ),
            })
            commands.append({
                "id": f"summarize_external_{token}",
                "purpose": f"summarize fresh {name} raw CSV rows",
                "status_policy": "planned-after-external-run",
                "run": f"python3 scripts/summarize_external_baseline_suite.py {raw} --output {summary} --markdown {markdown}",
            })
            artifacts.extend([f"comparison/external_{token}_raw.csv",
                              f"comparison/external_{token}_summary.csv",
                              f"comparison/external_{token}_summary.md"])
    return commands, {"task_status": task_status, "artifacts": artifacts}


def _direct_anchor_commands(gap: dict[str, Any], output: Path) -> list[dict[str, Any]]:
    result = _relative(output)
    commands = []
    for row in gap["rows"]:
        if row["section"] not in {"ntt_semantic_anchors", "large_n_batch"}:
            continue
        if row["operator"] == "ntt":
            args = [
                "build-e13-cufftdx/cuntt_bench",
                *_ntt_auto_select_args(
                    word_bits=row["word_bits"], direction=row["direction"],
                    input_order=row.get("input_order") or "natural",
                    output_order=row.get("output_order") or "natural",
                ),
                "--logN", str(row["logN"]), "--batch", str(row["batch"]),
                "--modulus", str(row["modulus"]), "--warmup", "10", "--repeat", "100",
                "--verify", "--verify-batches", "1", "--csv",
            ]
            if row["direction"] == "inverse":
                args.append("--inverse")
            commands.append({
                "id": row["id"], "family": "ntt", "status_policy": "pending",
                "comparison_reference": "missing-external-reference",
                "mapping_policy": "auto-select-requires-observed-mapping",
                "run": _shell(args) + f" > {result}/comparison/anchors/{row['id']}.csv 2>&1",
            })
        elif row["operator"] == "fft":
            precision = row["precision"]
            args = [
                "build-e13-cufftdx/cubutterfly_bench", "--auto-select", "--operator", "fft",
                "--precision", precision, "--placement", row["placement"],
                "--normalization", row["normalization"], "--logN", str(row["logN"]),
                "--batch", str(row["batch"]), "--warmup", "10", "--repeat", "100",
                "--verify", "--verify-batches", "1", "--csv",
            ]
            commands.append({
                "id": row["id"], "family": "fft", "status_policy": "pending",
                "comparison_reference": "cufft-available",
                "mapping_policy": "auto-select-requires-observed-mapping",
                "run": _shell(args) + f" > {result}/comparison/anchors/{row['id']}.csv 2>&1",
            })
    return commands


def _phase_commands(output: Path, suite_path: Path, gap: dict[str, Any],
                    external_suite_path: Path,
                    external_inventory: dict[str, Any]) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    result = _relative(output)
    current_library = "build-e13-cufftdx/libcuntt.so"
    external_commands, external_outputs = _external_phase(
        output, external_suite_path, external_inventory)
    commands: dict[str, list[dict[str, Any]]] = {
        "prerequisites": [
            {
                "id": "build_current_campaign_targets",
                "purpose": "build current Cufftdx/SM80 targets; this is a build step, not a GPU run",
                "run": "cmake --build build-e13-cufftdx --target cubutterfly_c_api_tests cubutterfly_tests cuntt_tests cubutterfly_xd_fft_layout_tests cubutterfly_rank3_permute_tests cubutterfly_public_planner_tests cubutterfly_mapping_tests cubutterfly_plan_bench -j2",
            },
        ],
        "correctness": [
            {
                "id": "ctest_current_operator_correctness",
                "purpose": "same-source C++ operator, C-ABI, xD layout, rank and mapping checks",
                "run": "CUDA_VISIBLE_DEVICES=${GPU_UUID} ctest --test-dir build-e13-cufftdx --output-on-failure -R 'cubutterfly_correctness|cuntt_correctness|cubutterfly_c_api|xd_fft_.*_correctness|rank3_permute_correctness|public_planner_consistency|mapping_record_roundtrip'",
            },
            {
                "id": "public_cabi_xd_correctness",
                "purpose": "public C-ABI xD C2C/R2C/C2R, zeta, NTT, structured and mapping replay",
                "run": f"CUDA_VISIBLE_DEVICES=${{GPU_UUID}} python3 tests/check_xd.py --library {current_library} --output {result}/correctness/check_xd.json",
            },
            {
                "id": "gap_public_cabi_small_execute",
                "purpose": "execute every non-large finite gap row through the public C ABI; large rows are isolated below",
                "run": f"CUDA_VISIBLE_DEVICES=${{GPU_UUID}} python3 scripts/check_operator_coverage_gaps.py --manifest {result}/gap_matrix.json --library {current_library} --execute --output {result}/correctness/gap_public_cabi_small.json",
            },
            {
                "id": "real_fft_correctness_and_reference",
                "purpose": "fresh compact-real correctness against PyTorch/cuFFT reference",
                "run": f"CUDA_VISIBLE_DEVICES=${{GPU_UUID}} python3 scripts/benchmark_xd_real.py --library {current_library} --rank3 --warmup 0 --trials 1 --output {result}/correctness/real_fft.json",
            },
            {
                "id": "negative_low_precision_real_contract",
                "purpose": "retain the four explicit FP16/BF16 R2C/C2R unsupported rows as negative tests",
                "run": f"CUDA_VISIBLE_DEVICES=${{GPU_UUID}} python3 scripts/check_operator_coverage_gaps.py --manifest {result}/gap_matrix.json --library {current_library} --section unsupported_real_low_precision --execute --output {result}/correctness/real_low_precision_negative.json",
                "expected_status": "unsupported",
            },
        ],
        "large_memory": [
            {
                "id": "gap_public_cabi_large_execute",
                "purpose": "execute only large_n_batch public C-ABI rows with a 64 GiB IO/workspace gate on the target A100 80GB",
                "failure_isolated": True,
                "run": f"CUDA_VISIBLE_DEVICES=${{GPU_UUID}} python3 scripts/check_operator_coverage_gaps.py --manifest {result}/gap_matrix.json --library {current_library} --section large_n_batch --include-large --memory-limit-bytes 68719476736 --execute --output {result}/correctness/gap_public_cabi_large.json",
            },
        ],
        "profile_gate": [
            {
                "id": "xd_profile_coverage_selected_resident",
                "purpose": "fresh selected/resident xD coverage; measure selector and verify forward rows",
                "run": f"CUDA_VISIBLE_DEVICES=${{GPU_UUID}} CUBUTTERFLY_COMPILE_MODE=research python3 scripts/benchmark_all_operators_xd.py --bench build-e13-cufftdx/cubutterfly_plan_bench --library {current_library} --output {result}/profile/xd_coverage --profile coverage --policy measure --compare-cufft --verify --gpu-uuid ${{GPU_UUID}} --compile-mode research --warmup 10 --repeat 50 --trials 3 --modulus 257 --matrix 1,0.25,-0.5,1 --keep-logs",
            },
            {
                "id": "xd_profile_scale_selected",
                "purpose": "fresh larger N/xD scaling rows with selected mapping only",
                "run": f"CUDA_VISIBLE_DEVICES=${{GPU_UUID}} CUBUTTERFLY_COMPILE_MODE=research python3 scripts/benchmark_all_operators_xd.py --bench build-e13-cufftdx/cubutterfly_plan_bench --library {current_library} --output {result}/profile/xd_scale --profile scale --policy measure --compare-cufft --verify --gpu-uuid ${{GPU_UUID}} --compile-mode research --warmup 10 --repeat 50 --trials 3 --modulus 257 --matrix 1,0.25,-0.5,1 --skip-resident-control --keep-logs",
            },
            {
                "id": "xd_profile_coverage_cabi_correctness",
                "purpose": "independent public C-ABI value checks for coverage rows; selector timing and mapping remain separate evidence",
                "run": f"CUDA_VISIBLE_DEVICES=${{GPU_UUID}} CUBUTTERFLY_COMPILE_MODE=research python3 scripts/check_gpu_operator_matrix.py --library {current_library} --profile coverage --include-resident --input-run {result}/profile/xd_coverage/rows.json --compile-mode research --modulus 257 --matrix 1,0.25,-0.5,1 --output {result}/profile/xd_coverage_correctness.json",
            },
            {
                "id": "xd_profile_scale_cabi_correctness",
                "purpose": "independent public C-ABI value checks for scale rows; selector timing and mapping remain separate evidence",
                "run": f"CUDA_VISIBLE_DEVICES=${{GPU_UUID}} CUBUTTERFLY_COMPILE_MODE=research python3 scripts/check_gpu_operator_matrix.py --library {current_library} --profile scale --input-run {result}/profile/xd_scale/rows.json --compile-mode research --modulus 257 --matrix 1,0.25,-0.5,1 --output {result}/profile/xd_scale_correctness.json",
            },
            {
                "id": "summarize_xd_profile_coverage",
                "purpose": "strict population/configuration summary; measured is not correctness",
                "run": f"python3 scripts/summarize_all_operators_xd.py {result}/profile/xd_coverage --output {result}/profile/xd_coverage_report.md",
            },
            {
                "id": "summarize_xd_profile_scale",
                "purpose": "strict population/configuration summary; measured is not correctness",
                "run": f"python3 scripts/summarize_all_operators_xd.py {result}/profile/xd_scale --output {result}/profile/xd_scale_report.md",
            },
        ],
        "comparison": [
            {
                "id": "rank1_comprehensive_comparison",
                "purpose": "fresh rank-1 selected cases plus cuFFT/VkFFT where available",
                "run": f"CUDA_VISIBLE_DEVICES=${{GPU_UUID}} CUBUTTERFLY_COMPILE_MODE=research python3 scripts/run_comprehensive_suite.py --manifest {_relative(suite_path)} --mode full --output {result}/comparison/rank1_raw.csv --resume --require-exclusive-gpu",
            },
            {
                "id": "summarize_rank1_comparison",
                "purpose": "performance table from fresh rank-1 rows only",
                "run": f"python3 scripts/summarize_comprehensive_suite.py {result}/comparison/rank1_raw.csv --manifest {_relative(suite_path)} --output {result}/comparison/rank1_summary.csv --markdown {result}/comparison/rank1_summary.md",
            },
            {
                "id": "xd_fft_three_way_comparison",
                "purpose": "fresh xD FFT selected/auto/composite versus torch/cuFFT",
                "run": f"CUDA_VISIBLE_DEVICES=${{GPU_UUID}} CUBUTTERFLY_COMPILE_MODE=research python3 scripts/benchmark_xd.py --library {current_library} --warmup 10 --replays 7 --capture-calls 20 --output {result}/comparison/xd_fft_three_way.json",
            },
            {
                "id": "xd_real_three_way_comparison",
                "purpose": "fresh compact-real native/legacy/cuFFT comparison",
                "run": f"CUDA_VISIBLE_DEVICES=${{GPU_UUID}} CUBUTTERFLY_COMPILE_MODE=research python3 scripts/benchmark_xd_real.py --library {current_library} --rank3 --warmup 10 --trials 7 --capture-calls 20 --output {result}/comparison/xd_real_three_way.json",
            },
            *external_commands,
            {
                "id": "summarize_recalibrated_campaign",
                "purpose": "final current-campaign aggregate; reads only fresh rank1, strict xD and external summaries",
                "run": f"python3 scripts/summarize_recalibrated_campaign.py --campaign {result}",
            },
        ]
    }
    final_summary = commands["comparison"].pop()
    commands["comparison"].extend(_direct_anchor_commands(gap, output))
    commands["comparison"].append(final_summary)
    return commands, external_outputs


def prepare(output: Path, *, reviewed: bool = False, gpu_uuid: str | None = None,
            hardware_profile: Path | None = None) -> dict[str, Any]:
    output = output.resolve()
    if (output / "manifest.json").exists():
        raise ValueError("refusing to overwrite a frozen campaign; use a fresh output directory")
    if reviewed and (not gpu_uuid or not gpu_uuid.startswith("GPU-") or hardware_profile is None):
        raise ValueError("reviewed campaign needs an exact GPU UUID and a hardware profile")
    gap = _load_gap()
    gap_rows = [_gap_case(row) for row in gap["rows"]]
    suite = _make_comparison_suite()
    search_workloads = _rank1_search_workloads(suite)
    suite_path = output / "comparison_suite.json"
    search_path = output / "rank1_search_workloads.json"
    external_suite_path = output / "external_suite.json"
    gap_path = output / "gap_matrix.json"
    manifest_path = output / "manifest.json"
    external_inventory = _external_tool_inventory()
    external_suite = _load_external_suite()
    external_suite.update({
        "campaign_id": "recalibrated_campaign_20261006",
        "source_manifest": _relative(EXTERNAL_SUITE_INPUT),
        "hardware": TARGET_HARDWARE_LABEL,
        "target_hardware": {
            "uuid": TARGET_GPU_UUID,
            "label": TARGET_HARDWARE_LABEL,
            "confirmation": "runner must record the runtime GPU identity before accepting rows",
        },
        "source_binary_overrides": {
            "butterfly": "build-e13-cufftdx/cubutterfly_bench",
            "ntt": "build-e13-cufftdx/cuntt_bench",
        },
        "timing_rows_imported": False,
        "external_catalog": _relative(EXTERNAL_CATALOG),
        "availability": external_inventory,
    })
    output.mkdir(parents=True, exist_ok=True)
    external_suite_path.parent.mkdir(parents=True, exist_ok=True)
    gap_path.write_text(json.dumps(gap, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    suite_path.write_text(json.dumps(suite, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    search_path.write_text(json.dumps(search_workloads, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    external_suite_path.write_text(json.dumps(external_suite, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    phases, external_outputs = _phase_commands(
        output, suite_path, gap, external_suite_path, external_inventory)

    profile = {
        "coverage": _profile_definition("coverage", skip_resident=False),
        "scale": _profile_definition("scale", skip_resident=True),
    }
    status_counts: dict[str, int] = {}
    for row in gap_rows:
        status_counts[row["initial_status"]] = status_counts.get(row["initial_status"], 0) + 1
    manifest: dict[str, Any] = {
        "schema": "cubutterfly-recalibrated-campaign-v1",
        "campaign_id": "recalibrated_campaign_20261006",
        "created_by": _relative(Path(__file__).resolve()),
        "execution": {"prepared_only": True, "gpu_execution": False, "timing_rows_imported": False},
        "scope": {
            "target": "current build-e13-cufftdx, Cufftdx ON, SM80",
            "families": list(FAMILIES),
            "alias_policy": "xor-zeta is a compatibility alias of subset-zeta; alias rows inherit subset evidence and are never counted as a second kernel",
            "authorized_finite_inputs": {
                "gap_manifest": _relative(GAP_INPUT),
                "gap_rows": len(gap_rows),
                "profile_axes": "existing all-operator coverage_v4 and scale_v4 axes only; no old timing rows are copied",
                "rank1_points": [[8, 1], [12, 16]],
                "historical_case_semantics": {"source": _relative(HISTORICAL_CONFIG), "cases": 37, "timings_read": False},
                "external_workload_semantics": {"source": _relative(EXTERNAL_SUITE_INPUT), "timings_read": False},
                "large_anchors": "exact rows copied from gap_matrix.json; fixed moduli retained for NTT",
                "input_fingerprints": _matrix_input_freeze(),
            },
        },
        "source_freeze": _source_freeze(),
        "builds": _build_identity(),
        "gap_population": {
            "row_count": len(gap_rows),
            "by_initial_status": status_counts,
            "by_section": {section: sum(row["section"] == section for row in gap_rows)
                           for section in sorted({row["section"] for row in gap_rows})},
            "rows": gap_rows,
        },
        "profiles": profile,
        "comparison_suite": {
            "path": _relative(suite_path),
            "cases": len(suite["cases"]),
            "cohorts": suite["cohorts"],
            "historical_case_semantics": suite["historical_case_semantics"],
            "search_workloads": {
                "path": _relative(search_path),
                "cases": len(search_workloads["cases"]),
                "calibration_status": search_workloads["calibration_status"],
            },
            "families": sorted({case["operator"] for case in suite["cases"]}),
            "fresh_output_required": True,
            "external_reference_status": suite["external_evidence"],
        },
        "external_comparison": {
            "path": _relative(external_suite_path),
            "catalog": _relative(EXTERNAL_CATALOG),
            "source_workload_matrix": _relative(EXTERNAL_SUITE_INPUT),
            "workloads": len(external_suite["workloads"]),
            "cases": sum(len(item["batches"]) * len(item["implementations"])
                         for item in external_suite["workloads"]),
            "timing_rows_imported": False,
            "availability": external_inventory,
            "tasks": external_outputs["task_status"],
            "fresh_artifacts": external_outputs["artifacts"],
        },
        "rank1_search": {
            "path": _relative(search_path),
            "cases": len(search_workloads["cases"]),
            "calibration_status": search_workloads["calibration_status"],
            "registry_matching": search_workloads["selection"]["registry_matching"],
            "search_executed": False,
        },
        "status_taxonomy": {
            "pending": "requested row has no result yet",
            "passed": "same-source correctness passed and required timing/mapping fields are present",
            "failed": "execution or correctness failed; do not merge into pass",
            "unsupported": "current implementation explicitly rejects the row or required negative test passed",
            "missing-external-reference": "correctness may be measured, but no same-build external baseline exists",
            "memory-limit": "runtime reports allocation/resource exhaustion; keep separate from unsupported and failed",
            "unverified": "timing exists without a correctness result; never promote to pass",
        },
        "mapping_policy": {
            "selected": "benchmark_all_operators_xd.py --policy measure; require the reported selected algorithm and preserve runner stdout/metadata; do not infer an unmeasured mapping_json",
            "rank1_auto_select": "run_comprehensive_suite.py --auto-select; record selected_implementation, selection_confidence and selection_reason; unmeasured defaults are not a calibrated win",
            "resident_control": "resident-t128 is an explicit control only when the runner reports resident-t128",
        },
        "resource_policy": {
            "preflight_required": "record target UUID, model, total/free memory and exclusive-process check before GPU phases",
            "memory_failure_markers": ["out of memory", "memory allocation", "cudaErrorMemoryAllocation", "CUBLAS_STATUS_ALLOC_FAILED"],
            "large_rows": [row["id"] for row in gap_rows if row["section"] == "large_n_batch"],
            "large_row_policy": "classify allocation failures as memory-limit; do not call them unsupported or pass",
        },
        "phases": phases,
        "fresh_outputs": {
            "root": _relative(output),
            "forbidden_inputs": [
                "results/comprehensive_current_20261006_cufftdx/*timing*",
                "results/all_operators_xd_20261006_gpu1_coverage_v4/rows.json",
                "results/all_operators_xd_20261006_gpu1_scale_v4/rows.json",
            ],
            "required_artifacts": [
                "correctness/check_xd.json", "correctness/real_fft.json",
                "correctness/gap_public_cabi_small.json", "correctness/gap_public_cabi_large.json",
                "profile/xd_coverage/rows.json", "profile/xd_scale/rows.json",
                "profile/xd_coverage_correctness.json", "profile/xd_scale_correctness.json",
                "rank1_search_workloads.json",
                "comparison/rank1_raw.csv", "comparison/rank1_summary.md",
                *external_outputs["artifacts"],
            ],
        },
        "files": {
            "gap_matrix.json": {"sha256": _sha256_file(gap_path), "bytes": gap_path.stat().st_size},
            "comparison_suite.json": {"sha256": _sha256_file(suite_path), "bytes": suite_path.stat().st_size},
            "rank1_search_workloads.json": {"sha256": _sha256_file(search_path), "bytes": search_path.stat().st_size},
            "external_suite.json": {
                "sha256": _sha256_file(external_suite_path), "bytes": external_suite_path.stat().st_size,
            },
        },
    }
    manifest["campaign_id"] = output.name
    if reviewed:
        if not gpu_uuid or hardware_profile is None:
            raise ValueError("reviewed campaign needs a GPU UUID and a hardware profile")
        if not gpu_uuid.startswith("GPU-"):
            raise ValueError("use the exact GPU UUID, not a local device index")
        profile = json.loads(hardware_profile.read_text())
        hardware = {key: profile[key] for key in
                    ("device", "compute_capability", "global_memory_bytes")}
        if any(value is None or value == "" for value in hardware.values()):
            raise ValueError("hardware profile is incomplete")
        # Build before freezing and calibrating. A benchmark queue must not
        # silently rebuild the library underneath its confirmed measurements.
        manifest["phases"]["prerequisites"] = []
        for step in manifest["phases"]["profile_gate"]:
            if "scripts/check_gpu_operator_matrix.py" in step["run"]:
                step["run"] += " --require-source-mapping"
                step["purpose"] = "independent C-ABI value check replaying the exact timed mapping"
            if "scripts/summarize_all_operators_xd.py" in step["run"]:
                step["run"] += " --require-verified-complete"
                step["purpose"] = "require complete declared coverage with verified timed mappings"
        manifest["measurement_contract"] = {
            "schema": "cubutterfly-reviewed-comparison-v1",
            "revision": "0.9.1-review.1",
            "gpu_uuid": gpu_uuid,
            "hardware": hardware,
            "compile_mode": "research",
            "build_policy": "build-before-prepare-and-calibration",
            "gate_policy": "stop-on-correctness-or-profile-failure",
            "resume_policy": "reuse completed steps; quarantine and repeat incomplete steps",
            "scope": "finite declared populations; aliases and overlapping cohorts are not additional independent workloads",
        }
        from run_recalibrated_campaign import execution_environment, parse_command
        manifest["measurement_contract"]["execution_environment"] = execution_environment()
        for steps in manifest["phases"].values():
            for step in steps:
                argv, _, redirect = parse_command(step["run"], gpu_uuid, ROOT)
                outputs = [redirect] if redirect else []
                if "--output" in argv:
                    output_arg = Path(argv[argv.index("--output") + 1])
                    output_arg = output_arg if output_arg.is_absolute() else ROOT / output_arg
                    outputs.extend([output_arg, output_arg.with_suffix(".checkpoint.json"),
                                    output_arg.with_name(output_arg.name + ".failures.json")])
                for option in ("--markdown",):
                    if option in argv:
                        path = Path(argv[argv.index(option) + 1])
                        outputs.append(path if path.is_absolute() else ROOT / path)
                script = next((index for index, value in enumerate(argv)
                               if value.endswith("/summarize_all_operators_xd.py")), None)
                if script is not None:
                    directory = Path(argv[script + 1])
                    directory = directory if directory.is_absolute() else ROOT / directory
                    outputs.append(directory / "coverage_summary.json")
                step["outputs"] = [str(path.resolve()) for path in dict.fromkeys(outputs)]
        # These external binaries/modules are outside the source tree, but
        # changing them also changes the comparison's measurement identity.
        baseline_artifacts = manifest["builds"]["baseline"]["artifacts"]
        for name, fields in (("dao-fht", ("python", "module", "native_module")),
                             ("gpu-ntt", ("binary",))):
            inventory = external_inventory[name]
            if inventory.get("available"):
                for field in fields:
                    baseline_artifacts[f"{name}_{field}"] = _artifact(Path(inventory[field]), required=True)
        # Metadata names the intended hardware only. The runner verifies it.
        external_suite["campaign_id"] = output.name
        external_suite["target_hardware"] = {
            "uuid": gpu_uuid, **hardware,
            "confirmation": "pending runtime identity verification",
        }
        external_suite["hardware"] = f"{hardware['device']} / {hardware['global_memory_bytes']} bytes / target {gpu_uuid}"
        external_suite_path.write_text(json.dumps(external_suite, indent=2, sort_keys=True) + "\n")
        manifest["files"]["external_suite.json"] = {
            "sha256": _sha256_file(external_suite_path), "bytes": external_suite_path.stat().st_size}
    body = dict(manifest)
    manifest["manifest_sha256"] = _sha256_bytes(_canonical(body))
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--gpu-uuid", default=TARGET_GPU_UUID, required=not TARGET_GPU_UUID)
    args = parser.parse_args(argv)
    globals()["TARGET_GPU_UUID"] = args.gpu_uuid
    globals()["TARGET_HARDWARE_LABEL"] = f"target UUID {args.gpu_uuid} (requires runtime confirmation)"
    manifest = prepare(args.output)
    print(json.dumps({
        "output": _relative(args.output.resolve()),
        "manifest_sha256": manifest["manifest_sha256"],
        "gap_rows": manifest["gap_population"]["row_count"],
        "comparison_cases": manifest["comparison_suite"]["cases"],
        "profiles": {name: value["semantic_rows_including_alias"] for name, value in manifest["profiles"].items()},
        "current_missing_binaries": [name for name, item in manifest["builds"]["current"]["artifacts"].items() if not item["exists"]],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
