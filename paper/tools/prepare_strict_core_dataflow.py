#!/usr/bin/env python3
"""Prepare the strict 2x2 FFT core/dataflow ablation without using a GPU.

The factorial is deliberately explicit.  Each case keeps one factor shape and
one batch fixed across all four cells.  The core contrast changes only
``fft_core``; the dataflow contrast changes only the factor slice/overlap
schedule.  The latter is named serial-factor versus sliced-hybrid in the
record so the experiment does not overclaim a monolithic no-hybrid baseline.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "paper" / "tools"))

from prepare_experiments import canonical, module_manifest, write
import run_controlled_experiment as controlled


SCHEMA = "cubutterfly-controlled-experiment-v1"
CONFIG_SCHEMA = "cubutterfly-strict-core-dataflow-config-v1"
CORE_VARIANTS = (("native", "register-tile"), ("imported", "cufftdx-block"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _positive_int(value, field: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _partition(value, log_n: int, field: str) -> list[int]:
    if not isinstance(value, list) or not value or any(type(item) is not int or item <= 0 for item in value):
        raise ValueError(f"{field} must be a non-empty positive integer list")
    if sum(value) != log_n:
        raise ValueError(f"{field} must sum to logN")
    return list(value)


def _validate_config(config: dict) -> dict:
    if config.get("schema") != CONFIG_SCHEMA:
        raise ValueError(f"configuration schema must be {CONFIG_SCHEMA}")
    protocol = config.get("protocol")
    if not isinstance(protocol, dict):
        raise ValueError("configuration protocol must be an object")
    expected_protocol = {"trials", "warmup", "repeat", "verify_batches"}
    if set(protocol) != expected_protocol:
        raise ValueError("configuration protocol fields are not exact")
    _positive_int(protocol["trials"], "protocol.trials")
    if type(protocol["warmup"]) is not int or protocol["warmup"] < 0:
        raise ValueError("protocol.warmup must be a non-negative integer")
    _positive_int(protocol["repeat"], "protocol.repeat")
    if type(protocol["verify_batches"]) is not int or protocol["verify_batches"] < 0:
        raise ValueError("protocol.verify_batches must be a non-negative integer")
    cases = config.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("configuration cases must be non-empty")
    seen = set()
    for index, case in enumerate(cases):
        location = f"cases[{index}]"
        case_id = case.get("id")
        if not isinstance(case_id, str) or not case_id or case_id in seen:
            raise ValueError(f"{location}.id must be unique and non-empty")
        seen.add(case_id)
        workload = case.get("workload")
        if not isinstance(workload, dict):
            raise ValueError(f"{location}.workload must be an object")
        if workload.get("operator") != "fft" or workload.get("precision") != "fp32":
            raise ValueError(f"{location} must be an FP32 FFT case")
        log_n = _positive_int(workload.get("logN"), f"{location}.workload.logN")
        _positive_int(workload.get("batch"), f"{location}.workload.batch")
        _partition(case.get("stage_partition"), log_n, f"{location}.stage_partition")
        factors = _partition(case.get("factor_partition"), log_n, f"{location}.factor_partition")
        if any(stage > 14 for stage in factors):
            raise ValueError(f"{location}.factor_partition exceeds the generated factor template limit")
        ept = _positive_int(case.get("factor_ept"), f"{location}.factor_ept")
        columns = _positive_int(case.get("factor_columns"), f"{location}.factor_columns")
        if ept & (ept - 1) or columns & (columns - 1):
            raise ValueError(f"{location} EPT and columns must be powers of two")
        # Both cores must share the exact EPT/column shape.  NativeFactor then
        # additionally requires EPT=sqrt(factor size), which the config cases
        # satisfy for every factor.
        for factor in factors:
            if factor % 2 or ept != 1 << (factor // 2):
                raise ValueError(f"{location} is not legal for the native register-tile factor core")
            if (1 << factor) < ept or (1 << factor) // ept * columns < 32:
                raise ValueError(f"{location} has an invalid common factor launch shape")
        if columns > 1 << (log_n - factors[0]):
            raise ValueError(f"{location}.factor_columns exceeds independent transform count")
        if columns > 1 << sum(factors[:-1]):
            raise ValueError(f"{location}.factor_columns exceeds processed digit count")
        if 2 > (1 << factors[-1]) // columns:
            raise ValueError(f"{location} cannot run the two-slice hybrid cell")
    return config


def _mapping(case: dict, core: str, *, hybrid: bool) -> dict:
    workload = case["workload"]
    return {
        "schema_version": 1,
        "kind": "butterfly",
        "backend": "factor-streamed",
        "fft_core": core,
        "compute_unit": "auto",
        "local_exchange": "shared",
        "shared_layout": "writer-aligned",
        "cross_twiddle": "recurrence",
        "direct_boundary": "strided",
        "stage_partition": list(case["stage_partition"]),
        "factor_partition": list(case["factor_partition"]),
        "factor_ept": case["factor_ept"],
        "factor_columns": case["factor_columns"],
        "data_tiles_per_cta": 1,
        "prefetch_depth": 0,
        # Keep the compiled range module identical in both dataflow cells;
        # only whether the slices are pipelined is an experimental axis.
        "factor_slices": 2,
        "factor_overlap": bool(hybrid),
    }


def study(config: dict) -> dict:
    """Expand the source config into the controlled-study runner schema."""
    _validate_config(config)
    cases = []
    for source in config["cases"]:
        variants = []
        for core_name, core in CORE_VARIANTS:
            for flow_name, hybrid in (("serial-factor", False), ("sliced-hybrid", True)):
                variants.append({
                    "id": f"{core_name}-{flow_name}",
                    "mapping": _mapping(source, core, hybrid=hybrid),
                })
        comparisons = [
            {"baseline": "native-serial-factor", "treatment": "imported-serial-factor",
             "changed_axes": ["fft_core"]},
            {"baseline": "native-sliced-hybrid", "treatment": "imported-sliced-hybrid",
             "changed_axes": ["fft_core"]},
            {"baseline": "native-serial-factor", "treatment": "native-sliced-hybrid",
             "changed_axes": ["factor_overlap"]},
            {"baseline": "imported-serial-factor", "treatment": "imported-sliced-hybrid",
             "changed_axes": ["factor_overlap"]},
        ]
        cases.append({
            "id": source["id"],
            "experiment": "E2/E4-strict-core-dataflow",
            "workload": dict(source["workload"]),
            "variants": variants,
            "comparisons": comparisons,
        })
    return {
        "schema": SCHEMA,
        "protocol": dict(config["protocol"]),
        "scope": config["scope"],
        "ablation": {
            "core_axis": "fft_core: register-tile versus cufftdx-block",
            "dataflow_axis": "factor_slices/factor_overlap: serial-factor versus sliced-hybrid",
            "common_axes": ["factor_partition", "factor_ept", "factor_columns", "data_tiles_per_cta",
                            "prefetch_depth", "stage_partition", "shared_layout", "cross_twiddle",
                            "direct_boundary", "batch", "precision", "placement", "direction"],
            "limitations": [
                "The non-hybrid control is serial execution of two fixed factor slices, not a monolithic whole-FFT kernel.",
                "A monolithic baseline is not available for the same factorized core contract without a separate lowering.",
                "Compilation, plan construction, and correctness preflight are outside timed kernel trials."
            ],
        },
        "cases": cases,
    }


def _command_text(values) -> str:
    return " ".join(shlex.quote(str(value)) for value in values)


def prepare(config_path: Path, target_path: Path, output_dir: Path) -> dict:
    config_path = config_path.expanduser().resolve()
    target_path = target_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    target = json.loads(target_path.read_text(encoding="utf-8"))
    if target.get("schema") != "cubutterfly-paper-target-v1":
        raise ValueError("unsupported target schema")
    value = study(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    study_path = output_dir / "strict_core_dataflow.json"
    prepared_path = output_dir / "prepared.json"
    modules_path = output_dir / "precompile" / "modules.json"
    for path, document in ((study_path, value),):
        if path.exists() and canonical(json.loads(path.read_text(encoding="utf-8"))) != canonical(document):
            raise ValueError(f"{path} changed; select a new output directory")
        write(path, document)
    # This calls only structural validation and command lowering.  It does not
    # inspect CUDA, execute a benchmark, or claim that a mapping is runnable.
    prepared = controlled.prepare_document(value, study_path, (ROOT / target["build_dir"]).resolve())
    if prepared_path.exists():
        existing = json.loads(prepared_path.read_text(encoding="utf-8"))
        controlled._prepared_compatible(existing, prepared)
    else:
        controlled._atomic_write(prepared_path, prepared)
    modules = module_manifest({"strict_core_dataflow": value}, target)
    if modules_path.exists() and canonical(json.loads(modules_path.read_text(encoding="utf-8"))) != canonical(modules):
        raise ValueError(f"{modules_path} changed; select a new output directory")
    write(modules_path, modules)
    py = target["python"]
    build = (ROOT / target["build_dir"]).resolve()
    journal = output_dir / "journal.json"
    precompile_dir = output_dir / "precompile"
    run_command = [py, ROOT / "scripts" / "run_controlled_experiment.py", "--mode", "run",
                   "--study", study_path, "--build-dir", build, "--output", journal,
                   "--gpu-uuid", target["gpu_uuid"]]
    compile_command = [py, ROOT / "scripts" / "precompile_research.py", "--mode", "compile",
                       "--output-dir", precompile_dir, "--jobs", 2, "--root", ROOT,
                       "--mathdx", ROOT / "external" / "mathdx" / "nvidia" / "mathdx"]
    runbook = [
        "# Strict 2x2 core/dataflow ablation",
        "",
        "This directory is CPU-prepared only. Preparation did not query a GPU or run a benchmark.",
        "The four cells are native/imported x serial-factor/sliced-hybrid for each FFT case.",
        "",
        "## Protocol",
        "",
        f"- trials: {value['protocol']['trials']}; warmup: {value['protocol']['warmup']}; repeat: {value['protocol']['repeat']}",
        "- Pair trial order is reversed by the controlled runner; compilation and preflight are excluded from kernel timing.",
        "- Every cell must pass full-batch correctness and retain the resolved mapping/runtime fingerprint across trials.",
        "- serial-factor uses factor_slices=2 and factor_overlap=false; sliced-hybrid uses factor_slices=2 and factor_overlap=true.",
        "",
        "## Precompile",
        "",
        "```bash",
        _command_text(compile_command),
        "```",
        "",
        "## Measure",
        "",
        "```bash",
        _command_text(run_command),
        "```",
        "",
        "## Interpretation",
        "",
        "The serial-factor row is the honest non-overlapped fixed-group control available under the factor-streamed lowering. It is not a monolithic whole-FFT implementation.",
        "A comparison is publishable only after all four cells of a case are accepted; unavailable or rejected cells remain gaps.",
        "",
    ]
    (output_dir / "RUNBOOK.md").write_text("\n".join(runbook), encoding="utf-8")
    manifest = {
        "schema": "cubutterfly-strict-core-dataflow-preparation-v1",
        "status": "prepared",
        "gpu_queried": False,
        "executable_claim": False,
        "config": {"path": str(config_path), "sha256": _sha256(config_path)},
        "target": {"path": str(target_path), "sha256": _sha256(target_path), "id": target["id"],
                   "gpu_uuid": target["gpu_uuid"], "compute_capability": target["compute_capability"]},
        "study": {"path": str(study_path), "sha256": _sha256(study_path), "cases": len(value["cases"]),
                  "variants": sum(len(case["variants"]) for case in value["cases"]),
                  "comparisons": sum(len(case["comparisons"]) for case in value["cases"])},
        "prepared": {"path": str(prepared_path), "sha256": _sha256(prepared_path)},
        "precompile": {"path": str(modules_path), "sha256": _sha256(modules_path),
                        "module_count": modules["module_count"]},
        "commands": {"precompile": [str(value) for value in compile_command],
                     "measure": [str(value) for value in run_command]},
        "limitations": value["ablation"]["limitations"],
    }
    write(output_dir / "manifest.json", manifest)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "paper/experiments/strict_core_dataflow.json")
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    result = prepare(args.config, args.target, args.output_dir)
    print(json.dumps({"status": result["status"], "gpu_queried": result["gpu_queried"],
                      "cases": result["study"]["cases"], "variants": result["study"]["variants"],
                      "comparisons": result["study"]["comparisons"],
                      "modules": result["precompile"]["module_count"],
                      "output": str(Path(args.output_dir).resolve())}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
