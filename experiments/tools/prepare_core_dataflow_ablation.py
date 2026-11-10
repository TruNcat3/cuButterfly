#!/usr/bin/env python3
"""Prepare the standalone radix-16 core/dataflow ablation on the CPU.

The preparation step expands the three representative FFT sizes into latency
and throughput invocations, records the target identity, and emits a direct
nvcc command.  It never queries CUDA, compiles, allocates a device buffer, or
claims that a cell is executable.  GPU execution is intentionally a later,
exclusive-window step.
"""
from __future__ import print_function

import argparse
import hashlib
import json
from pathlib import Path
import shlex


ROOT = Path(__file__).resolve().parents[2]
CONFIG_SCHEMA = "cubutterfly-core-dataflow-ablation-config-v1"
TARGET_SCHEMA = "cubutterfly-paper-target-v1"
GENERATOR = "core-dataflow-ablation-preparation-v1"


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False)


def digest(path):
    path = Path(path)
    value = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_json(path):
    path = Path(path).expanduser().resolve()
    return json.loads(path.read_text(encoding="utf-8")), {
        "path": str(path), "sha256": digest(path)
    }


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def command_text(argv):
    return " ".join(shlex.quote(str(value)) for value in argv)


def positive_int(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(name + " must be a positive integer")
    return value


def validate(config, target):
    if config.get("schema") != CONFIG_SCHEMA:
        raise ValueError("unexpected core/dataflow config schema")
    if target.get("schema") != TARGET_SCHEMA:
        raise ValueError("unexpected paper target schema")
    protocol = config.get("protocol")
    if not isinstance(protocol, dict):
        raise ValueError("protocol must be an object")
    for key in ("trials", "warmup", "repeat"):
        positive_int(protocol.get(key), "protocol." + key)
    cases = config.get("cases")
    if not isinstance(cases, list) or len(cases) != 3:
        raise ValueError("the controlled study must contain exactly three FFT cases")
    expected_logs = (8, 12, 16)
    seen = set()
    for index, case in enumerate(cases):
        prefix = "cases[{}]".format(index)
        if not isinstance(case, dict) or not isinstance(case.get("id"), str) or not case["id"]:
            raise ValueError(prefix + ".id must be non-empty")
        if case["id"] in seen:
            raise ValueError("duplicate case id: " + case["id"])
        seen.add(case["id"])
        workload = case.get("workload")
        if not isinstance(workload, dict):
            raise ValueError(prefix + ".workload must be an object")
        if workload.get("operator") != "fft" or workload.get("precision") != "fp32":
            raise ValueError(prefix + " must be an FP32 FFT")
        log_n = positive_int(workload.get("logN"), prefix + ".workload.logN")
        if log_n != expected_logs[index] or log_n % 4:
            raise ValueError("cases must cover radix-16 logN values 8, 12, and 16 in order")
        for key in ("latency_batch", "throughput_batch"):
            batch = positive_int(workload.get(key), prefix + ".workload." + key)
            if key == "throughput_batch" and batch * (1 << log_n) < (1 << 20):
                raise ValueError(prefix + " throughput batch does not expose enough parallel work")
        if case.get("stage_count") != log_n // 4:
            raise ValueError(prefix + ".stage_count must equal logN/4")
        if case.get("resident_prefix_stages") != 2:
            raise ValueError(prefix + ".resident_prefix_stages must be two")
        if case.get("resident_tile_points") != 256:
            raise ValueError(prefix + ".resident_tile_points must be 256")
    if not target.get("gpu_uuid") or not target.get("compute_capability"):
        raise ValueError("target must identify a GPU UUID and compute capability")


def compile_command(target, output_dir):
    capability = str(target["compute_capability"])
    try:
        major, minor = capability.split(".", 1)
        sm = int(major) * 10 + int(minor[0])
    except (ValueError, IndexError):
        raise ValueError("target compute_capability must look like 8.0")
    mathdx = ROOT / "external" / "mathdx" / "nvidia" / "mathdx"
    binary = Path(output_dir).resolve() / "cubutterfly_core_dataflow_ablation"
    return [
        "/usr/local/cuda/bin/nvcc", "-std=c++17", "-O3", "--expt-relaxed-constexpr",
        "-arch=sm_{}".format(sm), "-DCUBUTTERFLY_CUFFTDX_SM={}".format(sm * 10),
        "-I" + str(ROOT / "include"), "-I" + str(ROOT / "src"),
        "-I" + str(mathdx / "include"), "-I" + str(mathdx / "external" / "cutlass" / "include"),
        str(ROOT / "apps" / "cubutterfly_core_dataflow_ablation.cu"), "-lcufft",
        "-o", str(binary)
    ]


def measure_commands(config, target, output_dir):
    binary = Path(output_dir).resolve() / "cubutterfly_core_dataflow_ablation"
    telemetry = ROOT / "experiments" / "tools" / "run_with_gpu_telemetry.py"
    commands = []
    for case in config["cases"]:
        workload = case["workload"]
        for label, batch_key in (("latency", "latency_batch"), ("throughput", "throughput_batch")):
            commands.append({
                "case": case["id"],
                "cohort": label,
                "argv": [
                    str(target["python"]), str(telemetry), "--gpu-uuid", target["gpu_uuid"],
                    "--stop-on-thermal-throttle", "--output-prefix",
                    str(Path(output_dir).resolve() / "journal" / (case["id"] + "-" + label)), "--",
                    str(binary), "--logN", str(workload["logN"]), "--batch", str(workload[batch_key]),
                    "--core", "both", "--organization", "both",
                    "--trials", str(config["protocol"]["trials"]),
                    "--warmup", str(config["protocol"]["warmup"]),
                    "--repeat", str(config["protocol"]["repeat"]), "--csv"
                ],
                "env": {"CUDA_VISIBLE_DEVICES": target["gpu_uuid"]},
                "executed": False,
                "timing_scope": "CUDA events around only repeated enqueue/launch sequence; compile, plan, warmup, and correctness are outside timing"
            })
    return commands


def prepare(config_path, target_path, output_dir, root=ROOT):
    config_path = Path(config_path).expanduser().resolve()
    target_path = Path(target_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    config, config_identity = read_json(config_path)
    target, target_identity = read_json(target_path)
    validate(config, target)
    app_path = Path(root) / "apps" / "cubutterfly_core_dataflow_ablation.cu"
    if not app_path.is_file():
        raise ValueError("standalone ablation source is missing: {}".format(app_path))
    build = compile_command(target, output_dir)
    commands = measure_commands(config, target, output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": "cubutterfly-core-dataflow-ablation-preparation-v1",
        "generator": GENERATOR,
        "status": "prepared-not-executed",
        "gpu_probe_performed": False,
        "executable_claim": False,
        "config": config_identity,
        "target": dict(target_identity, id=target["id"], gpu_uuid=target["gpu_uuid"],
                        compute_capability=target["compute_capability"]),
        "source": {"path": str(app_path.resolve()), "sha256": digest(app_path)},
        "study": {
            "cases": len(config["cases"]),
            "cohorts": 2,
            "cores": ["native", "cufftdx-thread"],
            "organizations": ["stagewise-hbm", "resident-prefix-2"],
            "cells_per_case_cohort": 4,
            "cells_total": len(config["cases"]) * 2 * 4
        },
        "compile": {"argv": [str(value) for value in build], "executed": False},
        "measure": {"commands": commands, "executed": False},
        "limitations": [
            "The resident treatment retains exactly the first two radix-16 stages per 256-value CTA tile; later stages use HBM boundaries.",
            "The standalone study is a controlled mechanism ablation and is not a production planner or a replacement for the public lowering.",
            "GPU runtime feasibility, correctness, and performance remain unverified until an exclusive target window is assigned."
        ]
    }
    write_json(output_dir / "manifest.json", manifest)
    write_json(output_dir / "config.json", config)
    runbook = [
        "# Strict core x dataflow FFT ablation",
        "",
        "This directory is CPU-prepared only. No GPU was queried, compiled, or executed.",
        "The four cells are native/cuFFTDx Thread x stagewise HBM/resident-prefix-2.",
        "",
        "## Controlled organization",
        "",
        "- stagewise-hbm: every radix-16 stage launches one active thread per leaf and writes the complete stage result to HBM.",
        "- resident-prefix-2: the same 16-point leaf codelet runs for 16 leaves in one CTA; stages 0 and 1 exchange through 32 KiB of shared memory, then later stages use the baseline global kernels.",
        "- Both paths use the same radix-16 digit reversal, FP32 input/output, twiddles, block size, and cuFFT correctness reference.",
        "- N=256 has two stages and is fully covered by the resident prefix; N=4096 has one later global stage; N=65536 has two later global stages.",
        "",
        "## Protocol",
        "",
        "- Compile and cuFFT plan construction are outside timing.",
        "- Full-batch correctness is checked before measurements.",
        "- Each trial measures only repeated enqueue/launch sequences with CUDA events; warmup is synchronized before the first event.",
        "- The latency cohort uses B=1. The throughput cohort uses a fixed 2^24-point workload: B=65536/4096/256 for N=256/4096/65536.",
        "- Correctness uses the existing FP32 maximum-error guard (relative max <=2e-4 and N-scaled absolute bound); relative L2 error is recorded diagnostically.",
        "- Identify the target by UUID and record model, compute capability, and memory before execution. Do not treat an index as a hardware identity.",
        "",
        "## Compile",
        "",
        "```bash",
        command_text(build),
        "```",
        "",
        "## Measure",
        "",
        "".join("```bash\nCUDA_VISIBLE_DEVICES={} {}\n```\n\n".format(
            command["env"]["CUDA_VISIBLE_DEVICES"], command_text(command["argv"])) for command in commands),
        "Only run these commands in an exclusive target window. Unavailable or incorrect cells remain gaps.",
        ""
    ]
    (output_dir / "RUNBOOK.md").write_text("\n".join(runbook), encoding="utf-8")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "experiments/workloads/core_dataflow_ablation.json")
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = prepare(args.config, args.target, args.output_dir)
    print(json.dumps({
        "status": manifest["status"], "gpu_probe_performed": manifest["gpu_probe_performed"],
        "cases": manifest["study"]["cases"], "cells": manifest["study"]["cells_total"],
        "output": str(Path(args.output_dir).resolve())
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
