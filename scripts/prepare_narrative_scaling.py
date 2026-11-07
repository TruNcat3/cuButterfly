#!/usr/bin/env python3
"""Generate and precompile the finite M1 scaling cohort without GPU access."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import time

import compile_module
from research_compile_requests import compile_request
import resident_mapping
import run_controlled_experiment as controlled
import static_hardware_profile


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/experiments/narrative_scaling_20261027.json"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resolve(path):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def make_study(config):
    if (config["operators"] != [{"operator": "fft", "precision": "fp64"},
                               {"operator": "fwht", "precision": "fp64"},
                               {"operator": "ntt", "precision": "word64"}] or
            config["shapes"] != [{"logN": 16, "stage_partition": [8, 8], "threads": 128},
                                 {"logN": 20, "stage_partition": [10, 10], "threads": 256}] or
            config["batches"] != [1, 4, 16] or config["local_widths"] != [0, 2, 4] or
            config["modulus"] != 576460756061519873 or config["sm"] != 80 or
            config["protocol"] != {"trials": 3, "warmup": 10, "repeat": 20, "verify_batches": 0}):
        raise ValueError("configuration must preserve the bounded M1 cohort")
    cases = []
    for arithmetic in config["operators"]:
        operator = arithmetic["operator"]
        for shape in config["shapes"]:
            for batch in config["batches"]:
                workload = dict(arithmetic, logN=shape["logN"], batch=batch,
                                direction="forward", normalization="none", placement="out-of-place")
                if operator == "ntt":
                    workload.update(modulus=config["modulus"], input_order="natural", output_order="natural")
                else:
                    workload["accumulation"] = "native"
                variants = []
                for width in config["local_widths"]:
                    mapping = dict(schema_version=1, kind="ntt" if operator == "ntt" else "butterfly",
                                   backend="shared-iterative", compute_unit="radix2",
                                   stage_partition=shape["stage_partition"],
                                   local_stage_partitions=resident_mapping.resident_partition_seed(
                                       shape["stage_partition"], width) if width else [],
                                   exchange_chunks=[], stage_overlap=False, batch_tile_count=1)
                    if operator == "ntt":
                        mapping.update(threads_per_block=shape["threads"], dataflow_layout="linear",
                                       modular_multiply="shoup")
                    else:
                        mapping.update(tile_threads=shape["threads"], local_stages=shape["stage_partition"][0],
                                       fft_core="scalar", local_exchange="shared", shared_layout="linear",
                                       complex_multiply="four-mul", cross_twiddle="table")
                    variants.append(dict(id=f"local{width}", mapping=mapping))
                cases.append(dict(id=f"{operator}-{arithmetic['precision']}-n{shape['logN']}-b{batch}",
                                  experiment="M1-local-publication-scaling", workload=workload,
                                  variants=variants, comparisons=[
                                      dict(baseline="local0", treatment=f"local{width}",
                                           changed_axes=["local_stage_partitions"]) for width in (2, 4)]))
    study = dict(schema=controlled.SCHEMA, protocol=config["protocol"], scope=config["scope"], cases=cases)
    controlled._validate_study(study)
    return study


def module_requests(study):
    requests = {}
    cells = []
    for case in study["cases"]:
        for variant in case["variants"]:
            point = dict(case["workload"], mapping_json=variant["mapping"])
            request = compile_request(point, policy="research")
            if request is None:
                raise ValueError("M1 requires an auditable standalone module for every cell")
            key = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
            requests.setdefault(key, request)
            cells.append(dict(case=case["id"], variant=variant["id"], request_id=key))
    if len(requests) != 18 or len(cells) != 54:
        raise ValueError("M1 must retain 54 cells and 18 batch-independent modules")
    return requests, cells


def resource_rows(study, model):
    hardware = model["hardware"]
    rows = []
    for case in study["cases"]:
        workload = case["workload"]
        width = 16 if workload["operator"] == "fft" else 8
        n, batch = 1 << workload["logN"], workload["batch"]
        for variant in case["variants"]:
            mapping = variant["mapping"]
            threads = mapping.get("tile_threads", mapping.get("threads_per_block"))
            shared = [2 * (1 << stages) * width for stages in mapping["stage_partition"]]
            # Includes data, scratch and coefficient/headroom allocations; not a measured peak.
            memory_envelope = (8 * batch + 4) * n * width
            if (threads > hardware["max_threads_per_block"] or
                    max(shared) > hardware["shared_bytes_per_block"] or
                    memory_envelope > hardware["memory_bytes"]):
                raise ValueError(f"known resource lower bound exceeds capacity: {case['id']}")
            rows.append(dict(case=case["id"], variant=variant["id"], value_bytes=width,
                             threads_per_cta=threads, dynamic_shared_bytes=shared,
                             ctas_per_group=[batch * (1 << (workload["logN"] - stages))
                                             for stages in mapping["stage_partition"]],
                             device_memory_planning_envelope_bytes=memory_envelope,
                             register_allocation="unmeasured", static_shared_bytes="unmeasured",
                             occupancy="unmeasured", status="known-static-capacity-check-passed"))
    return rows


def compile_one(key, request, config, timeout):
    started = time.monotonic()
    try:
        module = compile_module.compile_mapping(request, root=ROOT, mathdx=resolve(config["mathdx"]),
                                                nvcc=resolve(config["nvcc"]), sm=config["sm"],
                                                cache=resolve(config["cache"]), timeout=timeout)
        manifest_path = module.parent / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        if sha(module) != manifest["module_sha256"] or manifest["sm"] != config["sm"]:
            raise ValueError("compiled module identity mismatch")
        # Persist the exact generated CUDA alongside the per-cohort record.
        source = compile_module.render_shared(request, config["sm"], manifest["identity"])
        return dict(request_id=key, status="ready", path=str(module), module_sha256=sha(module),
                    identity=manifest["identity"], manifest_path=str(manifest_path),
                    manifest_sha256=sha(manifest_path), compiler=manifest["compiler"],
                    compiler_command=manifest["command"], original_compile_seconds=manifest["compile_seconds"],
                    preparation_seconds=time.monotonic() - started,
                    generated_source=source, generated_source_sha256=hashlib.sha256(source.encode()).hexdigest())
    except Exception as error:
        return dict(request_id=key, status="failed", error=str(error),
                    preparation_seconds=time.monotonic() - started)


def prepare(config_path=DEFAULT_CONFIG, *, output=None, build_dir=None, hardware_model=None,
            workers=4, do_compile=True, timeout=600):
    if not 1 <= workers <= 4:
        raise ValueError("CPU compile worker count must be in [1,4]")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text())
    if config.get("schema") != "cubutterfly-narrative-scaling-preparation-v1":
        raise ValueError("unknown M1 configuration schema")
    output = resolve(output or config["output"])
    build_dir = resolve(build_dir or config["build_dir"])
    if (output / "guard").exists() or (output / "timing.json").exists():
        raise ValueError("cannot replace preparation after GPU execution has started")
    study = make_study(config)
    requests, cells = module_requests(study)
    model_path = resolve(hardware_model if hardware_model is not None else config["hardware_model"])
    model = static_hardware_profile.load(model_path)
    expected_capability = f"{config['sm'] // 10}.{config['sm'] % 10}"
    if model["hardware"]["compute_capability"] != expected_capability:
        raise ValueError(f"hardware model compute capability must match configured sm{config['sm']}")
    resources = resource_rows(study, model)
    output.mkdir(parents=True, exist_ok=True)
    study_path = output / "study.json"
    controlled._atomic_write(study_path, study)
    journal = controlled.prepare_document(study, study_path, build_dir)
    controlled._atomic_write(output / "prepared.json", journal)
    source_paths = [Path(__file__), ROOT / "scripts/compile_module.py",
                    ROOT / "scripts/research_compile_requests.py", ROOT / "scripts/resident_mapping.py",
                    ROOT / "scripts/run_controlled_experiment.py", ROOT / "src/shared_iterative.cuh",
                    ROOT / "src/shared_resident.cuh", ROOT / "src/operator_traits.cuh", ROOT / "src/ntt_traits.cuh"]
    source_hashes = {str(path.relative_to(ROOT)): sha(path) for path in source_paths}
    state = dict(schema="cubutterfly-narrative-scaling-prepared-v1", status="preparing",
                 config_path=str(config_path), config_sha256=sha(config_path), build_dir=str(build_dir),
                 gpu_queried=False, gpu_verified=False, compile_policy="research", sm=80, workers=workers,
                 hardware_model_path=str(model_path), hardware_model_sha256=sha(model_path),
                 hardware_model_source="explicit-override" if hardware_model is not None else "config",
                 hardware=model["hardware"], source_sha256=source_hashes,
                 study_sha256=sha(study_path), prepared_sha256=sha(output / "prepared.json"),
                 library_sha256=sha(build_dir / "libcuntt.so") if (build_dir / "libcuntt.so").is_file() else None,
                 binaries_sha256={name: sha(build_dir / name) for name in
                                  ("cubutterfly_bench", "cuntt_bench", "cubutterfly_stage_microbench")
                                  if (build_dir / name).is_file()},
                 requests=requests, cells=cells, resources=resources, modules=[],
                 numeric_policy={"integer": "exact-full-batch", "floating": "full-error-distribution-unfinished",
                                 "thresholds_changed": False}, started_unix=time.time())
    def save():
        state["updated_unix"] = time.time()
        controlled._atomic_write(output / "preparation.json", state)
    save()
    if do_compile:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(compile_one, key, request, config, timeout) for key, request in requests.items()]
            for future in as_completed(futures):
                row = future.result()
                source = row.pop("generated_source", None)
                if source is not None:
                    source_path = output / "sources" / (row["request_id"] + ".cu")
                    source_path.parent.mkdir(exist_ok=True)
                    source_path.write_text(source)
                    row["generated_source_path"] = str(source_path)
                state["modules"].append(row)
                save()
                print(json.dumps({key: row.get(key) for key in ("request_id", "status", "error")}), flush=True)
        state["status"] = "compile-ready-runtime-unverified" if all(
            row["status"] == "ready" for row in state["modules"]) else "compile-incomplete"
    else:
        state["status"] = "prepared-not-compiled"
    if any(sha(ROOT / name) != expected for name, expected in source_hashes.items()):
        state["status"] = "source-changed-during-preparation"
    save()
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--build-dir", type=Path)
    parser.add_argument("--hardware-model", type=Path,
                        help="explicit static capacity model for a separate hardware cohort")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--no-compile", action="store_true")
    args = parser.parse_args()
    state = prepare(args.config, output=args.output, build_dir=args.build_dir,
                    hardware_model=args.hardware_model, workers=args.workers,
                    do_compile=not args.no_compile, timeout=args.timeout)
    print(json.dumps({key: state[key] for key in ("status", "gpu_queried", "gpu_verified")}), flush=True)
    return 0 if state["status"] in {"compile-ready-runtime-unverified", "prepared-not-compiled"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
