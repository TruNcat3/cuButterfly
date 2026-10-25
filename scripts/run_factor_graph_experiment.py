#!/usr/bin/env python3
"""Prepare existing public factor modules/CPU reference, then measure only.

Run prepare without CUDA visibility; run measure inside run_isolated_gpu.py.
This is a finite explicit mapping experiment, not the automatic selector.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from compile_module import compile_mapping
from research_compile_requests import compile_request


ORGANIZATION_AXES = {"data_tiles_per_cta", "prefetch_depth"}


def expand_cases(suite):
    base = suite.pop("base_mapping", {})
    for case in suite["cases"]:
        case["mapping"] = dict(base if case.pop("inherit_base", True) else {}, **case["mapping"])


def check_control_groups(suite, rows=None):
    """Reject undeclared changes within each fixed-core organization control."""
    cases = {case["id"]: case for case in suite["cases"]}
    if len(cases) != len(suite["cases"]):
        raise ValueError("duplicate graph suite case id")
    mappings = {key: value["mapping"] for key, value in cases.items()}
    if rows is not None:
        mappings = {row["id"]: row["mapping"] for row in rows}
        if len(mappings) != len(rows) or mappings.keys() != cases.keys():
            raise ValueError("observed graph suite case identities differ")
    report = []
    for group in suite.get("control_groups", []):
        allowed = set(group["allowed_differences"])
        members = group["cases"]
        if (not allowed <= ORGANIZATION_AXES or len(members) < 2 or
                len(set(members)) != len(members) or any(key not in cases for key in members)):
            raise ValueError("invalid fixed-core organization control group")
        baseline = mappings[members[0]]
        if baseline.get("backend") != "factor-streamed" or baseline.get("fft_core") not in {
                "register-tile", "cufftdx-block"}:
            raise ValueError("organization controls require an explicit supported factor core")
        differences = {}
        if rows is not None:
            for key in members:
                requested = cases[key]["mapping"]
                changed = {field for field, expected in requested.items()
                           if not (field == "compute_unit" and expected == "auto")
                           and mappings[key].get(field) != expected}
                if changed:
                    raise ValueError(f"{group['id']}/{key}: actual mapping differs from request: "
                                     f"{sorted(changed)}")
        for key in members[1:]:
            current = mappings[key]
            changed = {field for field in baseline.keys() | current.keys()
                       if baseline.get(field) != current.get(field)}
            if not changed <= allowed:
                raise ValueError(f"{group['id']}/{key}: fixed-core control differs beyond allowed axes: "
                                 f"{sorted(changed - allowed)}")
            differences[key] = {field: [baseline.get(field), current.get(field)]
                                for field in sorted(changed)}
        report.append(dict(id=group["id"], baseline=members[0], fft_core=baseline["fft_core"],
                           semantics={key: suite[key] for key in
                                      ("operator", "precision", "logN", "batch", "inverse")},
                           allowed_differences=sorted(allowed), differences=differences))
    return report


def module_request(suite, mapping):
    return compile_request(dict(operator=suite["operator"], precision=suite["precision"],
                                logN=suite["logN"], mapping_json=mapping), "research")


def canonical_request(request):
    request = dict(request)
    if request.get("backend") == "factor-streamed":
        policies = request.pop("factor_io_policies", [])
        if any(policy != "dynamic" for policy in policies):
            request["factor_io_policies"] = policies
    return request


def module_lookup(argv):
    """Adapt the runtime compiler protocol to a prepared-module-only lookup."""
    parser = argparse.ArgumentParser()
    parser.add_argument("compiler_script")
    parser.add_argument("--mapping", required=True, type=json.loads)
    parser.add_argument("--sm", required=True, type=int)
    for key in ("root", "mathdx", "nvcc", "cache", "timeout"):
        parser.add_argument("--" + key, required=True)
    args = parser.parse_args(argv)
    manifest = json.loads(Path(os.environ["CUBUTTERFLY_FACTOR_GRAPH_PREPARED"]).read_text())
    if not manifest.get("complete") or args.sm != manifest["sm"]:
        raise RuntimeError("prepared module architecture mismatch or incomplete preparation")
    for entry in manifest["modules"]:
        if "point" in entry and canonical_request(entry["point"]) == canonical_request(args.mapping):
            if sha(entry["module"]) != entry["sha256"]:
                raise RuntimeError("prepared module identity changed")
            print(entry["module"])
            return 0
    raise RuntimeError("runtime requested an unprepared module; compilation is disabled during measurement")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def measurement_environment(manifest, manifest_path):
    hardware_path = Path(manifest["hardware_model"])
    if sha(hardware_path) != manifest["hardware_sha256"]:
        raise RuntimeError("prepared hardware model identity changed")
    # The public runtime checks this artifact against the visible CUDA device's
    # model, compute capability and exact memory capacity before plan creation.
    return dict(os.environ, CUBUTTERFLY_JIT_CACHE=manifest["cache"],
                CUBUTTERFLY_COMPILE_MODE="research",
                CUBUTTERFLY_JIT_PYTHON=str(Path(__file__).resolve()),
                CUBUTTERFLY_FACTOR_GRAPH_PREPARED=str(manifest_path),
                CUBUTTERFLY_HARDWARE_MODEL=str(hardware_path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["prepare", "measure"])
    parser.add_argument("--config", type=Path)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--hardware-model", type=Path)
    parser.add_argument("--cache", type=Path, default=Path.home()/".cache/cubutterfly/modules")
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--batch", type=int, help="explicit preparation-only workload override")
    parser.add_argument("--direction", choices=["forward", "inverse"], help="explicit preparation-only override")
    parser.add_argument("--output", type=Path, default=os.environ.get("XD_REPORT_DIR"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    binary = args.binary.resolve()
    prepared = args.prepared.resolve()
    manifest_path = prepared.with_suffix(".manifest.json")
    if args.phase == "prepare":
        if args.config is None or args.hardware_model is None or not 1 <= args.jobs <= 8:
            parser.error("prepare requires config, hardware-model, and 1..8 jobs")
        prepared.parent.mkdir(parents=True, exist_ok=True)
        suite = json.loads(args.config.read_text())
        if args.batch is not None:
            suite["batch"] = args.batch
        if args.direction is not None:
            suite["inverse"] = args.direction == "inverse"
        if suite["operator"] != "fft" or suite["precision"] != "fp64" or not 1 <= suite["batch"] <= 65535:
            raise ValueError("this bounded graph experiment requires FFT FP64 and batch in 1..65535")
        hardware = json.loads(args.hardware_model.read_text())["hardware"]
        expand_cases(suite)
        controls = check_control_groups(suite)
        suite["reference_file"] = str(prepared.with_suffix(".reference.bin"))
        prepared.write_text(json.dumps(suite, indent=2)+"\n")
        manifest = dict(config_sha256=sha(args.config), prepared_sha256=sha(prepared),
                        binary_sha256=sha(binary), library_sha256=sha(binary.parent/"libcuntt.so"),
                        hardware_model=str(args.hardware_model.resolve()), hardware_sha256=sha(args.hardware_model),
                        cache=str(args.cache.resolve()), modules=[],
                        sm=int(hardware["compute_capability"].replace(".", "")),
                        control_groups=controls,
                        created=datetime.now().astimezone().isoformat())
        def prepare(case):
            mapping = case["mapping"]
            if mapping["backend"] == "cufft":
                return dict(id=case["id"], provider="cufft", preparation="library plan created before timing")
            if mapping["backend"] != "factor-streamed":
                raise ValueError("only factor-streamed and complete cuFFT comparator are supported here")
            for log in mapping["factor_partition"]:
                # depth+1 input-ring slots plus one reusable FFT work tile;
                # provider scratch may make the compiled footprint larger.
                depth = mapping["prefetch_depth"] if mapping["data_tiles_per_cta"] > 1 else 0
                slots = depth + 2 if depth else 1
                lower_shared = (1 << log)*mapping["factor_columns"]*16*slots
                threads = (1 << log)//mapping["factor_ept"]*mapping["factor_columns"]
                if lower_shared > hardware["shared_bytes_per_block"] or threads > hardware["max_threads_per_block"]:
                    raise ValueError(f"{case['id']}: exceeds static hardware capacity before compilation")
            point = module_request(suite, mapping)
            module = compile_mapping(point, root=root, mathdx=root/"external/mathdx/nvidia/mathdx",
                                     nvcc=Path("/usr/local/cuda/bin/nvcc"),
                                     sm=int(hardware["compute_capability"].replace(".", "")),
                                     cache=args.cache.resolve(), timeout=300)
            print(f"PREPARED {case['id']}", flush=True)
            return dict(id=case["id"], point=point, module=str(module), sha256=sha(module))
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            for result in pool.map(prepare, suite["cases"]):
                manifest["modules"].append(result)
                manifest_path.write_text(json.dumps(manifest, indent=2)+"\n")
        subprocess.run([str(binary), "--prepare-reference", str(prepared)], check=True,
                       env=dict(os.environ, CUDA_VISIBLE_DEVICES=""), timeout=300)
        manifest["reference_sha256"] = sha(suite["reference_file"])
        manifest["complete"] = True
        manifest_path.write_text(json.dumps(manifest, indent=2)+"\n")
        return
    if args.output is None:
        parser.error("measure needs --output or XD_REPORT_DIR from the GPU guard")
    if args.batch is not None or args.direction is not None:
        parser.error("workload overrides require a new CPU preparation")
    manifest = json.loads(manifest_path.read_text())
    suite = json.loads(prepared.read_text())
    env = measurement_environment(manifest, manifest_path)
    if not manifest.get("complete") or sha(prepared) != manifest["prepared_sha256"] or sha(binary) != manifest["binary_sha256"] or sha(binary.parent/"libcuntt.so") != manifest["library_sha256"]:
        raise RuntimeError("preparation incomplete or build/config identity changed")
    if sha(suite["reference_file"]) != manifest["reference_sha256"]:
        raise RuntimeError("prepared reference identity changed")
    check_control_groups(suite)
    entries = {entry["id"]: entry for entry in manifest["modules"]}
    if len(entries) != len(suite["cases"]) or entries.keys() != {case["id"] for case in suite["cases"]}:
        raise RuntimeError("prepared module coverage differs from suite")
    for case in suite["cases"]:
        if case["mapping"]["backend"] != "cufft":
            if entries[case["id"]].get("point") != module_request(suite, case["mapping"]):
                raise RuntimeError("prepared compiler request differs from public mapping; prepare again")
    for entry in manifest["modules"]:
        if "module" in entry and sha(entry["module"]) != entry["sha256"]:
            raise RuntimeError("prepared module identity changed")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/"preparation.json").write_text(json.dumps(manifest, indent=2)+"\n")
    (args.output/"gpu.txt").write_text(subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid,name,memory.total,driver_version", "--format=csv,noheader"], text=True))
    completed = subprocess.run([str(binary), "--suite", str(prepared)], capture_output=True, text=True,
                               env=env, timeout=600)
    (args.output/"results.json").write_text(completed.stdout)
    (args.output/"stderr.log").write_text(completed.stderr)
    if completed.returncode:
        raise RuntimeError(f"graph suite failed with {completed.returncode}; see preserved output")
    expected_source = "cuButterfly hardware resources: static-hardware-model:" + str(Path(manifest["hardware_model"]))
    if expected_source not in completed.stderr:
        raise RuntimeError("runtime did not confirm the prepared hardware model")
    observed = json.loads(completed.stdout)
    if not observed["all_outputs_finite"] or len(observed["results"]) != len(suite["cases"]):
        raise RuntimeError("incomplete or nonfinite graph suite")
    controls = check_control_groups(suite, observed["results"])
    if controls:
        (args.output/"control_groups.json").write_text(json.dumps(controls, indent=2)+"\n")
    for row in observed["results"]:
        print(row["id"], row["median_ms"], "ms", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and Path(sys.argv[1]).name == "compile_module.py":
        raise SystemExit(module_lookup(sys.argv[1:]))
    main()
