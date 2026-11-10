#!/usr/bin/env python3
"""Frozen application-contract studies using public NTT plans and existing search.

The cyclic phase reuses the existing calibrated research acceptance workflow.
The composed phase measures GPU twists and all RNS channels together. It is
an application composition experiment, not an external-library comparison.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STUDY = ROOT / "results/crypto_application_20260920"
BASIS = ROOT / "results/comprehensive_20260919"
BATCH_STUDY = ROOT / "results/batch_precision_20260920"
sys.path.insert(0, str(ROOT / "scripts"))
from run_research_acceptance import cell_key


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


shared = module("crypto_shared_study", BATCH_STUDY / "study.py")
controller = module("crypto_shared_controller", BASIS / "resume_when_idle.py")
read, sha, freeze = shared.read, shared.sha, shared.freeze
COMPOSED_PROTOCOL = dict(trials=3, warmup=100, repeat=100,
                         input_patterns=["random", "boundary"], seed=20260920)


def configure(study):
    shared.HERE = Path(study).resolve()
    return shared.HERE


def manifest(study):
    return read(Path(study) / "crypto_manifest.json")


def validate_inputs(study):
    meta = manifest(study)
    if (meta.get("targets") != shared.TARGETS or meta.get("protocol") != COMPOSED_PROTOCOL
            or meta.get("compile_mode") != "research"
            or meta.get("dependencies") != [str(BASIS), str(BATCH_STUDY)]):
        raise ValueError("crypto study hardware/protocol/dependencies differ")
    for name, expected in meta["inputs"].items():
        if sha(Path(study) / name) != expected:
            raise ValueError(f"crypto study input changed: {name}")
    if sha(meta["binary"]["path"]) != meta["binary"]["sha256"]:
        raise ValueError("composition binary changed")
    for path, expected in meta["code"].items():
        if sha(path) != expected:
            raise ValueError(f"composition source/driver changed: {path}")
    build = read(Path(study) / "build.json")
    shared.verify_build(build)
    for name, expected in build.get("source_inputs", {}).items():
        if sha(ROOT / name) != expected:
            raise ValueError(f"public-build source/experiment input changed: {name}")
    return meta


def prepare(study, binary):
    study = configure(study)
    if not Path(binary).is_file():
        raise ValueError("compile the isolated crypto_ntt benchmark before preparation")
    shared.prepare()
    code = [Path(__file__), ROOT / "scripts/build_crypto_application_matrix.py",
            BATCH_STUDY / "study.py", BATCH_STUDY / "report.py", BASIS / "resume_when_idle.py"]
    code += [ROOT / "scripts" / name for name in (
        "run_research_acceptance.py", "research_baselines.py", "calibrate_local_hardware.py",
        "calibration_space.py", "calibration_seeds.py", "stage_service_calibration.py",
        "stage_cost_model.py", "stage_service_model.py", "stage_composition.py",
        "hardware_registry.py", "evolutionary_search.py", "verify_local_selector.py")]
    code += sorted(p for p in (ROOT / "benchmarks/crypto_ntt").iterdir()
                   if p.is_file() and p.suffix in (".cu", ".cuh", ".cpp", ".hpp", ".txt"))
    freeze(study / "crypto_manifest.json", dict(
        schema="crypto-application-execution-v1", targets=shared.TARGETS,
        dependencies=[str(BASIS), str(BATCH_STUDY)],
        protocol=COMPOSED_PROTOCOL, compile_mode="research",
        declared=dict(new_cyclic=len(read(study / "workloads.json")["workloads"]),
            new_composed=len(composed_rows(study)),
            total_distinct_contracts=len(read(study / "population.json")["cells"]) + len(composed_rows(study)),
            gpu_correctness_preflight=2 * len(preflight_workloads(study))),
        inputs={name: sha(study / name) for name in (
            "population.json", "workloads.json", "precompile_workloads.json",
            "composed_workloads.json", "build.json", "manifest.json", "run.sh")},
        code={str(path): sha(path) for path in code},
        binary=dict(path=str(Path(binary).resolve()), sha256=sha(binary)),
        composition_scope="Exact modular single-word plans; GPU twists included; sequential RNS channels. No CRT, key operations or external-library wins.",
        wait_policy="Fixed 48-hour window; waits for both existing studies and their scheduling locks; never resets their windows."))


def dependencies_ready(study, target):
    for path in manifest(study)["dependencies"]:
        directory = Path(path)
        if not shared.parent_resolved(target, directory):
            return [f"waiting for {directory.name} matrix"]
        try:
            with controller.acquire_lock(directory / target / "resume_when_idle.lock"):
                pass
        except controller.LockBusy:
            return [f"{directory.name} still owns GPU scheduling"]
    compilation = Path(study) / "precompile/compilation.json"
    if not compilation.exists() or not read(compilation).get("complete"):
        return ["waiting for finite CPU precompilation"]
    return []


def composed_rows(study):
    return read(Path(study) / "composed_workloads.json")["workloads"]


def bench_command(binary, workload, *, pattern, trial, mapping=None):
    command = [str(binary), "--log-n", str(workload["logN"]),
               "--batch", str(workload["batch"]), "--word-bits", str(workload["word_bits"]),
               "--moduli", ",".join(map(str, workload["moduli"])),
               "--mode", workload["mode"], "--direction", workload.get("direction", "forward"),
               "--warmup", str(COMPOSED_PROTOCOL["warmup"]),
               "--repeat", str(COMPOSED_PROTOCOL["repeat"]),
               "--seed", str(COMPOSED_PROTOCOL["seed"] + trial),
               "--input-pattern", pattern, "--json"]
    if workload["mode"] == "coset":
        command += ["--coset-generator", str(workload.get("coset_generator", 7))]
    if mapping is not None:
        command += ["--mapping-json", json.dumps(mapping, separators=(",", ":"))]
    return command


def validate_sample(sample, workload, uuid):
    """Reject a correct result for a different field/shape/direction or GPU."""
    fields = {"logN": int(workload["logN"]), "batch": int(workload["batch"]),
              "word_bits": int(workload["word_bits"]), "mode": workload["mode"],
              "direction": workload.get("direction", "forward")}
    if any(sample.get(k) != v for k, v in fields.items()):
        raise ValueError("composition result has a different semantic contract")
    if list(map(str, sample.get("moduli", []))) != list(map(str, workload["moduli"])):
        raise ValueError("composition modulus channels differ")
    if sample.get("gpu_uuid") != uuid or sample.get("compile_mode") != "research":
        raise ValueError("composition hardware/compile policy differs")
    if sample.get("correct") is not True:
        raise ValueError("composition exact modular verification failed")
    if (sample.get("verified_batches") != int(workload["batch"])
            or sample.get("verified_channels") != len(workload["moduli"])):
        raise ValueError("composition did not verify every batch/channel")
    if (sample.get("warmup") != COMPOSED_PROTOCOL["warmup"]
            or sample.get("repeat") != COMPOSED_PROTOCOL["repeat"]):
        raise ValueError("composition timing protocol differs")
    if (sample.get("normalization") != ("inverse" if fields["direction"] == "inverse" else "none")
            or sample.get("layout") != "channel-batch-coefficient"
            or sample.get("rns_stream_order") != "single-stream"):
        raise ValueError("composition normalization/layout/schedule differs")
    if workload["mode"] == "coset" and sample.get("coset_generator") != int(workload.get("coset_generator", 7)):
        raise ValueError("composition coset differs")
    if not math.isfinite(float(sample.get("kernel_ms", 0))) or float(sample.get("kernel_ms", 0)) <= 0:
        raise ValueError("invalid composition timing")


def validate_trial(sample, workload, uuid, pattern, trial, mapping=None):
    validate_sample(sample, workload, uuid)
    if (sample.get("input_pattern") != pattern
            or sample.get("seed") != COMPOSED_PROTOCOL["seed"] + trial):
        raise ValueError("composition input pattern/seed differs")
    if sample.get("mapping_request") != mapping:
        raise ValueError("composition mapping request differs")


def portfolios(study, target, workload):
    """Reuse exact-contract confirmed cyclic mappings; never transfer timings."""
    validator = module("crypto_cyclic_evidence", BATCH_STUDY / "report.py")
    known = {}
    for directory in (BASIS, BATCH_STUDY, Path(study)):
        path = directory / target / "acceptance/acceptance.json"
        if not path.exists():
            continue
        journal = read(path)
        shared.validate_parent(target, journal, read(Path(study) / "build.json"))
        for cell in journal["cells"]:
            if cell.get("status") == "complete":
                known[cell["id"]] = (cell, path)
    mappings, provenance = [], []
    for p in workload["moduli"]:
        cyclic = dict(operator="ntt", precision=f"word{workload['word_bits']}",
            logN=workload["logN"], batch=workload["batch"], modulus=str(p),
            input_order="natural", output_order="natural")
        if workload.get("direction", "forward") == "inverse":
            cyclic["direction"] = "inverse"
        hit = known.get(cell_key(cyclic))
        selected = validator._selected_sample(hit[0]) if hit else None
        if hit and selected and not validator._cell_reason(hit[0], cyclic, selected.get("runtime_fingerprint")):
            mappings.append(json.loads(selected["mapping_json"]))
            provenance.append(dict(kind="exact-contract-cyclic-search", cell_id=cell_key(cyclic),
                                   journal=str(hit[1]), journal_sha256=sha(hit[1])))
        else:
            mappings.append(None)
            provenance.append(dict(kind="public-default", reason="no complete exact-contract cyclic search"))
    result = [dict(name="public-default", mappings=None,
                   provenance="linked/specialized shared-iterative proposal; not claimed optimal")]
    if any(m is not None for m in mappings):
        result.append(dict(name="searched-cyclic-cores", mappings=mappings, provenance=provenance,
            scope="Exact cyclic mappings replayed inside composed semantics; not joint RNS/twist schedule search."))
    return result


def measure(command, env, uuid, log_path):
    clients = controller.query_compute_clients(uuid)
    if clients:
        raise RuntimeError("target GPU is not exclusive: " + "; ".join(clients))
    process = subprocess.Popen(command, cwd=ROOT, env=env, text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        while True:
            try:
                output, error = process.communicate(timeout=1)
                break
            except subprocess.TimeoutExpired:
                foreign = [row for row in controller.query_compute_clients(uuid)
                           if row.split(",", 1)[0].strip() != str(process.pid)]
                if foreign:
                    raise RuntimeError("target GPU is not exclusive: " + "; ".join(foreign))
        foreign = controller.query_compute_clients(uuid)
        if foreign:
            raise RuntimeError("target GPU is not exclusive after measurement: " + "; ".join(foreign))
        controller.atomic_write_json(log_path, dict(command=command, returncode=process.returncode,
                                                   stdout=output, stderr=error))
        if process.returncode:
            raise RuntimeError(f"composition benchmark failed ({process.returncode}): {(error or output)[-3000:]}")
        return json.loads(output)
    except BaseException as failure:
        if process.poll() is None:
            process.terminate()  # Only this runner's own benchmark process.
            try:
                output, error = process.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                output, error = process.communicate()
        else:
            output, error = process.communicate()
        controller.atomic_write_json(log_path, dict(command=command, status="excluded",
            reason=str(failure), stdout=output, stderr=error))
        raise


def run_composed(study, target, env):
    meta = validate_inputs(study)
    path = Path(study) / target / "composed.json"
    identity = dict(gpu_uuid=meta["targets"][target], manifest_sha256=sha(Path(study) / "crypto_manifest.json"))
    document = read(path) if path.exists() else dict(schema="crypto-composed-results-v1", identity=identity,
        status="partial", cells=[], external_baseline="unavailable; composition characterization only")
    if document["identity"] != identity:
        raise ValueError("composition resume identity changed")
    for workload in composed_rows(study):
        cell = next((c for c in document["cells"] if c["id"] == workload["id"]), None)
        if cell is None:
            cell = dict(id=workload["id"], workload=workload, status="pending", measurements=[],
                        portfolios=portfolios(study, target, workload))
            document["cells"].append(cell)
        if cell["workload"] != workload:
            raise ValueError("composition workload changed on resume")
        if cell["portfolios"] != portfolios(study, target, workload):
            raise ValueError("composition mapping evidence changed on resume")
        for pattern in COMPOSED_PROTOCOL["input_patterns"]:
            for trial in range(COMPOSED_PROTOCOL["trials"]):
                # Alternate portfolios within each trial to reduce order bias.
                for portfolio in cell["portfolios"][::1 if trial % 2 == 0 else -1]:
                    name = portfolio["name"]
                    previous = [r for r in cell["measurements"] if r["pattern"] == pattern and r["trial"] == trial and r["portfolio"] == name]
                    if previous:
                        if len(previous) != 1:
                            raise ValueError("duplicate composition trial")
                        validate_trial(previous[0]["sample"], workload, identity["gpu_uuid"], pattern, trial, portfolio["mappings"])
                        continue
                    command = bench_command(meta["binary"]["path"], workload, pattern=pattern, trial=trial, mapping=portfolio["mappings"])
                    log = Path(study) / target / "composed_logs" / f"{workload['id']}-{name}-{pattern}-{trial}.json"
                    try:
                        sample = measure(command, env, identity["gpu_uuid"], log)
                        validate_trial(sample, workload, identity["gpu_uuid"], pattern, trial, portfolio["mappings"])
                    except Exception as error:
                        cell.update(status="interrupted" if "not exclusive" in str(error) else "failed", error=str(error))
                        controller.atomic_write_json(path, document)
                        raise
                    cell["measurements"].append(dict(pattern=pattern, trial=trial, portfolio=name, sample=sample))
                    cell["status"] = "partial"
                    controller.atomic_write_json(path, document)
        cell.update(status="complete")
        cell.pop("error", None)
        controller.atomic_write_json(path, document)
    document["status"] = "complete"
    controller.atomic_write_json(path, document)


def preflight_workloads(study):
    specs = read(Path(study) / "population.json")["moduli"]
    by_bits = {int(s["bits"]): str(s["modulus"]) for s in specs}
    rows = []
    for bits, log_n, moduli in ((32, 8, [by_bits[23], by_bits[31]]),
                                (64, 12, [by_bits[40], by_bits[62]])):
        for mode in ("cyclic", "negacyclic", "coset"):
            for direction in ("forward", "inverse"):
                rows.append(dict(id=f"smoke-{bits}-{mode}-{direction}", mode=mode,
                    logN=log_n, batch=3, word_bits=bits, moduli=moduli,
                    direction=direction, coset_generator=7))
    return rows


def gpu_preflight(study, target, env):
    """Check the new GPU composition before spending time on full search."""
    meta = validate_inputs(study)
    path = Path(study) / target / "gpu_preflight.json"
    identity = dict(gpu_uuid=meta["targets"][target], manifest_sha256=sha(Path(study) / "crypto_manifest.json"))
    document = read(path) if path.exists() else dict(identity=identity, samples=[], status="partial",
        scope="GPU correctness gate only; these shortened workloads are outside reported performance population")
    if document["identity"] != identity:
        raise ValueError("GPU preflight resume identity changed")
    for workload in preflight_workloads(study):
        for pattern in COMPOSED_PROTOCOL["input_patterns"]:
            rows = [r for r in document["samples"] if r["id"] == workload["id"] and r["pattern"] == pattern]
            if len(rows) > 1:
                raise ValueError("duplicate GPU preflight sample")
            if rows:
                validate_trial(rows[0]["sample"], workload, identity["gpu_uuid"], pattern, 0)
                continue
            sample = measure(bench_command(meta["binary"]["path"], workload, pattern=pattern, trial=0),
                env, identity["gpu_uuid"], Path(study) / target / "preflight_logs" / f"{workload['id']}-{pattern}.json")
            validate_trial(sample, workload, identity["gpu_uuid"], pattern, 0)
            document["samples"].append(dict(id=workload["id"], pattern=pattern, sample=sample))
            controller.atomic_write_json(path, document)
    document["status"] = "complete"
    controller.atomic_write_json(path, document)


def execute(study, target):
    study = configure(study)
    meta = validate_inputs(study)
    try:
        with ExitStack() as stack:
            for dependency in meta["dependencies"]:
                stack.enter_context(controller.acquire_lock(Path(dependency) / target / "resume_when_idle.lock"))
                if not shared.parent_resolved(target, Path(dependency)):
                    raise ValueError("prior matrix unresolved")
            if controller.query_compute_clients(meta["targets"][target]):
                raise RuntimeError("target GPU is not exclusive")
            if not read(study / "precompile/compilation.json").get("complete"):
                raise ValueError("finite precompilation incomplete")
            shared.snapshot_target(target)
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": meta["targets"][target],
                   "CUBUTTERFLY_COMPILE_MODE": "research", "CUBUTTERFLY_TEMPLATE_ROOT": str(ROOT),
                   "CUBUTTERFLY_MATHDX_ROOT": str(ROOT / "external/mathdx/nvidia/mathdx"),
                   "CUBUTTERFLY_NVCC": "/usr/local/cuda/bin/nvcc", "CUBUTTERFLY_JIT_PYTHON": sys.executable,
                   "CUBUTTERFLY_SEARCH_SCORE_WORKERS": "8",
                   "CUBUTTERFLY_REGISTRY": str(study / target / "acceptance/registry.json")}
            gpu_preflight(study, target, env)
            if not shared.parent_resolved(target, study):
                code = subprocess.call(shared.command(target), cwd=ROOT, env=env)
                if code:
                    return code
                if not shared.parent_resolved(target, study):
                    raise ValueError("cyclic acceptance exited without resolving its matrix")
            run_composed(study, target, env)
            return 0
    except controller.LockBusy as error:
        print("parent-scheduling-lock-busy: " + str(error), file=sys.stderr)
        return 75


def report(study):
    study = Path(study)
    meta = validate_inputs(study)
    population = read(study / "population.json")
    summary = dict(schema="crypto-application-summary-v1", status="partial", targets={},
                   unsupported=population.get("unsupported", []))
    evidence = module("crypto_report_evidence", BATCH_STUDY / "report.py")
    declared_cyclic = {cell_key(w): w for w in read(study / "workloads.json")["workloads"]}
    for target, uuid in meta["targets"].items():
        cyclic = study / target / "acceptance/acceptance.json"
        composed = study / target / "composed.json"
        cj = read(cyclic) if cyclic.exists() else {}
        dj = read(composed) if composed.exists() else {}
        if cj:
            shared.validate_parent(target, cj, read(study / "build.json"))
        cyclic_complete, cyclic_unavailable, seen = 0, 0, set()
        for cell in cj.get("cells", []):
            identifier = cell["id"]
            if identifier in seen or identifier not in declared_cyclic:
                raise ValueError("cyclic report has duplicate/undeclared cells")
            seen.add(identifier)
            if cell.get("status") == "complete":
                selected = evidence._selected_sample(cell)
                problems = evidence._cell_reason(cell, declared_cyclic[identifier],
                    selected.get("runtime_fingerprint") if selected else None)
                if problems:
                    raise ValueError("invalid complete cyclic evidence: " + ", ".join(problems))
                cyclic_complete += 1
            elif cell.get("status") == "unavailable":
                cyclic_unavailable += 1
        if dj and dj.get("identity") != dict(gpu_uuid=uuid, manifest_sha256=sha(study / "crypto_manifest.json")):
            raise ValueError("composed report identity differs")
        rows, complete_cells = [], 0
        composed_seen = set()
        for cell in dj.get("cells", []):
            if cell["id"] in composed_seen:
                raise ValueError("duplicate composed cell")
            composed_seen.add(cell["id"])
            if cell.get("status") != "complete":
                continue
            workload = next(w for w in composed_rows(study) if w["id"] == cell["id"])
            if cell["workload"] != workload or cell["portfolios"] != portfolios(study, target, workload):
                raise ValueError("composed report workload/mapping evidence differs")
            expected = {(p, t, v["name"]) for p in COMPOSED_PROTOCOL["input_patterns"] for t in range(COMPOSED_PROTOCOL["trials"]) for v in cell["portfolios"]}
            if (len(cell["measurements"]) != len(expected)
                    or {(r["pattern"], r["trial"], r["portfolio"]) for r in cell["measurements"]} != expected):
                raise ValueError("incomplete composed trial grid")
            for row in cell["measurements"]:
                mapping = next(p["mappings"] for p in cell["portfolios"] if p["name"] == row["portfolio"])
                validate_trial(row["sample"], workload, uuid, row["pattern"], row["trial"], mapping)
            complete_cells += 1
            for pattern in COMPOSED_PROTOCOL["input_patterns"]:
                for portfolio in cell["portfolios"]:
                    ms = statistics.median(r["sample"]["kernel_ms"] for r in cell["measurements"] if r["pattern"] == pattern and r["portfolio"] == portfolio["name"])
                    rows.append(dict(id=cell["id"], workload=workload, pattern=pattern, portfolio=portfolio,
                        kernel_ms=ms, ring_batches_s=workload["batch"] * 1000 / ms,
                        residue_transforms_s=workload["batch"] * len(workload["moduli"]) * 1000 / ms,
                        modulus_bits=[int(p).bit_length() for p in workload["moduli"]],
                        combined_modulus_bits=math.prod(int(p) for p in workload["moduli"]).bit_length()))
        summary["targets"][target] = dict(cyclic_complete=cyclic_complete, cyclic_unavailable=cyclic_unavailable,
            cyclic_total=len(declared_cyclic),
            composed_complete=complete_cells,
            composed_total=len(composed_rows(study)), composed_rows=rows)
    if all(v["cyclic_complete"] == v["cyclic_total"] and v["composed_complete"] == v["composed_total"] for v in summary["targets"].values()):
        summary["status"] = "complete"
    elif all(v["cyclic_complete"] + v["cyclic_unavailable"] == v["cyclic_total"] and v["composed_complete"] == v["composed_total"] for v in summary["targets"].values()):
        summary["status"] = "complete-with-unavailable-cells"
    controller.atomic_write_json(study / "summary.json", summary)
    print(json.dumps({"status": summary["status"], "targets": {t: {k: v for k, v in d.items() if k != "composed_rows"} for t, d in summary["targets"].items()}}))


def queue(study, target, wait_seconds, poll_seconds):
    study = configure(study)
    validate_inputs(study)

    class CryptoController(controller.Controller):
        def stage_status(self):
            return "complete"

        def matrix_resolved(self):
            path = study / target / "composed.json"
            if not shared.parent_resolved(target, study) or not path.exists():
                return False
            data = read(path)
            cells = data.get("cells", [])
            return (data.get("status") == "complete" and bool(cells)
                    and all(c.get("status") == "complete" for c in cells)
                    and len(cells) == len(composed_rows(study))
                    and {c["id"] for c in cells} == {w["id"] for w in composed_rows(study)})

        @staticmethod
        def recoverable_occupancy(text, code):
            return (controller.Controller.recoverable_occupancy(text, code)
                    or (code == 75 and "parent-scheduling-lock-busy:" in text))

        def postprocess(self):
            code, output = self._run_command([sys.executable, str(Path(__file__).resolve()),
                "report", "--study-dir", str(study)], "report", os.environ.copy())
            if code:
                return self.fail(self.error_excerpt(output))
            self.state.update(status="complete", phase="report", last_error=None, runner_pid=None)
            self.save()
            return 0

    args = controller.parse_args([target, "--phase", "accept", "--wait-seconds", str(wait_seconds), "--poll-seconds", str(poll_seconds)])
    instance = CryptoController(args, base_dir=study,
        query=lambda uuid: dependencies_ready(study, target) or controller.query_compute_clients(uuid))
    try:
        with controller.acquire_lock(instance.lock_path):
            return instance.run()
    except controller.LockBusy as error:
        print(error, file=sys.stderr)
        return 75


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "execute", "queue", "report"))
    parser.add_argument("target", nargs="?", choices=tuple(shared.TARGETS))
    parser.add_argument("--study-dir", type=Path, default=DEFAULT_STUDY)
    parser.add_argument("--binary", type=Path, default=DEFAULT_STUDY / "build/crypto_ntt_bench")
    parser.add_argument("--wait-seconds", type=float, default=172800)
    parser.add_argument("--poll-seconds", type=float, default=30)
    args = parser.parse_args(argv)
    if args.mode == "prepare":
        prepare(args.study_dir, args.binary)
    elif args.mode == "report":
        report(args.study_dir)
    elif not args.target:
        parser.error("execute/queue requires a target")
    elif args.mode == "execute":
        return execute(args.study_dir, args.target)
    else:
        return queue(args.study_dir, args.target, args.wait_seconds, args.poll_seconds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
