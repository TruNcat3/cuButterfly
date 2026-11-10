#!/usr/bin/env python3
"""Prepare immutable case subsets and collect independently guarded releases."""
from __future__ import annotations

import argparse
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from prepare_study_subset import prepare as prepare_subset, sha
import run_controlled_experiment as controlled

ROOT = Path(__file__).resolve().parents[2]
GUARD = ROOT / "scripts/run_isolated_gpu.py"
SCHEMA = "cubutterfly-controlled-case-collection-v1"


def deadline(value):
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise ValueError("deadline requires an explicit timezone")
    return parsed.timestamp()


def read(path):
    return json.loads(Path(path).read_text())


def snapshot(study, build_dir):
    binaries = {controlled._binary_for(build_dir, case["workload"]) for case in study["cases"]}
    for name in ("cubutterfly_bench", "cubutterfly_stage_microbench"):
        if (build_dir / name).is_file():
            binaries.add((build_dir / name).resolve())
    library = build_dir / "libcuntt.so"
    return dict(source_sha256=controlled._source_identity(),
                binaries={path.name: dict(path=str(path), sha256=sha(path)) for path in sorted(binaries)},
                library_sha256=sha(library) if library.is_file() else None,
                guard_sha256=sha(GUARD), subset_helper_sha256=sha(Path(__file__).with_name("prepare_study_subset.py")),
                queue_sha256=sha(__file__), compile_mode=os.environ.get("CUBUTTERFLY_COMPILE_MODE", "auto"))


def prepare(study_path, build_dir, output, gpu_uuid):
    study_path, build_dir, output = (Path(path).resolve() for path in (study_path, build_dir, output))
    study = read(study_path)
    controlled._validate_study(study)
    if not gpu_uuid.lower().startswith("gpu-"):
        raise ValueError("queue requires an explicit physical GPU UUID")
    identity = dict(study_path=str(study_path), study_sha256=sha(study_path), build_dir=str(build_dir),
                    gpu_uuid=gpu_uuid.lower(), gpu_selector="GPU-" + gpu_uuid[4:].lower(),
                    protocol=study["protocol"], **snapshot(study, build_dir))
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        manifest = read(manifest_path)
        if manifest["identity"] != identity:
            raise ValueError("queue source/build/hardware/protocol identity changed")
        for row in manifest["cases"]:
            for filename, expected in row["sha256"].items():
                if sha(Path(row["directory"]) / filename) != expected:
                    raise ValueError("immutable subset identity changed: " + row["id"])
        return manifest
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, case in enumerate(study["cases"]):
        # The numeric directory avoids interpreting case IDs as filesystem paths.
        directory = output / "cases" / f"{index:03d}"
        prepare_subset(study_path, directory, build_dir, case_ids=[case["id"]])
        rows.append(dict(id=case["id"], directory=str(directory),
                         sha256={name: sha(directory / name) for name in
                                 ("study.json", "prepared.json", "manifest.json")}))
    manifest = dict(schema="cubutterfly-controlled-case-queue-v1", identity=identity,
                    gpu_queried=False, cases=rows)
    controlled._atomic_write(manifest_path, manifest)
    return manifest


def validate_release(path, row, manifest, expected_device=None):
    journal = read(path)
    identity = journal.get("identity", {})
    frozen = manifest["identity"]
    study_path = Path(row["directory"]) / "study.json"
    study = read(study_path)
    prepared = read(Path(row["directory"]) / "prepared.json")
    expected_binaries = {name: frozen["binaries"][name] for name in prepared["identity"]["binaries"]}
    for field, value in dict(study_path=str(study_path), study_sha256=row["sha256"]["study.json"],
                             build_dir=frozen["build_dir"], protocol=frozen["protocol"],
                             source_sha256=frozen["source_sha256"], binaries=expected_binaries,
                             compile_mode=frozen["compile_mode"], gpu_uuid=frozen["gpu_uuid"]).items():
        if identity.get(field) != value:
            raise ValueError("release identity changed: " + field)
    device = identity.get("device", {})
    if (str(device.get("uuid", "")).lower() != frozen["gpu_uuid"] or
            not device.get("name") or not device.get("compute_capability") or
            not isinstance(device.get("global_memory_bytes"), int) or device["global_memory_bytes"] <= 0):
        raise ValueError("release hardware identity is missing or incompatible")
    if expected_device is not None and device != expected_device:
        raise ValueError("release hardware identity changed within collection")
    if journal.get("study") != controlled._validate_study(study)[0] or journal.get("protocol") != study["protocol"]:
        raise ValueError("release study contents/protocol changed")
    if journal.get("status") != "complete" or journal.get("completed") is not True:
        return None
    if len(journal.get("cases", [])) != 1:
        raise ValueError("release must contain exactly one case")
    case = journal["cases"][0]
    source_case = study["cases"][0]
    if (case.get("id") != row["id"] or case.get("workload") != source_case["workload"] or
            case.get("status") != "complete" or
            [v["id"] for v in case["variants"]] != [v["id"] for v in source_case["variants"]]):
        return None
    for variant, requested in zip(case["variants"], source_case["variants"]):
        if variant.get("mapping") != requested["mapping"] or variant.get("preflight", {}).get("status") != "accepted":
            return None
        trials = variant.get("trials", [])
        if (len(trials) != frozen["protocol"]["trials"] or
                {trial.get("trial") for trial in trials} != set(range(frozen["protocol"]["trials"])) or
                not all(t.get("status") == "measured" and t.get("performance_accepted") for t in trials)):
            return None
    if len(case.get("comparisons", [])) != len(source_case["comparisons"]):
        return None
    if not all(controlled._comparison_result(case, comparison).get("performance_accepted")
               for comparison in source_case["comparisons"]):
        return None
    return dict(status="complete", release_path=str(path.resolve()), release_sha256=sha(path),
                subset_study_sha256=row["sha256"]["study.json"], device=device,
                identity=identity)


def interrupt_guard(process):
    if process.poll() is not None:
        return
    process.send_signal(signal.SIGINT)
    try:
        process.wait(timeout=12)
        return
    except subprocess.TimeoutExpired:
        # Escalate only descendant session leaders, including the benchmark's
        # separate session; killing just the guard could strand its child.
        rows = subprocess.check_output(["ps", "-eo", "pid=,ppid=,pgid="], text=True, timeout=1)
        processes = [tuple(map(int, line.split())) for line in rows.splitlines()]
        owned = {process.pid}
        while True:
            children = {pid for pid, parent, _ in processes if parent in owned}
            if children <= owned:
                break
            owned.update(children)
        for pgid in {group for pid, _, group in processes if pid in owned and pid == group}:
            try:
                os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        process.wait(timeout=2)


def run(manifest, output, until, *, now=time.time):
    cutoff = deadline(until)
    collection_path = output / "collection.json"
    collection = read(collection_path) if collection_path.exists() else dict(
        schema=SCHEMA, manifest_sha256=sha(output / "manifest.json"),
        parent_study_sha256=manifest["identity"]["study_sha256"], device=None, records={})
    if collection.get("schema") != SCHEMA or collection.get("manifest_sha256") != sha(output / "manifest.json"):
        raise ValueError("collection manifest identity changed")
    collection.update(status="running", deadline=until)
    active = None

    def save():
        controlled._atomic_write(collection_path, collection)

    def stop(signum, frame):
        if active is not None:
            interrupt_guard(active)
        raise KeyboardInterrupt(f"queue received signal {signum}")

    handlers = {sig: signal.signal(sig, stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        for row in manifest["cases"]:
            record = collection["records"].setdefault(row["id"], dict(status="pending", guards=[]))
            directory = Path(row["directory"])
            guards = directory / "guards"
            releases = sorted(guards.glob("*/release/timing.json")) if guards.exists() else []
            accepted = None
            for path in releases:
                accepted = validate_release(path, row, manifest, collection["device"])
                if accepted is not None:
                    break
            if accepted is not None:
                collection["device"] = accepted["device"]
                record.update(accepted)
                save()
                continue
            if now() >= cutoff:
                collection["status"] = "deadline"
                break
            guards.mkdir(exist_ok=True)
            guard_dir = guards / f"{len(list(guards.iterdir())):04d}"
            command = [sys.executable, str(GUARD), "--gpu-uuid", manifest["identity"]["gpu_selector"],
                       "--until", until, "--report-dir", str(guard_dir), "--", sys.executable,
                       str(Path(__file__).resolve()), "--study", manifest["identity"]["study_path"],
                       "--build-dir", manifest["identity"]["build_dir"], "--output-dir", str(output),
                       "--gpu-uuid", manifest["identity"]["gpu_selector"], "--until", until,
                       "--guard-child", row["id"]]
            guard_dir.mkdir()
            record["guards"].append(dict(directory=str(guard_dir), command=command))
            record["status"] = "running"
            save()
            with (guard_dir / "guard.log").open("x") as log:
                active = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                code = active.wait()
                active = None
            record["guards"][-1]["returncode"] = code
            release = guard_dir / "release/timing.json"
            accepted = validate_release(release, row, manifest, collection["device"]) if release.exists() else None
            if accepted is not None:
                collection["device"] = accepted["device"]
                record.update(accepted)
            else:
                record.update(status="gap", reason="guard or numerical/measurement completion failed")
            save()
            if code == 2 or now() >= cutoff:
                collection["status"] = "deadline"
                break
        else:
            collection["status"] = "complete" if all(
                row.get("status") == "complete" for row in collection["records"].values()) else "complete-with-gaps"
    except KeyboardInterrupt as error:
        collection.update(status="interrupted", reason=str(error))
    except ValueError as error:
        collection.update(status="blocked-identity", reason=str(error))
        raise
    finally:
        if active is not None:
            interrupt_guard(active)
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
        save()
    return collection


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--until", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--guard-child")
    args = parser.parse_args(argv)
    deadline(args.until)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.guard_child:
        manifest = prepare(args.study, args.build_dir, output, args.gpu_uuid)
        row = next(row for row in manifest["cases"] if row["id"] == args.guard_child)
        attempt = Path(os.environ["XD_REPORT_DIR"]).resolve()
        if attempt.parent.parent.parent != Path(row["directory"]) / "guards":
            raise ValueError("guard child output is outside its case attempt directory")
        return controlled.main(["--study", str(Path(row["directory"]) / "study.json"),
                                "--build-dir", manifest["identity"]["build_dir"], "--mode", "run",
                                "--gpu-uuid", args.gpu_uuid, "--output", str(attempt / "timing.json")])
    with (output / "queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = prepare(args.study, args.build_dir, output, args.gpu_uuid)
        if args.prepare_only:
            print(json.dumps(dict(status="prepared", cases=len(manifest["cases"]), gpu_queried=False)))
            return 0
        collection = run(manifest, output, args.until)
        return 0 if collection["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
