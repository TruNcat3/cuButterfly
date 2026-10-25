#!/usr/bin/env python3
"""CPU compilation broker for the unchanged public MEASURE runtime.

Initialize with ``--init --broker DIR --root ROOT --mathdx DIR --nvcc FILE
--sm 80 --cache DIR``; ``--worker --broker DIR --jobs N`` drains pending
requests, finishing active compilations on SIGINT/SIGTERM. Failed requests
remain failed until an explicitly requested ``--retry-failed`` worker run.
``--watch`` keeps a CPU producer alive for future requests and atomically
updates producer_status.json. ``--seed-cache-since LIBRARY`` queues unique
matching-SM cache mappings with manifests newer than that library; the worker
always validates the current compile key rather than trusting historical paths.
In watch mode seeding runs in a background thread, so scanning cannot delay
runtime compilation requests or producer heartbeats.

Set CUBUTTERFLY_JIT_PYTHON to this executable, CUBUTTERFLY_PREPARE_BROKER
to DIR and CUBUTTERFLY_PREPARE_EVENTS to a separate JSONL file per attempt.
The C++ runtime invokes this executable with compile_module.py and its normal
arguments. It only returns a verified ready path or queues a request and exits
nonzero. The caller MUST discard the entire attempt if its event file is not
empty: MEASURE can catch the compiler error and otherwise prune a candidate.

Public Python API: initialize(broker, config), runtime_request(broker, config,
mapping, events, timeout=600), work(broker, jobs=1, retry_failed=False),
status(broker). Config contains root, mathdx, nvcc, sm and cache. Paths are
absolute after normalization. Existing compile_module.py owns cache hashing
and compilation; this file never changes its inputs or the compile policy.
"""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time

SCHEMA = "cubutterfly-semantic-compile-broker-v1"
CONFIG_FIELDS = ("root", "mathdx", "nvcc", "sm", "cache")


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _id(value):
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _read(path):
    return json.loads(Path(path).read_text())


def _atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".publish-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(_canonical(value) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _config(config):
    config = dict(config)
    config["nvcc"] = shutil.which(str(config["nvcc"])) or config["nvcc"]
    result = {key: str(Path(config[key]).expanduser().resolve())
              for key in CONFIG_FIELDS if key != "sm"}
    result["sm"] = int(config["sm"])
    if result["sm"] <= 0:
        raise ValueError("sm must be positive")
    return result


def _stat(path):
    value = Path(path).stat()
    return {"size": value.st_size, "mtime_ns": value.st_mtime_ns}


def _sources(config):
    root, mathdx = Path(config["root"]), Path(config["mathdx"])
    files = set((root / "include").rglob("*.h"))
    files.update((root / "include").rglob("*.hpp"))
    files.update((root / "src").glob("*.cuh"))
    files.update((mathdx / "include").rglob("*.hpp"))
    files.update((mathdx / "include").rglob("*.hpp.inc"))
    files.update(root / "scripts" / name for name in
                 ("compile_module.py", "resident_mapping.py", "cufftdx_capabilities.py"))
    digest = hashlib.sha256()
    for path in sorted(files):
        digest.update(str(path).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def initialize(broker, config):
    """Create an immutable broker identity, or validate an existing one."""
    broker, config = Path(broker), _config(config)
    version = subprocess.run([config["nvcc"], "--version"], check=True,
                             capture_output=True, text=True).stdout
    identity = {"config": config, "sources_sha256": _sources(config),
                "compiler_version": version, "compiler_stat": _stat(config["nvcc"])}
    document = {"schema": SCHEMA, **identity, "id": _id(identity)}
    broker.mkdir(parents=True, exist_ok=True)
    with (broker / "init.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = broker / "config.json"
        if path.exists():
            if _read(path) != document:
                raise ValueError("broker identity changed; use a new broker directory")
        else:
            _atomic(path, document)
    for name in ("requests", "records", "locks"):
        (broker / name).mkdir(exist_ok=True)
    return document


def _document(broker):
    document = _read(Path(broker) / "config.json")
    if document.get("schema") != SCHEMA:
        raise ValueError("unsupported broker schema")
    identity = {key: document[key] for key in
                ("config", "sources_sha256", "compiler_version", "compiler_stat")}
    if _id(identity) != document.get("id"):
        raise ValueError("corrupt broker identity")
    return document


def _event(events, value):
    events = Path(events)
    events.parent.mkdir(parents=True, exist_ok=True)
    with events.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        handle.write(_canonical({"time_ns": time.time_ns(), **value}) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def runtime_request(broker, config, mapping, events, timeout=600):
    """Return a ready path; publish and record every unresolved request."""
    broker = Path(broker)
    identifier = None
    try:
        document = _document(broker)
        config = _config(config)
        if config != document["config"]:
            raise ValueError("runtime request differs from fixed broker configuration")
        if _stat(config["nvcc"]) != document["compiler_stat"]:
            raise ValueError("compiler changed since broker initialization")
        if not isinstance(mapping, dict):
            raise ValueError("mapping must be a JSON object")
        if not math.isfinite(float(timeout)) or float(timeout) < 0:
            raise ValueError("timeout must be finite and nonnegative")
        identifier = _id({"broker_id": document["id"], "mapping": mapping})
        request = {"id": identifier, "broker_id": document["id"],
                   "mapping": mapping, "timeout": float(timeout)}
        record_path = broker / "records" / (identifier + ".json")
        record = _read(record_path) if record_path.exists() else {}
        if record and (record.get("id") != identifier or
                       record.get("broker_id") != document["id"]):
            raise ValueError("record identity mismatch")
        if record and record.get("status") not in ("ready", "failed", "running"):
            raise ValueError("unknown broker record status")
        if record.get("status") == "ready":
            path = Path(record["path"])
            path.relative_to(Path(config["cache"]))
            if not path.is_file() or _stat(path) != record["module_stat"]:
                raise ValueError("ready module missing or changed")
            return path
        if not record or record.get("status") not in ("failed", "running"):
            _atomic(broker / "requests" / (identifier + ".json"), request)
        state = record.get("status", "pending")
        _event(events, {"kind": "compile_" + state, "request_id": identifier,
                        "broker_id": document["id"], "mapping": mapping,
                        "error": record.get("error", "module is not ready")})
        raise RuntimeError("compile broker request " + identifier + " is " + state)
    except RuntimeError:
        raise
    except Exception as error:
        _event(events, {"kind": "broker_error", "request_id": identifier,
                        "error": str(error)})
        raise


def _compiler(root):
    path = Path(root) / "scripts" / "compile_module.py"
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("semantic_broker_compiler", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _compile_one(broker, document, request_path, compiler, retry_failed):
    request = _read(request_path)
    identifier = request["id"]
    if identifier != request_path.stem or identifier != _id(
            {"broker_id": document["id"], "mapping": request["mapping"]}) or request["broker_id"] != document["id"]:
        raise ValueError("queued request identity mismatch")
    with (broker / "locks" / (identifier + ".lock")).open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        record_path = broker / "records" / (identifier + ".json")
        old = _read(record_path) if record_path.exists() else {}
        if old.get("status") == "ready" or (old.get("status") == "failed" and not retry_failed):
            return
        record = {"id": identifier, "broker_id": document["id"],
                  "status": "running", "started_ns": time.time_ns()}
        _atomic(record_path, record)
        try:
            config = document["config"]
            path = Path(compiler.compile_mapping(request["mapping"],
                root=Path(config["root"]), mathdx=Path(config["mathdx"]),
                nvcc=Path(config["nvcc"]), sm=config["sm"], cache=Path(config["cache"]),
                timeout=request["timeout"])).resolve()
            if not path.is_file():
                raise RuntimeError("compiler returned no module file")
            path.relative_to(Path(config["cache"]))
            record.update(status="ready", path=str(path), module_stat=_stat(path),
                          module_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        except Exception as error:
            record.update(status="failed", error=str(error) or type(error).__name__)
        record["finished_ns"] = time.time_ns()
        _atomic(record_path, record)


def status(broker):
    """Count queued requests by pending/running/ready/failed state."""
    broker = Path(broker)
    result = dict(pending=0, running=0, ready=0, failed=0)
    for path in sorted((broker / "requests").glob("*.json")):
        record_path = broker / "records" / path.name
        state = _read(record_path).get("status", "pending") if record_path.exists() else "pending"
        result[state] += 1
    return result


def _validated_worker(broker):
    document = _document(broker)
    config = document["config"]
    if _sources(config) != document["sources_sha256"] or _stat(config["nvcc"]) != document["compiler_stat"]:
        raise ValueError("broker source or compiler identity changed")
    return document, _compiler(config["root"])


def work(broker, jobs=1, retry_failed=False, *, _validated=None,
         _stop_requested=None, _heartbeat=None):
    """Drain the queue with bounded CPU workers; never load CUDA/library."""
    if jobs < 1:
        raise ValueError("jobs must be positive")
    broker = Path(broker)
    document, compiler = _validated or _validated_worker(broker)
    previous_cuda = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    stopping = False
    previous_signals = {}
    def stop(signum, frame):
        nonlocal stopping
        stopping = True
    def requested():
        return stopping or bool(_stop_requested and _stop_requested())
    try:
        if _stop_requested is None:
            for sig in (signal.SIGINT, signal.SIGTERM):
                previous_signals[sig] = signal.signal(sig, stop)
        seen, active = set(), set()
        with ThreadPoolExecutor(max_workers=jobs) as executor:
            while True:
                if _heartbeat:
                    _heartbeat()
                if not requested():
                    for path in sorted((broker / "requests").glob("*.json")):
                        if path in seen:
                            continue
                        if len(active) >= jobs:
                            break
                        seen.add(path)
                        record_path = broker / "records" / path.name
                        old = _read(record_path) if record_path.exists() else {}
                        if old.get("status") == "ready" or (old.get("status") == "failed" and not retry_failed):
                            continue
                        active.add(executor.submit(_compile_one, broker, document, path,
                                                   compiler, retry_failed))
                if not active:
                    break
                done, active = wait(active, timeout=0.25, return_when=FIRST_COMPLETED)
                for future in done:
                    future.result()
        return {**status(broker), "interrupted": requested()}
    finally:
        for sig, handler in previous_signals.items():
            signal.signal(sig, handler)
        if previous_cuda is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = previous_cuda


def _cache_directories(cache):
    with os.scandir(cache) as entries:
        for entry in entries:
            yield Path(entry.path)


def seed_cache_since(broker, library, timeout=0, *, _stop_requested=None):
    """Queue recent cache mappings only; no library load or old path trust."""
    broker, library = Path(broker), Path(library).resolve()
    document = _document(broker)
    config = document["config"]
    library_stat = _stat(library)
    source = {"library": str(library), "library_stat": library_stat,
              "broker_id": document["id"]}
    result = {"source": source, "recent_manifests": 0, "unique_mappings": 0,
              "queued": 0, "ready": 0, "failed": 0, "invalid_manifests": []}
    seen = set()
    cache = Path(config["cache"])
    if cache.is_dir():
        for directory in _cache_directories(cache):
            if _stop_requested and _stop_requested():
                result["interrupted"] = True
                break
            manifest_path = directory / "manifest.json"
            try:
                if not directory.is_dir() or not manifest_path.is_file():
                    continue
                if manifest_path.stat().st_mtime_ns < library_stat["mtime_ns"]:
                    continue
                result["recent_manifests"] += 1
                manifest = _read(manifest_path)
                if manifest.get("schema") != "cubutterfly-module-v1":
                    raise ValueError("unsupported cache manifest schema")
                if int(manifest.get("sm", 0)) != config["sm"]:
                    continue
                mapping = manifest["mapping"]
                if not isinstance(mapping, dict):
                    raise ValueError("cache mapping must be an object")
                identifier = _id(mapping)
                if identifier in seen:
                    continue
                seen.add(identifier)
                result["unique_mappings"] += 1
                if _stop_requested and _stop_requested():
                    result["interrupted"] = True
                    break
                try:
                    runtime_request(broker, config, mapping, broker / "seed_events.jsonl", timeout)
                    result["ready"] += 1
                except RuntimeError:
                    request_id = _id({"broker_id": document["id"], "mapping": mapping})
                    record_path = broker / "records" / (request_id + ".json")
                    record = _read(record_path) if record_path.exists() else {}
                    result["failed" if record.get("status") == "failed" else "queued"] += 1
            except (OSError, ValueError, KeyError, TypeError) as error:
                result["invalid_manifests"].append({"path": str(manifest_path), "error": str(error)})
    _atomic(broker / "cache_seed_status.json", result)
    return result


def watch(broker, jobs=1, retry_failed=False, poll_seconds=0.5, *,
          seed_library=None, seed_timeout=0):
    """Run a persistent CPU producer with heartbeat and graceful shutdown."""
    if jobs < 1 or not 0 < poll_seconds <= 1:
        raise ValueError("jobs must be positive and poll_seconds in (0, 1]")
    broker = Path(broker)
    validated = _validated_worker(broker)
    stopping = False
    started_ns = time.time_ns()
    last_heartbeat = 0.0
    seed_state = {"seed_status": "pending" if seed_library else "disabled", "seed_error": None}
    seed_thread = None
    def stop(signum, frame):
        nonlocal stopping
        stopping = True
    def heartbeat(state=None):
        nonlocal last_heartbeat
        now = time.monotonic()
        if state is None and now - last_heartbeat < 0.5:
            return
        last_heartbeat = now
        _atomic(broker / "producer_status.json", {
            "schema": SCHEMA, "broker_id": validated[0]["id"], "pid": os.getpid(),
            "started_ns": started_ns, "heartbeat_ns": time.time_ns(), "jobs": jobs,
            "status": state or ("stopping" if stopping else "running"),
            **seed_state.copy(), **status(broker)})
    def seed():
        seed_state["seed_status"] = "running"
        try:
            result = seed_cache_since(broker, seed_library, seed_timeout,
                                      _stop_requested=lambda: stopping)
            seed_state.update(seed_status="stopped" if result.get("interrupted") else "complete",
                              seed_invalid_manifests=len(result.get("invalid_manifests", [])))
        except Exception as error:
            seed_state.update(seed_status="failed", seed_error=str(error) or type(error).__name__)
    previous = {sig: signal.signal(sig, stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        heartbeat("running")
        if seed_library:
            seed_thread = threading.Thread(target=seed, name="semantic-cache-seed")
            seed_thread.start()
        while not stopping:
            work(broker, jobs, retry_failed, _validated=validated,
                 _stop_requested=lambda: stopping, _heartbeat=heartbeat)
            retry_failed = False
            heartbeat()
            if not stopping:
                time.sleep(poll_seconds)
        if seed_thread:
            seed_thread.join()
        heartbeat("stopped")
        return {**status(broker), "interrupted": True}
    except BaseException:
        stopping = True
        if seed_thread:
            seed_thread.join()
        heartbeat("failed")
        raise
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("compiler_script", nargs="?")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--init", action="store_true")
    mode.add_argument("--worker", action="store_true")
    mode.add_argument("--watch", action="store_true")
    mode.add_argument("--status", action="store_true")
    parser.add_argument("--broker", type=Path)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--seed-cache-since", type=Path)
    for field in CONFIG_FIELDS:
        parser.add_argument("--" + field, type=int if field == "sm" else str)
    parser.add_argument("--mapping")
    parser.add_argument("--timeout", type=float, default=600)
    args = parser.parse_args(argv)
    broker = args.broker or os.environ.get("CUBUTTERFLY_PREPARE_BROKER")
    events = os.environ.get("CUBUTTERFLY_PREPARE_EVENTS")
    handed_off = False
    try:
        if not broker:
            raise ValueError("missing CUBUTTERFLY_PREPARE_BROKER or --broker")
        if args.status:
            print(_canonical(status(broker)))
            return 0
        if args.seed_cache_since and not (args.worker or args.watch):
            raise ValueError("--seed-cache-since requires --worker or --watch")
        if args.worker or args.watch:
            if args.seed_cache_since and args.worker:
                print(_canonical(seed_cache_since(broker, args.seed_cache_since, args.timeout)), file=sys.stderr)
            if args.watch:
                result = watch(broker, args.jobs, args.retry_failed,
                               seed_library=args.seed_cache_since, seed_timeout=args.timeout)
            else:
                result = work(broker, args.jobs, args.retry_failed)
            print(_canonical(result))
            return 130 if result["interrupted"] else int(bool(result["failed"] or result["running"] or result["pending"]))
        config = {key: getattr(args, key) for key in CONFIG_FIELDS}
        if any(value is None for value in config.values()):
            raise ValueError("root, mathdx, nvcc, sm and cache are required")
        if args.init:
            config["nvcc"] = shutil.which(config["nvcc"]) or config["nvcc"]
            print(_canonical(initialize(broker, config)))
            return 0
        if not events:
            raise ValueError("CUBUTTERFLY_PREPARE_EVENTS is required for runtime requests")
        expected = Path(config["root"]) / "scripts" / "compile_module.py"
        if not args.compiler_script or Path(args.compiler_script).resolve() != expected.resolve():
            raise ValueError("runtime compiler script does not match broker root")
        mapping = json.loads(args.mapping)
        handed_off = True
        print(runtime_request(broker, config, mapping, events, args.timeout))
        return 0
    except Exception as error:
        if events and not handed_off and not (args.init or args.worker or args.watch or args.status):
            _event(events, {"kind": "broker_runtime_error", "error": str(error)})
        print("semantic compile broker: " + str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
