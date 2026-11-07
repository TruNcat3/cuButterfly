#!/usr/bin/env python3
"""Hardware calibration and model-guided search of library-exported mappings.

All six mathematical operators use the linked library's candidate inventory.
Coverage records retain budget omissions, failures and confirmed finalists.
"""

from __future__ import annotations

import argparse
import copy
import csv
import datetime as dt
import io
import hashlib
import json
import pathlib
import re
import statistics
import subprocess
import sys
import time
import os
import tempfile
import math
import heapq
import gzip
from typing import Any
from hardware_registry import promote, default_path, canonical_semantics

from calibration_space import (stratified_candidates, candidate_id, command_for, runtime_candidates,
                               predicted_sample, resolved_tile_threads, PROJECTION_VERSION)
from calibration_seeds import seed_snapshot, candidates_for_workload, validate_mapping
from fit_local_cost_model import fit, predict, design_point_key, FEATURE_VERSION
import stage_cost_model
import static_hardware_profile
from initialize_hardware_profile import bind_device_identity
from schedule_cost_model import calibrate as calibrate_scheduling
from pipeline_schedule_model import calibrate as calibrate_pipeline_scheduling
from stage_service_calibration import calibrate as calibrate_stage_services
from local_selector_data import write_header as write_local_selector_header, semantic_key
from run_comprehensive_suite import require_exclusive_gpu


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if not (ROOT / "CMakeLists.txt").exists():
    ROOT = ROOT / "share/cuButterfly"
PROFILE_SCRIPT = SCRIPT_DIR / "initialize_hardware_profile.py"
CHECK_SCRIPT = SCRIPT_DIR / "check_hardware_profile.py"
COST_MODEL_SCRIPT = SCRIPT_DIR / "fit_local_cost_model.py"
MODEL_UPDATE_POLICY_VERSION = "correct-training-change-v1"
SEARCH_SELECTION_VERSION = "resolved-feasibility-and-distinct-finalists-v1"


class _SearchPredictionCache:
    """Cache invariant candidate work; a resolved descriptor invalidates it.

    Candidate dictionaries are owned by the search and are immutable. A new
    evolutionary proposal is a new object. Model fits create new model objects;
    binding one clears scores but keeps fixed features for the same profile.
    """
    def __init__(self, descriptors, project, persistent=None, *, persist_frontier=True):
        self.descriptors = descriptors
        self.project = project
        self.identities = {}
        self.prepared = {}
        self.scores = {}
        self.predictor = None
        self.model = None
        self.persistent = persistent
        self.persist_frontier = persist_frontier
        self.descriptor_versions = {}
        self.stats = dict(score_requests=0, score_cache_hits=0, feature_preparations=0,
                          feature_seconds=0.0, prediction_seconds=0.0, disk_score_hits=0,
                          persisted_scores=0)

    def identity(self, point):
        item = self.identities.get(id(point))
        if item is None or item[0] is not point:
            item = (point, candidate_id(point))
            self.identities[id(point)] = item
        return item[1]

    def bind(self, model):
        if model is self.model:
            return
        if self.predictor is None:
            self.predictor = stage_cost_model.PreparedPredictor(model)
        self.predict = self.predictor.bind(model)
        self.model = model
        self.scores.clear()
        if self.persistent is not None:
            self.persistent.bind(model)

    def flush(self):
        if self.persistent is not None:
            self.persistent.flush()

    def persist(self, points):
        """Checkpoint useful scores without writing each transient frontier.

        Every candidate is still evaluated exactly. Unpersisted scores can be
        recomputed after interruption; measurement journals remain independent.
        """
        if self.persistent is not None:
            for point in points:
                score = self.score(point)
                key = self.identity(point)
                self.persistent.put(key, self.descriptors.get(key), score)
                self.stats['persisted_scores'] += 1
            self.persistent.flush()

    def score(self, point):
        self.stats['score_requests'] += 1
        key = self.identity(point)
        descriptor = self.descriptors.get(key)
        # In-place descriptor updates invalidate features in both disk-backed
        # and memory-only searches. Missing descriptors need no stored token.
        version = (json.dumps(descriptor, sort_keys=True, separators=(",", ":"), allow_nan=False)
                   if descriptor is not None else None)
        if self.descriptor_versions.get(key) != version:
            self.prepared.pop(key, None)
            self.scores.pop(key, None)
            if version is None:
                self.descriptor_versions.pop(key, None)
            else:
                self.descriptor_versions[key] = version
        previous = self.prepared.get(key)
        if previous is not None and previous[0] is not descriptor:
            self.prepared.pop(key)
            self.scores.pop(key, None)
            previous = None
        cached_score = self.scores.get(key)
        if cached_score is not None and cached_score[0] is descriptor:
            self.stats['score_cache_hits'] += 1
            return cached_score[1]
        if self.persistent is not None:
            found, result = self.persistent.get(key, descriptor)
            if found:
                self.stats['disk_score_hits'] += 1
                self.scores[key] = (descriptor, result)
                return result
        if descriptor and descriptor.get('status') == 'unavailable':
            result = float('inf')
        else:
            try:
                if previous is None:
                    started = time.perf_counter()
                    previous = (descriptor, self.predictor.prepare(self.project(point)))
                    self.prepared[key] = previous
                    self.stats['feature_preparations'] += 1
                    self.stats['feature_seconds'] += time.perf_counter() - started
                started = time.perf_counter()
                result = self.predict(previous[1])
                self.stats['prediction_seconds'] += time.perf_counter() - started
            except (TypeError, ValueError, KeyError, ZeroDivisionError, OverflowError, IndexError):
                result = float('inf')
        self.scores[key] = (descriptor, result)
        if self.persistent is not None and self.persist_frontier:
            self.persistent.put(key, descriptor, result)
            self.stats['persisted_scores'] += 1
        return result


def cpu_preparation_context(args, measurement):
    """Content identities for exact CPU reuse; GPU timing policy stays separate."""
    names = ('calibrate_local_hardware.py', 'calibration_space.py', 'stage_cost_model.py',
             'stage_service_model.py', 'stage_composition.py', 'fit_local_cost_model.py',
             'schedule_cost_model.py', 'pipeline_schedule_model.py', 'stage_schedule_native.py',
             'stage_schedule_native.cpp', 'calibration_cpu_cache.py', 'hardware_registry.py',
             'resident_mapping.py', 'benchmark_semantics.py', 'shared_work_geometry.py')
    return dict(schema='cubutterfly-calibration-cpu-context-v1', measurement=measurement,
                partition_budget=int(os.environ.get('CUBUTTERFLY_PARTITION_BUDGET', '128')),
                projection_scope=getattr(args, 'search_stage_projection', 'full'),
                sources={name:_sha256_file(SCRIPT_DIR / name) for name in names})


def load_cpu_preparation(args, measurement):
    source = getattr(args, 'cpu_preparation_input', None)
    if source is None:
        return None
    source = pathlib.Path(source).resolve()
    manifest = json.loads(source.read_text())
    if (manifest.get('schema') != 'cubutterfly-calibration-cpu-input-v1'
            or manifest.get('context') != cpu_preparation_context(args, measurement)):
        raise ValueError('CPU preparation build, sources or projection identity changed')
    if getattr(args, 'cpu_preparation_inventory_only', False):
        # Captured inventories remain useful when feedback or CPU source
        # changes invalidate score epochs. Inventory hashes are still checked.
        return source.parent, manifest
    status_path = pathlib.Path(args.cpu_preparation_dir) / 'status.json'
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    if (not status.get('scoring_complete')
            or status.get('input_sha256') != _sha256_file(source)):
        raise ValueError('CPU preparation has not published complete matching scores yet')
    if not (pathlib.Path(args.cpu_preparation_dir) / 'score_cache/scores-v1.sqlite3').is_file():
        raise ValueError('CPU preparation score database is missing')
    return source.parent, manifest


def prepared_inventory(preparation, workload):
    root, manifest = preparation
    cells = [cell for cell in manifest['cells']
             if _search_workload_key(cell['workload']) == _search_workload_key(workload)]
    if len(cells) != 1 or cells[0].get('complete'):
        raise ValueError('CPU preparation does not contain the pending workload inventory')
    reference = cells[0]['inventory']
    path = (root / reference['path']).resolve(strict=True)
    if not path.is_relative_to(root) or _sha256_file(path) != reference['sha256']:
        raise ValueError('CPU inventory path or digest mismatch')
    with gzip.open(path, 'rt') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate cuButterfly for the current GPU during installation.")
    parser.add_argument("--build-dir", type=pathlib.Path, required=True)
    parser.add_argument("--profile-dir", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path)
    parser.add_argument("--trials", type=int, default=5, help="hardware capability probe trials")
    parser.add_argument("--operator-trials", type=int, default=3)
    parser.add_argument("--operator-warmup", type=int, default=100)
    parser.add_argument("--operator-repeat", type=int, default=100)
    parser.add_argument("--verify-batches", type=int, default=0,
                        help="check this many evenly spaced transforms within the actual timed batch; 0 checks all")
    parser.add_argument("--skip-operator-calibration", action="store_true")
    parser.add_argument("--search-only", action="store_true",
                        help="omit legacy smoke cells and calibrate only the configured search workloads")
    parser.add_argument("--search-workloads", type=pathlib.Path, default=ROOT / "config/install_search_workloads.json")
    parser.add_argument("--search-budget", type=int, default=64,
                        help="screened library mappings per workload; 0 exhausts the exported inventory (64 by default)")
    parser.add_argument("--search-finalists", type=int, default=3)
    parser.add_argument("--cost-model", choices=("staged", "legacy"), default="staged")
    parser.add_argument("--stage-calibration", choices=("full", "bounded", "skip"), default="full",
                        help="independent stage service calibration; full is required for complete migration")
    parser.add_argument("--stage-point-budget", type=int, default=0,
                        help="maximum distinct executable stage templates to describe (0 keeps the full inventory)")
    parser.add_argument("--stage-import-checkpoint", type=pathlib.Path, action="append", default=[],
                        help="reuse strictly compatible independent-stage measurements")
    parser.add_argument("--search-strategy", choices=("model", "evolutionary"), default="model")
    parser.add_argument("--search-stage-projection", choices=("full", "plan"), default="full",
                        help="use independent stage-service composition during search, or only resolved plan groups")
    parser.add_argument("--mapping-seeds", type=pathlib.Path, action="append", default=[],
                        help="mapping-only seed file or registry; repeat to add sources alongside the local registry")
    parser.add_argument("--seed-registry", type=pathlib.Path,
                        help="fixed registry source for mapping seeds; independent of the output registry during recovery")
    parser.add_argument("--seed-budget", type=int, default=8,
                        help="historical mappings revalidated per workload before the exploration budget; 0 disables seeds")
    parser.add_argument("--seed-only", action="store_true",
                        help="revalidate only matching mapping seeds within seed-budget; omit runtime inventory and exploration")
    parser.add_argument('--low-fft-error-report-only', action='store_true',
                        help='report low-precision FFT error against the same-precision CPU reference without the legacy threshold')
    parser.add_argument("--search-seconds", type=float,
                        help="total search wall-time budget; 0 for unrestricted research")
    parser.add_argument("--compile-seconds", type=float,
                        help="additional specialization compilation budget, independent of search (0 = unrestricted)")
    parser.add_argument("--resume-search", action="store_true",
                        help="reuse matching search measurements only when the benchmark binary hash is unchanged")
    parser.add_argument("--retry-compilation-gaps", action="store_true",
                        help="on resume, retry unmeasured compilation gaps, retaining successful screens and selection order")
    parser.add_argument("--resume-build-priors", action="store_true",
                        help="development resume: preserve older-build screens as labeled priors; confirm finalists on current build")
    parser.add_argument('--cpu-preparation-input', type=pathlib.Path,
                        help='Frozen candidate inventory from capture_calibration_cpu_inputs.py')
    parser.add_argument('--cpu-preparation-dir', type=pathlib.Path,
                        help='Independent CPU preparation output, including persistent exact scores')
    parser.add_argument('--cpu-preparation-inventory-only', action='store_true',
                        help='Reuse frozen candidates and compiled modules; recompute scores in memory without a disk score cache')
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--embed-legacy-selector", action="store_true",
                        help="also generate the legacy build-time header; the dynamic registry needs no rebuild")
    args = parser.parse_args()
    if args.seed_only:
        if args.seed_budget <= 0:
            parser.error('--seed-only requires a positive --seed-budget; 0 disables seeds')
        if args.skip_operator_calibration:
            parser.error('--seed-only conflicts with --skip-operator-calibration')
        if args.cpu_preparation_input or args.cpu_preparation_dir or args.cpu_preparation_inventory_only:
            parser.error('--seed-only conflicts with CPU preparation inventory options')
    if args.resume_build_priors and not args.resume_search:
        parser.error("--resume-build-priors requires --resume-search")
    if args.cpu_preparation_inventory_only:
        if not args.cpu_preparation_input or args.cpu_preparation_dir:
            parser.error('inventory-only requires --cpu-preparation-input without --cpu-preparation-dir')
    elif bool(args.cpu_preparation_input) != bool(args.cpu_preparation_dir):
        parser.error('--cpu-preparation-input and --cpu-preparation-dir must be provided together')
    if args.search_seconds is None:
        args.search_seconds = 0 if args.stage_calibration == "full" else 300
    if args.compile_seconds is None:
        args.compile_seconds = 0 if args.stage_calibration == "full" else 600
    return args


def build_python(build_dir: pathlib.Path) -> str:
    cache = build_dir / "CMakeCache.txt"
    if cache.exists():
        match = re.search(r"^Python3_EXECUTABLE:[^=]*=(.*)$", cache.read_text(), re.MULTILINE)
        if match and pathlib.Path(match.group(1)).is_file():
            return match.group(1)
    return sys.executable


def run(command: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    if check and completed.returncode:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"command failed ({completed.returncode}): {' '.join(command)}\n{detail}")
    return completed


def write_checkpoint(path, value):
    temporary=path.with_suffix(path.suffix+".tmp")
    # Full coverage remains a standalone JSON document with every candidate,
    # model update and selection. Whitespace alone added hundreds of MB to
    # each atomic rewrite of a large search checkpoint.
    compact = isinstance(value, dict) and value.get('schema') == 'cubutterfly-search-coverage-v1'
    temporary.write_text(json.dumps(value, separators=(',', ':'))+'\n' if compact
                         else json.dumps(value,indent=2)+'\n')
    temporary.replace(path)


def ensure_static_hardware_model(profile, identity, build_dir, profile_dir, output_dir, *, refresh=False):
    """Persist current caps independently of optional performance calibration."""
    bound_profile = bind_device_identity(profile, identity)
    expected = {"device": identity["device"], "compute_capability": identity["compute_capability"],
                "global_memory_bytes": int(identity["global_memory_bytes"])}
    artifact = profile_dir / static_hardware_profile.FILENAME
    capture = None
    if artifact.exists() and not refresh:
        model = static_hardware_profile.load(artifact, expected)
    else:
        probe = build_dir / "cubutterfly_plan_probe"
        command = [str(probe), "--hardware-live"]
        raw = json.loads(run(command).stdout)
        if raw.get("resource_source") != "live-cuda-properties:explicit-hardware-probe":
            raise ValueError("hardware refresh requires an explicit live hardware probe, not a static model")
        hardware = {key: raw[key] for key in static_hardware_profile.FIELDS}
        capture = dict(command=command, resource_source=raw["resource_source"],
                       created_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                       probe_sha256=hashlib.sha256(probe.read_bytes()).hexdigest())
        model = static_hardware_profile.validate(dict(schema=static_hardware_profile.SCHEMA,
                                                      hardware=hardware, provenance=capture), expected)
        snapshot = profile_dir / "hardware_probe.json"
        write_checkpoint(snapshot, dict(hardware=hardware, provenance=capture))
        bound_profile["provenance"] = {**profile.get("provenance", {}), "hardware_capture": capture}
    profile.update(bound_profile)
    profile["hardware"] = model["hardware"]
    write_checkpoint(output_dir / "calibration_device.json", profile)
    if capture is not None:
        # Export from the current calibration identity, not an older profile's
        # possibly incomplete caps; hash the explicit live probe as evidence.
        model = static_hardware_profile.export_from_calibration(output_dir, artifact,
            dict(hardware=model["hardware"], provenance=dict(verified=True,
                sources=[str(snapshot)], **capture)), refresh=refresh)
        static_hardware_profile.validate(model, expected)
    # The runtime ranker resolves profiles beside hardware_model.json, which
    # need not be output_dir. Persist the same verified identity there before
    # any benchmark can start automatic candidate selection.
    write_checkpoint(profile_dir / "hardware_profile.json", profile)
    return model


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class _SharedLibraryGuard:
    """Track a shared library without hashing it when its stat identity is stable."""

    def __init__(self, build_dir: pathlib.Path):
        self.path = build_dir / "libcuntt.so"
        self._signature = None
        self._digest = None

    def digest(self) -> str | None:
        try:
            stat = self.path.stat()
        except FileNotFoundError:
            signature = None
        else:
            signature = (stat.st_dev, stat.st_ino, stat.st_size,
                         stat.st_mtime_ns, stat.st_ctime_ns)
        if signature != self._signature:
            self._signature = signature
            self._digest = _sha256_file(self.path) if signature is not None else None
        return self._digest

    def assert_matches(self, expected: str | None, phase: str) -> None:
        actual = self.digest()
        if actual != expected:
            raise RuntimeError(
                f"libcuntt.so changed {phase} benchmark; refusing to mix timing identities")


def effective_compile_mode(build_dir):
    value = os.environ.get("CUBUTTERFLY_COMPILE_MODE")
    if value is None:
        header = pathlib.Path(build_dir) / "generated/jit_config.hpp"
        if not header.is_file():
            return "build-default-unknown"
        match = re.search(r'CUBUTTERFLY_DEFAULT_CODEGEN_MODE\s+"([^\"]+)"', header.read_text())
        if not match:
            raise ValueError("build has no compile-policy identity")
        value = match.group(1)
    if value not in ("precompiled", "auto", "on-demand", "research"):
        raise ValueError("CUBUTTERFLY_COMPILE_MODE must be precompiled, auto, on-demand, or research")
    return value


def measurement_identity(args, library_guard: _SharedLibraryGuard | None = None):
    identity = dict(binary_sha256={name:hashlib.sha256((args.build_dir/name).read_bytes()).hexdigest()
                    for name in ("cubutterfly_bench","cuntt_bench")},
                    compile_mode=effective_compile_mode(args.build_dir),
                    verify_batches=getattr(args,"verify_batches",0))
    if getattr(args, 'seed_only', False):
        identity['seed_only'] = True
    if getattr(args, 'low_fft_error_report_only', False):
        identity['low_fft_error_report_only'] = True
    platform_rules = os.environ.get("CUBUTTERFLY_PLATFORM_TUNING")
    if platform_rules:
        identity["platform_tuning_sha256"] = hashlib.sha256(pathlib.Path(platform_rules).read_bytes()).hexdigest()
    library_sha256 = (library_guard or _SharedLibraryGuard(args.build_dir)).digest()
    # Static builds intentionally retain the historical identity shape.
    if library_sha256 is not None:
        identity["library_sha256"] = library_sha256
    from jit_calibration_identity import calibration_identity
    jit_runtime = calibration_identity()
    if jit_runtime is not None:
        identity['jit_runtime'] = jit_runtime
    return identity


def _validate_build_prior_identity(previous, current, previous_protocol=None, current_protocol=None):
    """Only build provenance may differ; measurement contracts remain strict."""
    build_fields = {"binary_sha256", "library_sha256", "jit_runtime"}
    old_contract = {key: value for key, value in previous.items() if key not in build_fields}
    new_contract = {key: value for key, value in current.items() if key not in build_fields}
    if old_contract != new_contract:
        raise ValueError("development build priors require unchanged compile policy and verification coverage")
    if previous_protocol is not None:
        for key in ("operator_trials", "operator_warmup", "operator_repeat"):
            if previous_protocol.get(key) != current_protocol.get(key):
                raise ValueError("development build priors require unchanged timing protocol")


def _archive_build_priors(paths, identity):
    """Keep immutable source documents; never relabel their measured hashes."""
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    directory = paths[0].parent / "build_priors" / digest
    directory.mkdir(parents=True, exist_ok=True)
    sources = []
    for path in paths:
        if not path.exists():
            continue
        content = path.read_bytes()
        sha = hashlib.sha256(content).hexdigest()
        destination = directory / (sha + "-" + path.name)
        if not destination.exists():
            destination.write_bytes(content)
        sources.append({"path": str(destination), "sha256": sha})
    return {"origin_identity": copy.deepcopy(identity), "sources": sources,
            "use": "development-search-prior-only; current-build finalist confirmation required"}


def _build_prior_row(row, identity):
    prior = copy.deepcopy(row)
    prior.setdefault("origin_identity", copy.deepcopy(identity))
    prior["measurement_role"] = "development-build-prior"
    if prior.get("correct") and prior.get("samples"):
        prior["status"] = "screened"
    return prior


def _stage_service_usable_for_ranking(profile: dict[str, Any] | None) -> bool:
    """Return whether empirical stage curves are eligible to rank new plans.

    Validation and inventory coverage are separate. A validated sampled scope
    may rank queries covered by its curves; the per-query service matcher still
    rejects missing keys and unsupported extrapolation. An unvalidated curve
    table cannot become an empirical prior simply because samples exist.
    """
    if not isinstance(profile, dict) or not isinstance(profile.get("stage_service"), dict):
        return False
    service = profile["stage_service"]
    if not service.get("curves"):
        return False
    complete = {"complete", "complete-bounded"}
    complete_scope = bool(
        (service.get("validated") is True and service.get("calibration_status") in complete)
        or (service.get("selected_scope_validated") is True
            and service.get("selected_scope_status") in complete
            and service.get("calibration_status") == "complete-sampled"))
    scope = service.get('ranking_scope', {})
    keys = scope.get('validated_curve_keys', [])
    certified_scope = (scope.get('schema') == 'cubutterfly-stage-ranking-scope-v1'
                       and isinstance(keys, list) and bool(keys)
                       and len(set(keys)) == len(keys) == scope.get('validated_curve_count')
                       and all(key in service['curves'] for key in keys))
    return complete_scope or certified_scope


def rank_model(profile: dict[str, Any], output: pathlib.Path) -> int:
    capabilities = profile["capabilities"]
    records: list[dict[str, Any]] = []
    for log_n in (8, 12, 16, 20):
        for budget in (32, 64, 128, 256):
            for word_bytes in (4, 8):
                for us in range(1, log_n + 1):
                    if budget % us:
                        continue
                    ud = budget // us
                    ts = (log_n + us - 1) // us
                    td = (((1 << log_n) // 2) + ud - 1) // ud
                    cells = us * ud
                    rates = {
                        "compute": capabilities["equivalent_butterflies_per_second"] / cells,
                        "boundary": capabilities["global_feedback_bytes_per_second"] / (4 * ud * word_bytes),
                    }
                    if us > 1:
                        rates["interstage"] = capabilities["interstage_shared_bytes_per_second"] / (4 * ud * (us - 1) * word_bytes)
                        rates["synchronization"] = capabilities["cta_barriers_per_second"] / (us - 1)
                    bottleneck = min(rates, key=rates.get)
                    records.append({
                        "device": profile["device"],
                        "logN": log_n,
                        "word_bytes": word_bytes,
                        "spatial_budget_C": budget,
                        "stage_space_Us": us,
                        "data_space_Ud": ud,
                        "stage_time_Ts": ts,
                        "data_time_Td": td,
                        "limiting_capability": bottleneck,
                        "calibrated_body_us": (ts * td) / rates[bottleneck] * 1.0e6,
                    })
    records.sort(key=lambda row: (row["logN"], row["word_bytes"], row["spatial_budget_C"], row["calibrated_body_us"]))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    return len(records)


def target_batch(log_n: int) -> int:
    return max(1, (1 << 22) // (1 << log_n))


def candidate_commands(build_dir: pathlib.Path, cufftdx: bool) -> list[tuple[str, list[str]]]:
    binary = build_dir / "cubutterfly_bench"
    common = ["--precision", "fp32", "--placement", "out-of-place", "--normalization", "none"]
    candidates: list[tuple[str, list[str]]] = []

    def add(name: str, args: list[str]) -> None:
        candidates.append((name, [str(binary), *args]))

    for log_n in (8, 12, 16):
        batch = target_batch(log_n)
        add(f"fft-log{log_n}-cufft", ["--operator", "fft", "--backend", "cufft", "--logN", str(log_n), "--batch", str(batch), *common])
    add("fft-log8-cta-dft8", ["--operator", "fft", "--backend", "temporal-tile", "--fft-core", "cta-dft8", "--compute-unit", "radix8", "--tile-threads", "128", "--logN", "8", "--batch", str(target_batch(8)), *common])
    add("fft-log12-scalar-hierarchical", ["--operator", "fft", "--backend", "hierarchical", "--fft-core", "scalar", "--compute-unit", "radix4", "--local-stages", "8", "--tile-threads", "256", "--logN", "12", "--batch", str(target_batch(12)), *common])
    add("fft-log12-shared-temporal-radix4", ["--operator", "fft", "--backend", "temporal-tile", "--fft-core", "scalar", "--compute-unit", "radix4", "--tile-threads", "256", "--logN", "12", "--batch", str(target_batch(12)), *common])
    if cufftdx:
        add("fft-log12-cufftdx-direct", ["--operator", "fft", "--backend", "temporal-tile", "--fft-core", "cufftdx-direct", "--tile-threads", "1024", "--logN", "12", "--batch", str(target_batch(12)), *common])
        add("fft-log12-cufftdx-direct512", ["--operator", "fft", "--backend", "temporal-tile", "--fft-core", "cufftdx-direct", "--tile-threads", "512", "--logN", "12", "--batch", str(target_batch(12)), *common])
        add("fft-log12-cufftdx-online", ["--operator", "fft", "--backend", "online-reorder", "--fft-core", "cufftdx-block", "--local-stages", "6", "--prefix-threads", "512", "--suffix-threads", "512", "--prefix-ept", "8", "--suffix-ept", "8", "--reorder-columns", "1", "--logN", "12", "--batch", str(target_batch(12)), *common])
        add("fft-log12-cufftdx-resident", ["--operator", "fft", "--backend", "online-reorder", "--fft-core", "cufftdx-resident", "--local-stages", "6", "--tile-threads", "512", "--reorder-columns", "1", "--logN", "12", "--batch", str(target_batch(12)), *common])
    add("fwht-log15-warp-register", ["--operator", "fwht", "--backend", "temporal-tile", "--local-exchange", "warp-register", "--compute-unit", "radix2", "--tile-threads", "256", "--logN", "15", "--batch", str(target_batch(15)), *common])
    add("subset-zeta-log12-hierarchical", ["--operator", "subset-zeta", "--backend", "hierarchical", "--compute-unit", "radix4", "--local-stages", "8", "--tile-threads", "256", "--logN", "12", "--batch", str(target_batch(12)), *common])
    add("superset-zeta-log12-hierarchical", ["--operator", "superset-zeta", "--backend", "hierarchical", "--compute-unit", "radix4", "--local-stages", "8", "--tile-threads", "256", "--logN", "12", "--batch", str(target_batch(12)), *common])
    add("xor-zeta-log12-hierarchical", ["--operator", "xor-zeta", "--backend", "hierarchical", "--compute-unit", "radix4", "--local-stages", "8", "--tile-threads", "256", "--logN", "12", "--batch", str(target_batch(12)), *common])
    add("structured-2x2-log12-hierarchical", ["--operator", "structured-2x2", "--backend", "hierarchical", "--compute-unit", "radix4", "--local-stages", "8", "--tile-threads", "256", "--stage-matrix", "0.9238795,-0.3826834,0.3826834,0.9238795", "--logN", "12", "--batch", str(target_batch(12)), *common])
    return candidates


def ntt_candidate_commands(build_dir: pathlib.Path) -> list[tuple[str, list[str]]]:
    binary = build_dir / "cuntt_bench"
    return [
        ("ntt-log12-hybrid2d-radix4-word32", [str(binary), "--backend", "hybrid2d", "--compute-unit", "radix4", "--cross-twiddle", "fused", "--word-bits", "32", "--output-order", "natural", "--modulus", "998244353", "--logN", "12", "--batch", str(target_batch(12))]),
        ("ntt-log16-hybrid2d-radix4-word64", [str(binary), "--backend", "hybrid2d", "--compute-unit", "radix4", "--cross-twiddle", "fused", "--word-bits", "64", "--output-order", "natural", "--modulus", "576460756061519873", "--logN", "16", "--batch", str(target_batch(16))]),
    ]


def _stage_template_key(point: dict[str, Any]) -> tuple[Any, ...]:
    """Return the physical template axes that determine stage service.

    The exported mapping inventory also contains independent policy axes such
    as factor overlap and overlap batch chunks. Those are composed after the
    physical stage curves are learned. Shared bulk CTA batch width instead
    changes the physical template and must remain distinct.
    """
    try:
        raw_mapping = point.get("mapping_json", {})
        mapping = json.loads(raw_mapping) if isinstance(raw_mapping, str) else raw_mapping
    except (TypeError, ValueError):
        mapping = {}
    if not isinstance(mapping, dict):
        mapping = {}
    stage_partition = mapping.get("stage_partition", point.get("stage_partition", []))
    if isinstance(stage_partition, str):
        try:
            stage_partition = json.loads(stage_partition)
        except (TypeError, ValueError):
            stage_partition = [stage_partition]
    axes = {**point, **mapping}
    if axes.get("suffix_core") == "cufftdx-block":
        axes.pop("suffix_core")
    if axes.get("prefix_twiddle") == "runtime":
        axes.pop("prefix_twiddle")
    shared = axes.get("backend") == "shared-iterative"
    if shared:
        from research_compile_requests import _shared_partition
        stage_partition = _shared_partition(axes, point.get("operator", "fft"),
                                            int(point.get("logN", point.get("log_n"))))
    local_partitions = mapping.get("local_stage_partitions", point.get("local_stage_partitions", []))
    if isinstance(local_partitions, str):
        try:
            local_partitions = json.loads(local_partitions)
        except (TypeError, ValueError):
            local_partitions = [local_partitions]
    def value(name, default=None):
        return axes.get(name, default)

    def frozen(raw):
        if isinstance(raw, (dict, list, tuple)):
            return json.dumps(raw, sort_keys=True, separators=(",", ":"), default=str)
        return raw

    # Keep semantic contracts and every group-local implementation axis in the
    # bounded sampler identity.  Omitting one of these fields would let a
    # bounded run calibrate (for example) native and cooperative prefixes, or
    # dynamic and static factor I/O, under one representative curve.
    semantic = tuple((name, frozen(value(name))) for name in (
        "direction", "normalization", "placement", "modulus", "stage_matrix",
        "stage_matrices", "input_order", "output_order", "element_stride", "batch_stride",
    ) if value(name) is not None)
    physical = tuple((name, frozen(value(name))) for name in (
        "backend", "fft_core", "compute_unit", "local_exchange", "shared_layout",
        "cross_twiddle", "direct_boundary", "reorder_columns", "prefix_codelet",
        "prefix_shared_layout", "prefix_twiddle", "prefix_codelet_lanes", "factor_io_policies",
        "suffix_core",
        "factor_ept", "factor_columns", "data_tiles_per_cta", "prefetch_depth",
        "factor_partition", "exchange_chunks",
        "register_threads", "tile_threads", "resolved_tile_threads",
    ) if value(name) is not None)
    return (
        point.get("operator", "fft"), point.get("precision", "fp32"), point.get("logN", point.get("log_n")),
        semantic, physical,
        frozen(stage_partition),
        frozen(local_partitions),
        int(value("register_threads", 0) or 0),
        resolved_tile_threads({**point, **mapping}),
        bool(value("stage_overlap", False)),
        int(value("stage_space", 0) or 0),
        int(value("factor_slices", 1) or 1),
        int(value("batch_tile_count", 1) or 1) if shared and not value("stage_overlap", False) else 1,
    )


def _bound_stage_points(points: list[dict[str, Any]], budget: int) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Bound stage inventory expansion with an explicit stratified sample.

    A runtime mapping inventory enumerates policy combinations far beyond the
    physical execution groups used by the stage cost model.  Keep one stable
    representative per physical template, then fill the requested budget by a
    stable hash order.  The sampler is recorded separately so omitted policy
    combinations remain visible in the migration audit.  A bounded selection is
    a representative sample, not physically equivalent coverage of the omitted
    policy combinations.
    """
    requested = len(points)
    if budget <= 0 or requested <= budget:
        return points, {"requested": requested, "template_count": len({_stage_template_key(p) for p in points}),
                        "selected": requested, "omitted": 0}
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for point in points:
        groups.setdefault(_stage_template_key(point), []).append(point)

    def stable_id(point: dict[str, Any]) -> str:
        payload = json.dumps(point, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    representatives = [min(bucket, key=stable_id) for bucket in groups.values()]
    representatives.sort(key=stable_id)
    # Anchor every semantic workload stratum before filling the remainder.
    # This prevents a small budget from accidentally selecting only FFT
    # templates when the installation workload file also contains FWHT/NTT
    # and finite-field transforms.
    strata: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for point in representatives:
        stratum = (point.get("operator", "fft"), point.get("precision", "fp32"),
                   point.get("logN", point.get("log_n")))
        strata.setdefault(stratum, []).append(point)
    anchors = [min(bucket, key=stable_id) for bucket in strata.values()]
    anchors.sort(key=stable_id)
    selected = anchors[:budget]
    selected_ids = {id(point) for point in selected}
    selected.extend(point for point in representatives
                    if id(point) not in selected_ids and len(selected) < budget)
    selected_ids = {id(point) for point in selected}
    if len(selected) < budget:
        remainder = sorted((point for point in points if id(point) not in selected_ids), key=stable_id)
        selected.extend(remainder[:budget - len(selected)])
    return selected, {"requested": requested, "template_count": len(groups),
                      "selected": len(selected), "omitted": requested - len(selected)}


def _stage_sampling_scope(sampling: dict[str, Any]) -> dict[str, Any]:
    """Add explicit inventory/scope semantics to the compact sampling audit."""
    omitted = int(sampling.get("omitted", 0) or 0)
    return {
        **sampling,
        "scope": "full-inventory" if omitted == 0 else "stratified-sample",
        "inventory_complete": omitted == 0,
        "selected_scope": "all-requested-points" if omitted == 0 else "selected-stage-points",
    }


def _annotate_stage_service_scope(stage_service: dict[str, Any],
                                  sampling: dict[str, Any]) -> dict[str, Any]:
    """Keep selected-point validation while preventing a sampled run from being full."""
    result = dict(stage_service)
    coverage = dict(result.get("coverage", {}))
    selected_status = result.get("calibration_status", result.get("status"))
    selected_validated = bool(result.get("validated", False))
    inventory_complete = bool(sampling.get("inventory_complete", True))
    coverage.update({
        "stage_point_sampling": sampling,
        "inventory_complete": inventory_complete,
        "inventory_scope": sampling.get("scope"),
        "selected_scope_validated": selected_validated,
    })
    result["coverage"] = coverage
    result["stage_point_sampling"] = sampling
    result["inventory_complete"] = inventory_complete
    result["selected_scope_status"] = selected_status
    result["selected_scope_validated"] = selected_validated
    if not inventory_complete and selected_status in {"complete", "complete-bounded"}:
        result["calibration_status"] = "complete-sampled"
        result["validated"] = False
    return result


def _search_workload_key(workload: dict[str, Any]) -> str:
    return json.dumps(workload, sort_keys=True)


def _resume_selection_prefix(cell: dict[str, Any], retained: dict[str, dict[str, Any]]):
    """Recover only a fully persisted, contiguous selection prefix.

    ``selection_index`` is written before a candidate is measured, while
    ``screened`` advances only after the measurement is persisted.  Requiring
    both the contiguous audit entries and their journal rows prevents a
    checkpoint taken mid-selection from being treated as completed work.
    """
    if not isinstance(cell, dict) or cell.get("complete"):
        return None
    screened_count = cell.get("screened")
    if type(screened_count) is not int or screened_count <= 0:
        return None
    candidates = cell.get("candidates")
    if not isinstance(candidates, list):
        return None
    indexed = {}
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        selection_index = candidate.get("selection_index")
        if type(selection_index) is not int or selection_index < 0:
            continue
        if selection_index in indexed:
            return None
        indexed[selection_index] = candidate
    prefix = []
    for selection_index in range(screened_count):
        candidate = indexed.get(selection_index)
        if not isinstance(candidate, dict):
            return None
        candidate_id_value = candidate.get("id")
        configuration = candidate.get("configuration")
        if not isinstance(candidate_id_value, str) or not isinstance(configuration, dict):
            return None
        try:
            if candidate_id(configuration) != candidate_id_value:
                return None
        except (TypeError, ValueError):
            return None
        row = retained.get(candidate_id_value)
        if not isinstance(row, dict) or row.get("name") != candidate_id_value:
            return None
        row_configuration = row.get("configuration")
        if not isinstance(row_configuration, dict):
            return None
        try:
            if candidate_id(row_configuration) != candidate_id_value:
                return None
        except (TypeError, ValueError):
            return None
        prefix.append((copy.deepcopy(configuration), copy.deepcopy(row)))
    return prefix


def low_fft_numeric_policy(args, point, command):
    """Keep the low-FFT reporting switch off unrelated benchmark requests."""
    requested = getattr(args, 'low_fft_error_report_only', False)
    applicable = (point.get('operator', 'fft') == 'fft' and
                  point.get('precision', 'fp32') in {'fp16', 'bf16', 'fp16-fp32'})
    flag = '--low-fft-error-report-only'
    if flag in command and not applicable:
        raise ValueError('low-FFT error reporting applies only to FP16/BF16 FFT requests')
    if requested and applicable:
        return [*command, *([] if flag in command else [flag])], 'same-precision-cpu-reference-error-report-only'
    return command, 'default-verification; low-fft-error-report-only-not-applicable' if requested else 'default-verification'


def workload_search_inputs(args, snapshot, workload, candidate_binary, cpu_preparation=None):
    """Choose the existing search inputs without enumerating a seed-only frontier."""
    seeds = candidates_for_workload(snapshot, workload)
    if getattr(args, 'seed_only', False):
        budget = getattr(args, 'seed_budget', 0)
        if budget <= 0 or not seeds:
            raise ValueError(f"seed-only has no usable mapping seeds for {workload}; seed-budget must be positive")
        return [seed['point'] for seed in seeds[:budget]], seeds
    points = (prepared_inventory(cpu_preparation, workload) if cpu_preparation is not None
              else runtime_candidates(candidate_binary, workload, run))
    return points, seeds


def run_compiled_search(args, output):
    binary = args.build_dir / "cubutterfly_bench"
    library_guard = _SharedLibraryGuard(args.build_dir)
    staged = getattr(args, "cost_model", "legacy") == "staged"
    evolutionary = getattr(args, "search_strategy", "model") == "evolutionary"
    profile = json.loads(output.with_name("calibration_device.json").read_text()) if staged else None
    probe = args.build_dir / "cubutterfly_plan_probe"
    if staged and not probe.is_file():
        raise ValueError("staged costs require cubutterfly_plan_probe; build/install that target first")
    if staged:
        profile["hardware"] = json.loads(run([str(probe), "--hardware"]).stdout)
        write_checkpoint(output.with_name("calibration_device.json"), profile)
    stage_service_for_ranking = _stage_service_usable_for_ranking(profile) if staged else False
    stage_service_partial = bool(stage_service_for_ranking and not (
        profile['stage_service'].get('validated') or profile['stage_service'].get('selected_scope_validated')))
    units = list(csv.DictReader(io.StringIO(run([str(binary), "--list-processing-units"]).stdout)))
    for unit in units:
        for field in ("logN", "threads", "ept"):
            unit[field] = int(unit[field])
    register_mappings = list(csv.DictReader(io.StringIO(run([str(binary), "--list-register-tile-mappings"]).stdout)))
    specification = json.loads(args.search_workloads.read_text())
    if specification.get("schema") != "cubutterfly-install-search-v1":
        raise ValueError("unsupported install search workload schema")
    seed_budget = getattr(args, "seed_budget", 0)
    seed_only = getattr(args, 'seed_only', False)
    snapshot = getattr(args, "_prepared_seed_snapshot", None)
    if snapshot is None:
        snapshot = seed_snapshot(args, output.with_name("mapping_seed_snapshot.json")) if seed_budget else {"seeds": []}
    stage_sampling_path = output.with_name("stage_point_sampling.json")
    try:
        stage_sampling = json.loads(stage_sampling_path.read_text()) if stage_sampling_path.exists() else None
    except (OSError, json.JSONDecodeError):
        stage_sampling = None
    measurement = measurement_identity(args, library_guard)
    cpu_preparation = load_cpu_preparation(args, measurement)
    disk_scores = None
    if cpu_preparation is not None and not getattr(args, 'cpu_preparation_inventory_only', False):
        from calibration_cpu_cache import DiskScoreCache
        disk_scores = DiskScoreCache(args.cpu_preparation_dir / 'score_cache', cpu_preparation[1]['context'])
    audit = {"schema": "cubutterfly-search-coverage-v1", "scope": specification["scope"],
             "measurement_identity": measurement,
             "search_protocol": {key:getattr(args,key,None) for key in
                ("search_budget","search_finalists","search_seconds","compile_seconds",
                 "operator_trials","operator_warmup","operator_repeat")},
             "partition_projection_budget": int(os.environ.get("CUBUTTERFLY_PARTITION_BUDGET", "128")),
             "partition_projection": "library cursor spans all resource-feasible group counts; 0 removes enumeration budget",
             "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
             "compiled_local_units": units, "search_budget_per_workload": args.search_budget,
             "compiled_register_tile_mappings": register_mappings,
             "optimality": "best confirmed measured candidate; no global-optimum claim",
             "workloads": []}
    audit["search_protocol"]["seed_budget"] = seed_budget
    if getattr(args, 'low_fft_error_report_only', False):
        audit['search_protocol']['low_fft_error_report_only'] = True
        audit['numeric_policy'] = 'same-precision CPU error report only for low FFT; default verification for other requests'
    if seed_only:
        audit.update(search_scope='seed-only', inventory_complete=False,
                     optimality='best confirmed measured mapping seed; full library search not performed')
        audit['partition_projection'] = 'seed-only; runtime inventory not enumerated'
        audit['search_protocol']['seed_only'] = True
    audit["search_protocol"]["cost_feature_version"] = FEATURE_VERSION
    audit["search_protocol"]["cost_projection_version"] = PROJECTION_VERSION
    audit["search_protocol"]["model_update_policy_version"] = MODEL_UPDATE_POLICY_VERSION
    audit["search_protocol"]["selection_policy_version"] = SEARCH_SELECTION_VERSION
    audit["search_protocol"]["cost_model"] = "staged" if staged else "legacy"
    audit["search_protocol"]["search_strategy"] = "evolutionary" if evolutionary else "model"
    audit["search_protocol"]["stage_calibration"] = getattr(args, "stage_calibration", "skip")
    audit["search_protocol"]["search_stage_projection"] = getattr(args, "search_stage_projection", "full")
    audit["search_protocol"]["stage_point_budget"] = getattr(args, "stage_point_budget", 0)
    audit["search_protocol"]["stage_point_sampling"] = stage_sampling or {
        "scope": "not-recorded", "inventory_complete": False,
    }
    projection_requested = getattr(args, "search_stage_projection", "full")
    projection_available = bool(staged and profile and profile.get("stage_service"))
    audit["search_protocol"]["stage_projection_scope"] = (
        "full-stage-service" if projection_requested == "full" and projection_available
        else "plan-only-no-stage-service" if projection_requested == "plan"
        else "no-stage-service")
    audit["search_protocol"]["stage_projection_complete"] = (
        projection_requested == "full" and projection_available)
    audit["search_protocol"]["stage_service_ranking"] = (
        "validated-curve-subset-empirical" if stage_service_partial else
        "validated-scope-empirical" if stage_service_for_ranking else
        "hardware-rate-prior-fallback" if staged else "not-applicable")
    if staged:
        audit['search_protocol']['ranking_scope_policy'] = 'certified-curves-v1'
        from stage_schedule_native import backend as schedule_backend
        audit['cpu_scoring'] = dict(implementation='prepared-features-exact-events-v1',
                                    event_simulator=schedule_backend())
    if staged:
        audit["search_protocol"]["stage_model_version"] = stage_cost_model.VERSION
        audit["search_protocol"]["plan_probe_sha256"] = hashlib.sha256(probe.read_bytes()).hexdigest()
    audit["search_protocol"]["seed_snapshot_sha256"] = hashlib.sha256(
        json.dumps(snapshot, sort_keys=True).encode()).hexdigest()
    expected_library = measurement.get("library_sha256")

    def benchmark_run(command):
        library_guard.assert_matches(expected_library, "before")
        completed = run(command, check=False)
        library_guard.assert_matches(expected_library, "after")
        return completed

    result = []
    search_start = time.monotonic()
    def compile_spent():
        budget_file=os.environ.get("CUBUTTERFLY_COMPILE_BUDGET_FILE")
        return float(pathlib.Path(budget_file).read_text() or "0") if budget_file and pathlib.Path(budget_file).exists() else 0
    compile_start=compile_spent()
    def search_elapsed():
        return time.monotonic()-search_start-(compile_spent()-compile_start)
    cached = {}
    retained = {}
    completed_cells = {}
    resumable_cells = {}
    build_priors = {}
    prior_inventory_ids = set()
    journal = output.with_name("search_measurements.json")
    if args.resume_search and output.exists() and journal.exists():
        previous = json.loads(output.read_text())
        changed_build = previous.get("measurement_identity") != audit["measurement_identity"]
        if changed_build:
            if not getattr(args, "resume_build_priors", False):
                raise ValueError("cannot resume search measurements from different binaries, compile policy or verification coverage")
            _validate_build_prior_identity(previous.get("measurement_identity", {}), measurement,
                                           previous.get("search_protocol", {}), audit["search_protocol"])
            audit["development_build_priors"] = _archive_build_priors(
                [output, journal, output.with_name("plan_descriptors.json")], previous["measurement_identity"])
            prior_inventory_ids = {candidate["id"] for cell in previous.get("workloads", [])
                                   for candidate in cell.get("candidates", []) if "id" in candidate}
            audit["development_build_priors"]["previous_inventory_source"] = next(
                source["path"] for source in audit["development_build_priors"]["sources"]
                if source["path"].endswith("-" + output.name))
        elif previous.get("development_build_priors"):
            audit["development_build_priors"] = previous["development_build_priors"]
            source = audit["development_build_priors"].get("previous_inventory_source")
            if source:
                old_inventory = json.loads(pathlib.Path(source).read_text())
                prior_inventory_ids = {candidate["id"] for cell in old_inventory.get("workloads", [])
                                       for candidate in cell.get("candidates", []) if "id" in candidate}
        for row in json.loads(journal.read_text()):
            if changed_build:
                row = _build_prior_row(row, previous["measurement_identity"])
            retained[row["name"]] = row
            if row.get("correct") and row.get("samples"):
                if row.get("measurement_role") == "development-build-prior":
                    build_priors[row["name"]] = row
                else:
                    cached[row["name"]] = row
        previous_protocol = previous.get("search_protocol", {})
        if previous.get('compilation_budget_revision'):
            audit['compilation_budget_revision'] = previous['compilation_budget_revision']
        same_protocol = previous_protocol == audit["search_protocol"]
        if (getattr(args, 'retry_compilation_gaps', False) and not same_protocol
                and {k:v for k,v in previous_protocol.items() if k != 'compile_seconds'} ==
                    {k:v for k,v in audit['search_protocol'].items() if k != 'compile_seconds'}):
            same_protocol = True
            audit['compilation_budget_revision'] = dict(
                previous=previous_protocol.get('compile_seconds'), current=args.compile_seconds,
                reason='explicit compilation-gap recovery; measurement and selection protocol unchanged')
        if not changed_build and same_protocol:
            previous_cells = previous.get("workloads", [])
            completed_cells={_search_workload_key(cell["workload"]):cell for cell in previous_cells
                             if isinstance(cell, dict) and cell.get("workload") is not None and cell.get("complete")}
            resumable_cells={_search_workload_key(cell["workload"]):cell for cell in previous_cells
                             if isinstance(cell, dict) and cell.get("workload") is not None and not cell.get("complete")}
    def save_measurements(rows):
        # Resuming an earlier cell must not erase completed later cells if
        # another interruption occurs before the traversal reaches them.
        retained.update((row["name"],row) for row in rows)
        write_checkpoint(journal,list(retained.values()))

    descriptors = {}
    descriptor_path = output.with_name("plan_descriptors.json")
    descriptor_identity = {"measurement": audit["measurement_identity"],
        "service_projection_version": "actual-batch-tile-v1",
        "stage_projection_scope": audit["search_protocol"]["stage_projection_scope"],
        "stage_point_budget": audit["search_protocol"]["stage_point_budget"],
        "stage_point_sampling": audit["search_protocol"]["stage_point_sampling"],
        "probe": audit["search_protocol"].get("plan_probe_sha256"),
        "stage_probe": hashlib.sha256((args.build_dir / "cubutterfly_stage_microbench").read_bytes()).hexdigest()
            if staged and profile.get("stage_service") else None,
        "stage_service_sha256": hashlib.sha256(
            json.dumps(profile.get("stage_service"), sort_keys=True, separators=(",", ":"), default=str).encode()
        ).hexdigest() if staged and profile.get("stage_service") else None,
        "profile": {key:value for key,value in (profile or {}).items() if key != "stage_service"}}
    if staged and args.resume_search and descriptor_path.exists():
        previous = json.loads(descriptor_path.read_text())
        if previous.get("identity") == descriptor_identity:
            descriptors = previous.get("descriptors", {})

    def projected(point):
        resolved = descriptors.get(candidate_id(point), {})
        if resolved.get("status") == "resolved":
            projection = resolved.get("stage_service_projection")
            if not isinstance(projection, dict):
                projection = ({"status": "omitted", "reason": "search-stage-projection=plan"}
                              if staged and getattr(args, "search_stage_projection", "full") == "plan"
                              else {})
            return {**point, **resolved.get("sample", {}), **json.loads(resolved["mapping_json"]), "N": 1 << int(point["logN"]),
                "stage_service_projection": projection,
                "stage_projection_status": projection.get("status"),
                "stage_projection_reason": projection.get("reason"),
                "execution_groups_json": resolved["execution_groups_json"], "descriptor_source": "resolved-plan"}
        return {**predicted_sample(point), "descriptor_source": "projected"}

    def resolve(point):
        key = candidate_id(point)
        if key not in descriptors:
            actual_stage = (staged and profile.get("stage_service") and
                            getattr(args, "search_stage_projection", "full") == "full")
            command = ([str(args.build_dir / "cubutterfly_stage_microbench"), "--describe-only"]
                       if actual_stage else [str(probe)])
            completed = benchmark_run(command + ["--point-json", json.dumps(point)])
            try:
                value = json.loads(completed.stdout)
                if actual_stage and value.get("status") == "resolved":
                    from stage_composition import resolve_batch_services
                    def describe_tile(tile_point):
                        return resolve(tile_point)
                    value["stage_service_projection"] = resolve_batch_services(point, value, describe_tile)
                    sample = value["sample"]
                    value = {**value, "mapping_json":sample["mapping_json"],
                        "execution_groups_json":value["groups"],
                        "runtime_fingerprint":sample["runtime_fingerprint"], "descriptor_source":"actual-stage-probe"}
                    projection = value.get("stage_service_projection", {})
                    value["stage_projection_status"] = projection.get("status")
                    value["stage_projection_reason"] = projection.get("reason")
                elif (staged and value.get("status") == "resolved" and
                      getattr(args, "search_stage_projection", "full") == "plan"):
                    value["stage_service_projection"] = {
                        "status": "omitted", "reason": "search-stage-projection=plan",
                    }
                    value["stage_projection_status"] = "omitted"
                    value["stage_projection_reason"] = "search-stage-projection=plan"
            except (TypeError, ValueError, KeyError, IndexError):
                value = dict(status="unavailable", reason=completed.stderr or "plan probe returned no JSON")
            # Compilation exhaustion may be retried with a later round's budget.
            if "compilation budget exhausted" not in value.get("reason", ""):
                descriptors[key] = value
                write_checkpoint(descriptor_path, dict(identity=descriptor_identity, descriptors=descriptors))
            return value
        return descriptors[key]

    def measure(point, trials, warmup, repeat, *, seed=False, confirm=False):
        candidate_binary = args.build_dir / ("cuntt_bench" if point.get("operator") == "ntt" else "cubutterfly_bench")
        try:
            command = command_for(candidate_binary, point)
            command, numeric_policy = low_fft_numeric_policy(args, point, command)
        except (TypeError, ValueError, KeyError) as error:
            # A runtime inventory or imported seed is executable input, not a
            # guarantee that every public axis is still legal for this build.
            # Keep one malformed point in the audit and continue the cell.
            return {
                "name": candidate_id(point), "command": [], "configuration": point,
                "samples": [], "errors": [str(error)], "correct": False,
                "status": "invalid-candidate", "trials": 0,
            }
        old = cached.get(candidate_id(point))
        if confirm and old and old.get("measurement_role") == "development-build-prior":
            old = None
        reusable=bool(old and old.get("correct") and old.get("samples") and all(
            int(s["warmup"])>=warmup and int(s["repeat"])>=repeat for s in old["samples"]))
        descriptor = descriptors.get(candidate_id(point), {})
        if reusable and descriptor.get("descriptor_source") == "actual-stage-probe":
            # A compatible earlier run may predate independent stage
            # calibration. Reuse its timings while upgrading only the
            # descriptor, after checking the actual code/mapping identity.
            old = copy.deepcopy(old)
            try:
                for sample in old["samples"]:
                    if sample.get("runtime_fingerprint") != descriptor.get("runtime_fingerprint"):
                        raise ValueError("cached timing and stage descriptor build mismatch")
                    validate_mapping(json.loads(descriptor["mapping_json"]), json.loads(sample["mapping_json"]))
                    sample.setdefault("benchmark_execution_groups_json",sample.get("execution_groups_json","[]"))
                    sample["execution_groups_json"] = json.dumps(descriptor["execution_groups_json"])
                    sample["stage_service_projection"] = descriptor.get("stage_service_projection", {})
                    sample["descriptor_source"] = "actual-stage-probe"
            except (ValueError, TypeError, KeyError):
                reusable = False
        if (reusable and staged and descriptor.get("status") == "resolved" and
                isinstance(descriptor.get("stage_service_projection"), dict) and
                descriptor["stage_service_projection"].get("status") == "omitted"):
            old = copy.deepcopy(old)
            for sample in old["samples"]:
                sample["stage_service_projection"] = descriptor["stage_service_projection"]
                sample["stage_projection_status"] = "omitted"
                sample["stage_projection_reason"] = descriptor["stage_service_projection"].get("reason")
        if reusable and seed:
            try:
                for sample in old["samples"]:
                    validate_mapping(json.loads(point["mapping_json"]), json.loads(sample.get("mapping_json", "{}")))
            except (ValueError, TypeError):
                reusable = False
        if reusable and len(old["samples"])>=trials:
            library_guard.assert_matches(expected_library, "reuse")
            return dict(copy.deepcopy(old),status="measured")
        record = {"name": candidate_id(point), "command": command, "configuration": point,
                  "samples": copy.deepcopy(old["samples"]) if reusable else [],
                  "errors": [], "correct": False, "status": "unavailable"}
        if getattr(args, 'low_fft_error_report_only', False):
            record['numeric_policy'] = numeric_policy
        for _ in range(len(record["samples"]),trials):
            require_exclusive_gpu()
            verification = ["--verify-batches",str(args.verify_batches)] if getattr(args,"verify_batches",0) else []
            completed = benchmark_run(command + ["--warmup", str(warmup), "--repeat", str(repeat), "--verify", "--csv"] + verification)
            require_exclusive_gpu()
            if completed.returncode:
                record["correct"]=False
                record["errors"].append(completed.stderr.strip() or completed.stdout.strip())
                if "compilation budget exhausted" in record["errors"][-1]:
                    record["status"]="needs-compilation"
                return record
            samples = list(csv.DictReader(io.StringIO(completed.stdout)))
            if len(samples) != 1 or samples[0].get("correct") != "1":
                record["correct"]=False
                record["status"] = "incorrect"
                return record
            sample = samples[0]
            descriptor = descriptors.get(candidate_id(point), {})
            if staged and descriptor.get("status") == "resolved" and sample.get("runtime_fingerprint") != descriptor.get("runtime_fingerprint"):
                record["errors"].append("plan probe and timing binary resolved different runtime fingerprints")
                return record
            if staged and descriptor.get("descriptor_source") == "actual-stage-probe":
                try:
                    validate_mapping(json.loads(descriptor["mapping_json"]), json.loads(sample["mapping_json"]))
                except (ValueError, TypeError, KeyError) as error:
                    record.update(correct=False, status="mapping-mismatch")
                    record["errors"].append(str(error))
                    return record
                sample["benchmark_execution_groups_json"] = sample.get("execution_groups_json", "[]")
                sample["execution_groups_json"] = json.dumps(descriptor["execution_groups_json"])
                sample["stage_service_projection"] = descriptor.get("stage_service_projection", {})
                sample["descriptor_source"] = "actual-stage-probe"
            elif (staged and descriptor.get("status") == "resolved" and
                  isinstance(descriptor.get("stage_service_projection"), dict) and
                  descriptor["stage_service_projection"].get("status") == "omitted"):
                sample["stage_service_projection"] = descriptor["stage_service_projection"]
                sample["stage_projection_status"] = "omitted"
                sample["stage_projection_reason"] = descriptor["stage_service_projection"].get("reason")
            if point.get("operator") == "ntt":
                sample.update(operator="ntt", precision="word"+sample["word_bits"],
                              placement=sample["output_order"], direction="inverse" if sample["inverse"] == "1" else "forward")
            sample["measurement_exclusive_gpu"] = "1"
            sample.setdefault("modulus", "0")
            if seed:
                try:
                    validate_mapping(json.loads(point["mapping_json"]), json.loads(sample.get("mapping_json", "{}")))
                    # Match target semantics, including placement and strides;
                    # source semantics are never copied into the command.
                    expected_semantics, resolved_semantics = canonical_semantics(point), canonical_semantics(sample)
                    for key in ("placement", "direction", "normalization", "element_stride", "batch_stride",
                                "accumulation", "modulus", "input_order", "output_order"):
                        if key in point and resolved_semantics.get(key) != expected_semantics.get(key):
                            raise ValueError(f"resolved seed semantics mismatch for {key}")
                except (ValueError, TypeError) as error:
                    record.update(correct=False, status="mapping-mismatch")
                    record["errors"].append(str(error))
                    return record
            if record["samples"] and any(sample.get(key) != record["samples"][0].get(key)
                                          for key in ("mapping_json", "runtime_fingerprint")):
                raise ValueError("mapping or code changed during search confirmation")
            # A timed fallback is not evidence for the requested design point.
            try:
                for key, value in point.items():
                    if key == "stage_matrix" or ("mapping_json" in point and key not in ("backend", "fft_core", "logN", "batch", "precision", "operator")):
                        continue
                    if key not in sample or str(sample[key]) != str(value):
                        raise ValueError(f"resolved mapping mismatch for {key}: {sample.get(key)} != {value}")
            except ValueError as error:
                record["status"] = "mapping-mismatch"
                record["errors"].append(str(error))
                return record
            record["samples"].append(sample)
            record.update(status="screened",correct=True,trials=len(record["samples"]),
                median_kernel_ms=statistics.median(float(s["kernel_ms"]) for s in record["samples"]))
            cached[record["name"]]=copy.deepcopy(record)
            save_measurements([record])
        record.update(status="measured", correct=True, trials=trials,
                      median_kernel_ms=statistics.median(float(s["kernel_ms"]) for s in record["samples"]))
        return record

    def retry_compilation_rows(rows, by_id):
        if not getattr(args, 'retry_compilation_gaps', False):
            return rows
        updated = list(rows)
        for index, previous in enumerate(rows):
            if previous.get('status') != 'needs-compilation':
                continue
            point = previous['configuration']
            print('retry compilation gap ' + previous['name'], flush=True)
            if staged:
                resolve(point)
            row = measure(point, 1, max(3, min(args.operator_warmup, 10)),
                          max(5, min(args.operator_repeat, 10)), seed=bool(previous.get('seed_sources')))
            row['prior_attempts'] = [*previous.get('prior_attempts', []),
                dict(status=previous['status'], errors=previous.get('errors', []),
                     trials=previous.get('trials', 0))]
            if previous.get('seed_sources'):
                row['seed_sources'] = previous['seed_sources']
            if row['correct']:
                row['status'] = 'screened'
            updated[index] = row
            by_id[row['name']].update(status=row['status'], errors=row.get('errors', []),
                                      median_kernel_ms=row.get('median_kernel_ms'))
            save_measurements([row])
        return updated

    def confirm_finalists(screened, cell, by_id, chosen_ids):
        valid = sorted((r for r in screened if r['correct']), key=lambda r:r['median_kernel_ms'])
        finalist_keys, finalists = set(), set()
        for row in valid:
            samples = row.get('samples')
            key = (design_point_key({'name':row.get('name', ''), 'sample':samples[0]})
                   if isinstance(samples, list) and samples and isinstance(samples[0], dict)
                   else ('incomplete-screen', row.get('name', '')))
            if key in finalist_keys:
                continue
            finalist_keys.add(key)
            finalists.add(row['name'])
            if len(finalists) >= args.search_finalists:
                break
        confirmed = []
        for index, row in enumerate(screened):
            if row['name'] in finalists:
                previous_attempts = row.get('prior_attempts')
                row = measure(row['configuration'], args.operator_trials,
                              args.operator_warmup, args.operator_repeat,
                              seed=row['name'] in chosen_ids, confirm=True)
                if previous_attempts:
                    row['prior_attempts'] = previous_attempts
            elif row['correct']:
                row['status'] = 'screened'
            by_id[row['name']].update(status=row['status'], errors=row.get('errors', []),
                                     median_kernel_ms=row.get('median_kernel_ms'))
            if row['name'] in chosen_ids:
                row['seed_sources'] = by_id[row['name']]['seed_sources']
            confirmed.append(row)
            save_measurements(confirmed + screened[index+1:])
        cell['seed_coverage']['confirmed'] = sum(
            r['name'] in chosen_ids and r['correct'] and r['status'] == 'measured' for r in confirmed)
        cell['measurement_coverage'] = dict(
            selected=len(confirmed), correct=sum(bool(r['correct']) for r in confirmed),
            confirmed=sum(r['status'] == 'measured' for r in confirmed),
            needs_compilation=sum(r['status'] == 'needs-compilation' for r in confirmed))
        return confirmed

    for workload_index, workload in enumerate(specification["workloads"]):
        remaining_cells=len(specification["workloads"])-workload_index
        cell_start=search_elapsed()
        cell_budget=max(0,args.search_seconds-cell_start)/remaining_cells if args.search_seconds else 0
        workload = {"operator": "fft", **workload}
        completed=completed_cells.get(_search_workload_key(workload))
        if completed:
            library_guard.assert_matches(expected_library, "reuse")
            audit["workloads"].append(completed)
            previous_rows = [retained[p['id']] for p in completed['candidates'] if p['id'] in retained]
            if getattr(args, 'retry_compilation_gaps', False) and any(
                    r.get('status') == 'needs-compilation' for r in previous_rows):
                # Revisit only selected, unmeasured points. Completed cells need
                # neither full inventory enumeration nor a different search path.
                by_id = {p['id']:p for p in completed['candidates']}
                previous_rows.sort(key=lambda r:by_id[r['name']].get('selection_index', 0))
                previous_rows = retry_compilation_rows(previous_rows, by_id)
                chosen_ids = {r['name'] for r in previous_rows if r.get('seed_sources')}
                previous_rows = confirm_finalists(previous_rows, completed, by_id, chosen_ids)
                completed['compilation_gaps_retried'] = True
                write_checkpoint(output, audit)
            result.extend(previous_rows)
            print(f"resume completed {workload['operator']} {workload['precision']} logN={workload['logN']} batch={workload['batch']}",flush=True)
            continue
        candidate_binary = args.build_dir / ("cuntt_bench" if workload.get("operator") == "ntt" else "cubutterfly_bench")
        points, seeds = workload_search_inputs(args, snapshot, workload, candidate_binary, cpu_preparation)
        chosen_seeds = seeds[:seed_budget]
        prior_screened = []
        for point in points:
            prior = build_priors.get(candidate_id(point))
            if prior is None:
                continue
            try:
                for sample in prior["samples"]:
                    validate_mapping(json.loads(point["mapping_json"]), json.loads(sample.get("mapping_json", "{}")))
            except (ValueError, TypeError, KeyError):
                continue
            prior = copy.deepcopy(prior)
            prior["configuration"] = point
            prior_screened.append(prior)
        prior_ids = {row["name"] for row in prior_screened}
        chosen_seeds = [seed for seed in chosen_seeds if candidate_id(seed["point"]) not in prior_ids]
        chosen_ids = {candidate_id(s["point"]) for s in chosen_seeds}
        resume_cell = resumable_cells.get(_search_workload_key(workload))
        resume_prefix = _resume_selection_prefix(resume_cell, retained) if resume_cell else None
        resume_fast = bool(resume_prefix)
        remaining = stratified_candidates([p for p in points if candidate_id(p) not in chosen_ids | prior_ids], 0)
        if prior_inventory_ids:
            remaining.sort(key=lambda point: candidate_id(point) in prior_inventory_ids)
        exploration_budget = (0 if seed_only else
            (args.search_budget or len(remaining)) if evolutionary else min(args.search_budget or len(remaining), len(remaining)))
        budget = len(chosen_seeds) + exploration_budget + len(prior_screened)
        if resume_fast:
            previous_budget = resume_cell.get("requested_screened_count")
            if (type(previous_budget) is int and previous_budget != budget) or len(resume_prefix) > budget:
                resume_fast = False
                resume_prefix = None
            else:
                # Keep the original stratified order and full budget. Only
                # already screened prefix IDs are removed from the frontier.
                resume_prefix_ids = {candidate_id(item[0]) for item in resume_prefix}
                remaining = [point for point in remaining if candidate_id(point) not in resume_prefix_ids]
        unique = {candidate_id(p): p for p in points}
        unique.update((candidate_id(s["point"]), s["point"]) for s in seeds)
        points = list(unique.values())
        fresh_cell = {"workload": workload, "enumerated": len(points), "screened": 0,
                "candidate_source": "linked-library+mapping-seeds" if seeds else "linked-library", "model_updates": [],
                "requested_screened_count": budget,
                "seed_coverage": {"available": len(seeds), "requested": len(chosen_seeds),
                                  "attempted": 0, "confirmed": 0, "budget_omitted": len(seeds)-len(chosen_seeds)},
                "budget_omitted": len(points),
                "partitions": sorted({tuple(json.loads(p["mapping_json"]).get("stage_partition", [])) for p in points}),
                # Keep the checkpoint compact. The full mapping inventory is
                # regenerated from the unchanged build on resume; only
                # selected candidates need their configuration serialized.
                "candidates": [{"id": candidate_id(p), "status": "budget-omitted"} for p in points],
                "candidate_audit": "compact-id-only-v1"}
        if seed_only:
            fresh_cell.update(candidate_source='mapping-seeds-only', search_scope='seed-only', inventory_complete=False)
        if resume_fast:
            cell = copy.deepcopy(resume_cell)
            cell["workload"] = workload
            cell["screened"] = len(resume_prefix)
            cell["requested_screened_count"] = budget
            cell["enumerated"] = max(int(cell.get("enumerated", 0)), len(points), len(cell.get("candidates", [])))
            cell["budget_omitted"] = len(cell.get("candidates", [])) - len(resume_prefix)
            cell.pop("complete", None)
            cell.pop("stop_reason", None)
            by_id = {candidate["id"]: candidate for candidate in cell.get("candidates", [])
                     if isinstance(candidate, dict) and isinstance(candidate.get("id"), str)}
            for point in points:
                identifier = candidate_id(point)
                if identifier in by_id:
                    continue
                candidate = {"id": identifier, "configuration": point, "status": "budget-omitted"}
                cell.setdefault("candidates", []).append(candidate)
                by_id[identifier] = candidate
            cell["enumerated"] = max(cell["enumerated"], len(cell["candidates"]))
        else:
            cell = fresh_cell
        audit["workloads"].append(cell)
        by_id = {p["id"]: p for p in cell["candidates"]}
        for seed in seeds:
            seed_id = candidate_id(seed["point"])
            by_id[seed_id]["configuration"] = seed["point"]
            by_id[candidate_id(seed["point"])]["seed_sources"] = seed["sources"]
        screened = [row for _, row in resume_prefix] if resume_fast else list(prior_screened)
        if resume_fast:
            screened = retry_compilation_rows(screened, by_id)
        if prior_screened and not resume_fast:
            cell["development_prior_screened"] = len(prior_screened)
            cell["screened"] = len(prior_screened)
            for row in prior_screened:
                by_id[row["name"]].update(status="development-build-prior", configuration=row["configuration"],
                                          origin_identity=row["origin_identity"], median_kernel_ms=row["median_kernel_ms"])
        if resume_fast:
            print(f"resume screened prefix {workload['operator']} {workload['precision']} "
                  f"logN={workload['logN']} batch={workload['batch']} {len(screened)}/{budget}", flush=True)
            for row in screened:
                by_id[row["name"]].update(status=row.get("status", "screened"),
                                           errors=row.get("errors", []),
                                           median_kernel_ms=row.get("median_kernel_ms"))
        ranking_profile = profile
        if stage_service_partial:
            ranking_profile = dict(profile, stage_service=dict(profile['stage_service'], enforce_ranking_scope=True))
        if staged and not stage_service_for_ranking:
            # Keep the empirical profile available to describe/verify plans,
            # but remove it from the ranking model until its selected scope is
            # validated.  stage_cost_model.fit then uses measured hardware
            # capability rates as an explicit fallback prior.
            ranking_profile = dict(profile)
            ranking_profile.pop("stage_service", None)
        model = stage_cost_model.fit([], ranking_profile) if staged else None
        model_training_ids = None
        prediction_cache = (_SearchPredictionCache(descriptors, projected, disk_scores,
                            persist_frontier=False) if staged else None)
        cell['score_persistence'] = 'shortlist-and-selected-only; exact full-frontier scores remain in memory'
        for i in range(len(screened), budget):
            iteration_started = time.perf_counter()
            if args.search_seconds and search_elapsed()-cell_start >= cell_budget:
                cell["stop_reason"] = "fair-share-search-time-budget"
                break
            pool = {r["name"]: r for r in ([*retained.values(), *screened] if staged else screened)
                    if r["correct"] and r.get("samples")}
            training = [{"sample": r["samples"][0], "median_kernel_ms": r["median_kernel_ms"]} for r in pool.values()]
            training_ids = tuple((r["name"], r["median_kernel_ms"]) for r in pool.values())
            if len(training) >= (1 if staged else 8) and training_ids != model_training_ids:
                model = stage_cost_model.fit(training, ranking_profile) if staged else fit(training, 1.0e-3)
                model_training_ids = training_ids
                update = {"screened": i, "training_rows": len(training),
                    "training_candidate_ids": list(pool),
                    "feature_version": FEATURE_VERSION,
                    "projection_version": PROJECTION_VERSION,
                    "use": "target-local search ranking; every fourth selection explores a stratum"}
                audit_model = dict(model) if staged else None
                if staged and profile.get("stage_service"):
                    audit_model.pop("stage_service", None)
                    audit_model["profile"] = {key:value for key,value in model.get("profile", {}).items() if key != "stage_service"}
                    audit_model["stage_service_artifact"] = str(output.with_name("stage_service_profile.json"))
                    audit_model["stage_service_sha256"] = hashlib.sha256(output.with_name("stage_service_profile.json").read_bytes()).hexdigest()
                update.update({"stage_model": audit_model} if staged else
                    {"coefficients": model[0], "feature_means": model[1], "feature_scales": model[2]})
                cell["model_updates"].append(update)
            if staged:
                prediction_cache.bind(model)
            selection_index = i - (len(prior_screened) if not resume_fast else 0)
            is_seed = selection_index < len(chosen_seeds)
            exploit = not is_seed and model and (selection_index-len(chosen_seeds))%4
            def score(point):
                if staged:
                    return prediction_cache.score(point)
                # A runtime rejection overrides a cheap symbolic estimate.
                # Keep its descriptor in the audit, but never rank it ahead of
                # a resolved candidate because its hypothetical cost is lower.
                if staged and descriptors.get(candidate_id(point), {}).get("status") == "unavailable":
                    return float("inf")
                try:
                    return stage_cost_model.predict(projected(point), model) if staged else predict(predicted_sample(point), *model)
                except (TypeError, ValueError, KeyError, ZeroDivisionError, OverflowError, IndexError):
                    # A syntactic neighbor may need a different lowering or
                    # violate a local shape constraint. The runtime probe and
                    # verifier decide feasibility; it is not a zero-cost point.
                    return float("inf")
            proposals = []
            # Exploration selects the next stratum directly. Building and
            # ranking a neighborhood here cannot affect that choice.
            if evolutionary and exploit and screened:
                from evolutionary_search import evolutionary_candidates
                parents = [r["configuration"] for r in sorted(screened, key=lambda r:r.get("median_kernel_ms", float("inf"))) if r["correct"]]
                proposals = evolutionary_candidates(parents, remaining, {r["name"] for r in screened}, score, limit=32)
            choices = proposals if proposals and exploit else remaining
            if not is_seed and not choices:
                cell["stop_reason"] = "candidate-neighborhood-exhausted"
                break
            if is_seed:
                point = chosen_seeds[selection_index]["point"]
            elif exploit:
                # nsmallest preserves stable tie order without sorting the
                # entire frontier. Every candidate still receives its score.
                shortlist = heapq.nsmallest(4, choices, key=score) if staged else choices
                if staged:
                    prediction_cache.persist(shortlist)
                    for proposed in shortlist:
                        resolve(proposed)
                    # Retain unavailable proposals as explicit failures; don't
                    # pretend that a projected neighbor is an executable kernel.
                point = min(shortlist, key=score)
            else:
                point = choices[0]
            point_identity = prediction_cache.identity if staged else candidate_id
            selected_id = point_identity(point)
            remaining = [p for p in remaining if point_identity(p) != selected_id]
            if candidate_id(point) not in by_id:
                record = dict(id=candidate_id(point), configuration=point, status="budget-omitted", source="evolutionary-neighbor")
                cell["candidates"].append(record); by_id[record["id"]] = record
                points.append(point); cell["enumerated"] = len(points)
                cell["generated_neighbors"] = cell.get("generated_neighbors", 0) + 1
            selection = by_id[candidate_id(point)]
            selection["configuration"] = point
            selection["selection_reason"] = "historical-seed" if is_seed else ("evolutionary-model" if proposals and exploit else "model" if exploit else "stratified-exploration")
            selection["selection_index"] = i
            if exploit:
                estimate = score(point)
                selection["predicted_kernel_ms"] = estimate if math.isfinite(estimate) else None
            if staged:
                description = resolve(point)
                selection["plan_probe"] = description
                try:
                    selection["cost_breakdown"] = stage_cost_model.predict(projected(point), model, True)
                except (TypeError, ValueError, KeyError, ZeroDivisionError, OverflowError, IndexError) as error:
                    selection["cost_breakdown"] = dict(status="unestimated", reason=str(error))
            print(f"search {workload['operator']} {workload['precision']} logN={workload['logN']} batch={workload['batch']} {i + 1}/{budget}", flush=True)
            if staged:
                prediction_cache.persist([point])
            row = measure(point, 1, max(3, min(args.operator_warmup, 10)), max(5, min(args.operator_repeat, 10)), seed=is_seed)
            if is_seed:
                cell["seed_coverage"]["attempted"] += 1
                row["seed_sources"] = selection["seed_sources"]
            if row["correct"]: row["status"]="screened"
            by_id[row["name"]]["status"] = row["status"]
            screened.append(row)
            cell["screened"] = len(screened)
            cell["budget_omitted"] = len(points)-len(screened)
            if staged:
                cell['cpu_scoring'] = dict(prediction_cache.stats)
                if disk_scores is not None:
                    cell['disk_scoring'] = dict(disk_scores.stats)
                cell['last_iteration_seconds'] = time.perf_counter() - iteration_started
            write_checkpoint(output,audit)
            # Persist completed screens immediately. An unrelated GPU job may
            # arrive before this cell reaches finalist confirmation.
            save_measurements(result+[dict(r,status="screened") if r["correct"] else r for r in screened])
        result.extend(confirm_finalists(screened, cell, by_id, chosen_ids))
        # Preserve completed cells if calibration is interrupted later.
        save_measurements(result)
        cell["complete"]=True
        write_checkpoint(output,audit)
    write_checkpoint(output,audit)
    if disk_scores is not None:
        disk_scores.close()
    return result


def incumbent_commands(args):
    """Remeasure the installed selector so a bounded search keeps its incumbent.

    No winning parameters are encoded here. The resolved mapping returned by
    the binary is the evidence; unavailable workloads simply have no incumbent.
    """
    document = json.loads(args.search_workloads.read_text())
    for workload in document["workloads"]:
        point = {"operator": "fft", **workload}
        binary = args.build_dir / ("cuntt_bench" if point.get("operator") == "ntt" else "cubutterfly_bench")
        yield "incumbent-" + candidate_id(point), command_for(binary, point) + ["--auto-select"]


def run_operator_calibration(args: argparse.Namespace, output: pathlib.Path, cufftdx: bool) -> dict[str, Any]:
    raw: list[dict[str, Any]] = []
    library_guard = _SharedLibraryGuard(args.build_dir)
    identity=measurement_identity(args, library_guard)
    expected_library = identity.get("library_sha256")
    identity_path=output.with_name("operator_measurement_identity.json")
    cached={}
    if args.resume_search and output.exists() and identity_path.exists():
        previous_identity = json.loads(identity_path.read_text())
        if previous_identity != identity:
            if not getattr(args, "resume_build_priors", False):
                raise ValueError("cannot resume incumbent measurements from different binaries, compile policy or verification coverage")
            _validate_build_prior_identity(previous_identity, identity)
            coverage = output.with_name("search_coverage.json")
            if coverage.exists():
                old_coverage = json.loads(coverage.read_text())
                _validate_build_prior_identity(previous_identity, identity,
                    old_coverage.get("search_protocol", {}), vars(args))
            _archive_build_priors([output, identity_path], previous_identity)
        else:
            cached={row["name"]:row for row in json.loads(output.read_text())}
    write_checkpoint(identity_path,identity)
    retained=dict(cached)
    def save_incumbents():
        retained.update((row["name"],row) for row in raw)
        write_checkpoint(output,list(retained.values()))
    commands = [] if getattr(args, 'seed_only', False) else list(incumbent_commands(args))
    if not args.search_only and not getattr(args, 'seed_only', False):
        commands = candidate_commands(args.build_dir, cufftdx) + ntt_candidate_commands(args.build_dir) + commands
    for name, command in commands:
        if getattr(args, 'low_fft_error_report_only', False):
            def option(key, default):
                return command[command.index(key) + 1] if key in command else default
            command, numeric_policy = low_fft_numeric_policy(args, dict(
                operator=option('--operator', 'ntt' if pathlib.Path(command[0]).name == 'cuntt_bench' else 'fft'),
                precision=option('--precision', 'fp32')), command)
        print(f"incumbent {name}",flush=True)
        previous=cached.get(name)
        reusable=bool(previous and previous.get("correct") and previous.get("samples") and all(
            int(s["warmup"])>=args.operator_warmup and int(s["repeat"])>=args.operator_repeat for s in previous["samples"]))
        if reusable and len(previous["samples"])>=args.operator_trials:
            library_guard.assert_matches(expected_library, "reuse")
            raw.append(dict(previous,status="measured"))
            save_incumbents()
            continue
        samples: list[dict[str, str]] = copy.deepcopy(previous["samples"]) if reusable else []
        errors: list[str] = []
        record={"name":name,"command":command,"samples":samples,"errors":errors,
                "correct":False,"status":"unavailable","trials":0}
        if getattr(args, 'low_fft_error_report_only', False):
            record['numeric_policy'] = numeric_policy
        raw.append(record)
        for trial in range(len(samples),args.operator_trials):
            require_exclusive_gpu()
            verification = ["--verify-batches",str(args.verify_batches)] if getattr(args,"verify_batches",0) else []
            library_guard.assert_matches(expected_library, "before")
            completed = run(command + ["--warmup", str(args.operator_warmup), "--repeat", str(args.operator_repeat), "--verify", "--csv"] + verification, check=False)
            library_guard.assert_matches(expected_library, "after")
            require_exclusive_gpu()
            if completed.returncode:
                errors.append(completed.stderr.strip() or completed.stdout.strip())
                break
            try:
                rows = list(csv.DictReader(io.StringIO(completed.stdout)))
                if len(rows) != 1:
                    raise ValueError("benchmark did not return exactly one CSV row")
                sample = rows[0]
                sample["measurement_exclusive_gpu"]="1"
                if "word_bits" in sample:
                    sample["operator"] = "ntt"
                    sample["precision"] = f"word{sample['word_bits']}"
                    sample["placement"] = sample["output_order"]
                    sample["direction"] = "inverse" if sample["inverse"] == "1" else "forward"
                else:
                    sample.setdefault("modulus", "0")
                if samples and any(sample.get(key)!=samples[0].get(key) for key in ("mapping_json","runtime_fingerprint")):
                    raise ValueError("incumbent mapping or code changed during confirmation")
                samples.append(sample)
                record.update(status="screened",correct=all(s.get("correct")=="1" for s in samples),trials=len(samples),
                    median_kernel_ms=statistics.median(float(s["kernel_ms"]) for s in samples))
                save_incumbents()
            except (ValueError, csv.Error) as error:
                errors.append(str(error))
                break
        if samples and len(samples) == args.operator_trials and not errors:
            values = [float(row["kernel_ms"]) for row in samples]
            record.update({
                "name": name,
                "status": "measured" if all(row.get("correct") == "1" for row in samples) else "incorrect",
                "trials": len(samples),
                "median_kernel_ms": statistics.median(values),
                "correct": all(row.get("correct") == "1" for row in samples),
                "command": command,
                "samples": samples,
            })
        else:
            record.update(status="unavailable",correct=False,errors=errors)
        save_incumbents()
    raw.extend(run_compiled_search(args, output.with_name("search_coverage.json")))
    output.write_text(json.dumps(raw, indent=2) + "\n")
    summary_path = output.with_suffix(".csv")
    with summary_path.open("w", newline="") as handle:
        fields = ["name", "status", "trials", "median_kernel_ms", "correct"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in raw:
            writer.writerow({field: row.get(field, "") for field in fields})
    recommendations: dict[str, dict[str, Any]] = {}
    operator_families: set[str] = set()
    for row in raw:
        if row.get("status") != "measured" or not row.get("correct"):
            continue
        operator_families.update(sample.get("operator", "") for sample in row.get("samples", []) if sample.get("operator"))
        key = json.dumps(semantic_key(row["samples"][0]))
        current = recommendations.get(key)
        if current is None or row["median_kernel_ms"] < current["median_kernel_ms"]:
            recommendations[key] = {
                "candidate": row["name"],
                "median_kernel_ms": row["median_kernel_ms"],
            }
    recommendation_path = output.with_name("operator_recommendations.json")
    recommendation_path.write_text(json.dumps(recommendations, indent=2) + "\n")
    return {
        "status": "complete",
        "cufftdx_enabled": cufftdx,
        "candidates": raw,
        "operator_families": sorted(operator_families),
        "summary": str(summary_path),
        "recommendations": str(recommendation_path),
    }


def _compute_capability_parts(value: str) -> tuple[int, int]:
    match = re.search(r"(\d+)\.(\d+)", value)
    if not match:
        raise ValueError(f"cannot parse compute capability: {value}")
    return int(match.group(1)), int(match.group(2))


def composition_seed_records(operator_records, checkpoint_path, profile):
    """Add verified stage designs without changing operator training records."""
    checkpoint_path = pathlib.Path(checkpoint_path)
    if not checkpoint_path.is_file():
        return operator_records
    from stage_service_calibration import _profile_identity
    from stage_composition import _groups

    checkpoint = json.loads(checkpoint_path.read_text())
    identity = checkpoint.get("identity", {})
    expected = _profile_identity(profile)
    if not isinstance(identity, dict) or any(identity.get(key) != value for key, value in expected.items()):
        raise ValueError("composition seed stage checkpoint hardware/runtime identity mismatch")

    def sample_key(sample):
        try:
            mapping = sample.get("mapping_json")
            mapping = json.loads(mapping) if isinstance(mapping, str) else mapping
            if not isinstance(mapping, dict) or not mapping:
                return None
            return json.dumps({"semantics": canonical_semantics(sample), "mapping": mapping},
                              sort_keys=True, separators=(",", ":"))
        except (TypeError, ValueError):
            return None

    merged = []
    seen = {}
    for record in operator_records:
        samples = record.get("samples", [])
        key = None
        if samples and isinstance(samples[0], dict):
            key = sample_key(samples[0])
        if key is not None and key in seen:
            position = seen[key]
            if len(_groups(record)) > len(_groups(merged[position])):
                merged[position] = record
            continue
        if key is not None:
            seen[key] = len(merged)
        merged.append(record)
    for index, record in enumerate(checkpoint.get("records", [])):
        if record.get("status") != "measured" or record.get("correct") is not True:
            continue
        sample, groups = record.get("sample"), record.get("groups")
        if (not isinstance(sample, dict) or not isinstance(groups, list) or not groups
                or not all(isinstance(group, dict) for group in groups)):
            continue
        for field, value in expected.items():
            actual = sample.get(field, record.get(field))
            if actual not in (None, "") and str(actual) != str(value):
                raise ValueError("composition seed record hardware/runtime identity mismatch: " + field)
        key = sample_key(sample)
        if key is None:
            continue
        seed = {
            "name": "stage-service-" + hashlib.sha256(key.encode()).hexdigest()[:20],
            "status": "measured", "correct": True,
            "samples": [{**copy.deepcopy(sample), "correct": "1",
                         "execution_groups_json": json.dumps(groups, sort_keys=True)}],
            "source": {"kind": "verified-stage-service-design", "checkpoint": str(checkpoint_path),
                       "record_index": index, "candidate_key": record.get("candidate_key"),
                       "identity": copy.deepcopy(identity)},
        }
        if key in seen:
            position = seen[key]
            if len(_groups(merged[position])) >= len(groups):
                continue
            merged[position] = seed
        else:
            seen[key] = len(merged)
            merged.append(seed)
    return merged


def _composition_calibration(args, output_dir, profile, operator_result, guarded_run):
    record = {
        "schema": "cubutterfly-composition-entry-v1",
        "scope": "independent-heldout-cross-group-overlap",
        "status": "skipped",
        "path": "",
        "summary": {},
    }
    if (args.cost_model != "staged" or getattr(args, "stage_calibration", "skip") == "skip"
            or args.skip_operator_calibration):
        record["reason"] = "composition calibration requires staged operator and stage calibration"
        return record
    if operator_result.get("status") != "complete":
        record.update(status="incomplete", reason="operator sampling has not completed")
        return record
    from calibrate_composition import run_calibration

    destination = output_dir / "composition_calibration"
    records = composition_seed_records(
        operator_result.get("candidates", []), output_dir / "stage_calibration.json", profile)
    report = run_calibration(
        args.build_dir, destination, copy.deepcopy(profile), records,
        trials=args.operator_trials, warmup=args.operator_warmup, repeat=args.operator_repeat,
        resume=args.resume_search, run=guarded_run, guard=require_exclusive_gpu)
    summary = report.get("summary", {})
    record.update(status=summary.get("status", "incomplete"),
                  path=str(destination / "composition_calibration.json"), summary=summary)
    if not summary.get("overlap_requests"):
        record["gap"] = "no cross-group overlap requests in the measured population"
    return record


def bind_static_model_identity(model, operator_result):
    """Attach observed build and direct-launch protocol without changing validation."""
    from fit_local_cost_model import is_external
    result = copy.deepcopy(model)
    records = [record for record in operator_result.get("candidates", [])
               if record.get("status") == "measured" and record.get("correct")
               and record.get("samples") and not is_external(record, record["samples"][0])]
    samples = [sample for record in records for sample in record["samples"]]
    fingerprints = {sample.get("runtime_fingerprint") for sample in samples}
    result.pop("runtime_fingerprint", None)
    result.pop("protocol", None)
    if not samples or len(fingerprints) != 1 or not all(fingerprints):
        result["static_ranking_identity"] = dict(status="unavailable",
            reason="confirmed internal samples lack one common runtime fingerprint")
        return result
    try:
        warmups = sorted({int(sample["warmup"]) for sample in samples})
        repeats = sorted({int(sample["repeat"]) for sample in samples})
        trials = sorted({len(record["samples"]) for record in records})
        if min(warmups) < 0 or min(repeats) <= 0:
            raise ValueError("invalid measurement counts")
    except (KeyError, TypeError, ValueError):
        result["static_ranking_identity"] = dict(status="unavailable",
            reason="confirmed samples lack valid warmup/repeat measurement counts")
        return result
    result["runtime_fingerprint"] = next(iter(fingerprints))
    result["protocol"] = dict(timing="direct-launch", warmup=warmups, repeat=repeats, trials=trials,
        source="cubutterfly_bench/cuntt_bench CUDA events around host launch loops; confirmed internal samples",
        applicability="direct launch service estimate; CUDA Graph replay is outside this protocol")
    result["static_ranking_identity"] = dict(status="complete", confirmed_records=len(records),
        confirmed_samples=len(samples), source="operator_calibration.json")
    return result


def main() -> int:
    args = parse_args()
    if args.trials <= 0 or args.operator_trials <= 0 or args.operator_warmup < 0 or args.operator_repeat <= 0:
        raise SystemExit("trials and repeat values must be positive")
    if args.search_budget < 0 or args.seed_budget < 0 or args.search_finalists < 1 or args.search_seconds < 0 or args.compile_seconds < 0 or args.stage_point_budget < 0:
        raise SystemExit("search budget must be nonnegative and finalists positive")
    build_dir = args.build_dir.expanduser().resolve()
    profile_dir = args.profile_dir.expanduser().resolve()
    output_dir = (args.output_dir or profile_dir).expanduser().resolve()
    args.build_dir = build_dir
    python = build_python(build_dir)
    profile_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Freeze and check CPU-only selection inputs before any GPU work. Reuse the
    # same snapshot after calibration instead of rereading a mutable registry.
    if not args.skip_operator_calibration and args.seed_budget:
        args._prepared_seed_snapshot = seed_snapshot(args, output_dir / "mapping_seed_snapshot.json")
    if args.seed_only:
        specification = json.loads(args.search_workloads.read_text())
        for workload in specification['workloads']:
            workload_search_inputs(args, args._prepared_seed_snapshot, {'operator': 'fft', **workload}, None)
    compile_budget = tempfile.TemporaryDirectory(prefix="cubutterfly-install-budget-")
    os.environ["CUBUTTERFLY_COMPILE_BUDGET_FILE"] = str(pathlib.Path(compile_budget.name)/"seconds")
    os.environ["CUBUTTERFLY_COMPILE_BUDGET_SECONDS"] = str(args.compile_seconds)

    profile_command = [python, str(PROFILE_SCRIPT), "--microbench", str(build_dir / "cuntt_hardware_microbench"), "--output-dir", str(profile_dir), "--trials", str(args.trials)]
    if args.force:
        profile_command.append("--force")
    require_exclusive_gpu()
    if args.force or not (profile_dir / "hardware_profile.json").exists():
        run(profile_command)
    run([python, str(CHECK_SCRIPT), str(profile_dir / "hardware_profile.json")])
    profile = json.loads((profile_dir / "hardware_profile.json").read_text())
    identity = next(csv.DictReader(io.StringIO(run([str(build_dir / "cubutterfly_bench"), "--device-identity"]).stdout)))
    profile = bind_device_identity(profile, identity)
    identity_path = output_dir / "calibration_device.json"
    if args.resume_search and identity_path.exists():
        old = json.loads(identity_path.read_text())
        if any(old.get(key) != profile.get(key) for key in ("device", "compute_capability", "global_memory_bytes")):
            raise ValueError("cannot resume search measurements from a different device model/SM/memory")
    ensure_static_hardware_model(profile, identity, build_dir, profile_dir, output_dir, refresh=args.force)
    os.environ["CUBUTTERFLY_HARDWARE_MODEL"] = str(profile_dir / static_hardware_profile.FILENAME)
    calibration_library_guard = _SharedLibraryGuard(build_dir)
    calibration_library_expected = calibration_library_guard.digest()

    def guarded_calibration_run(command, *, check=True):
        calibration_library_guard.assert_matches(calibration_library_expected, "before")
        result = run(command, check=check)
        calibration_library_guard.assert_matches(calibration_library_expected, "after")
        return result

    stage_sampling = None
    if args.cost_model == "staged" and not args.skip_operator_calibration:
        schedule_probe = build_dir / "cubutterfly_schedule_microbench"
        if not schedule_probe.is_file():
            raise ValueError("staged calibration requires cubutterfly_schedule_microbench; build/install that target first")
        profile["scheduling"] = calibrate_scheduling(schedule_probe, output_dir / "schedule_calibration.json",
                                                   profile, guarded_calibration_run, require_exclusive_gpu)
        stage_mode = getattr(args, "stage_calibration", "skip")
        if stage_mode != "skip":
            pipeline_probe = build_dir / "cubutterfly_pipeline_microbench"
            if not pipeline_probe.is_file():
                raise ValueError("stage calibration requires cubutterfly_pipeline_microbench; build/install that target first")
            profile["pipeline_scheduling"] = calibrate_pipeline_scheduling(
                pipeline_probe, output_dir / "pipeline_schedule_calibration.json",
                profile, guarded_calibration_run, require_exclusive_gpu)
            stage_probe = build_dir / "cubutterfly_stage_microbench"
            if not stage_probe.is_file():
                raise ValueError("independent stage calibration requires cubutterfly_stage_microbench")
            specification = json.loads(args.search_workloads.read_text())
            stage_points = []
            for workload in specification["workloads"]:
                workload = {"operator": "fft", **workload}
                candidate_binary = build_dir / ("cuntt_bench" if workload["operator"] == "ntt" else "cubutterfly_bench")
                if args.seed_only:
                    points, _ = workload_search_inputs(args, args._prepared_seed_snapshot, workload, candidate_binary)
                    stage_points.extend(points)
                else:
                    stage_points.extend(runtime_candidates(candidate_binary, workload, run))
            stage_points, stage_sampling = _bound_stage_points(stage_points, args.stage_point_budget)
            stage_sampling = _stage_sampling_scope(stage_sampling)
            if args.seed_only:
                stage_sampling.update(scope='seed-only', inventory_complete=False)
            (output_dir / "stage_point_sampling.json").write_text(
                json.dumps({"schema": "cubutterfly-stage-point-sampling-v1", **stage_sampling}, indent=2) + "\n")
            profile["stage_service"] = calibrate_stage_services(stage_probe, output_dir / "stage_calibration.json",
                profile, stage_points, lambda command: guarded_calibration_run(command, check=False), require_exclusive_gpu, mode=stage_mode,
                warmup=args.operator_warmup, repeat=args.operator_repeat, trials=args.operator_trials,
                time_budget=args.search_seconds if stage_mode == "bounded" else 0,
                import_checkpoints=getattr(args, "stage_import_checkpoint", []),
                resume_build_priors=getattr(args, "resume_build_priors", False))
            profile["stage_service"] = _annotate_stage_service_scope(profile["stage_service"], stage_sampling)
            write_checkpoint(output_dir / "stage_service_profile.json", profile["stage_service"])
        write_checkpoint(output_dir / "calibration_device.json", profile)
        if profile.get("stage_service", {}).get("calibration_status") == "interrupted":
            raise InterruptedError("stage calibration interrupted; checkpoints saved; resume before search or publication")
    rank_rows = rank_model(profile, output_dir / "unfolding_rank.csv")

    cache_text = (build_dir / "CMakeCache.txt").read_text() if (build_dir / "CMakeCache.txt").exists() else ""
    inventory = run([str(build_dir / "cubutterfly_bench"), "--list-processing-units"], check=False)
    cufftdx = inventory.returncode == 0 and len(list(csv.DictReader(io.StringIO(inventory.stdout)))) > 0
    operator_result: dict[str, Any]
    if args.skip_operator_calibration:
        operator_result = {"status": "skipped", "cufftdx_enabled": cufftdx, "candidates": []}
    else:
        operator_result = run_operator_calibration(args, output_dir / "operator_calibration.json", cufftdx)

    composition_calibration = _composition_calibration(
        args, output_dir, profile, operator_result, guarded_calibration_run)

    cost_model_path = output_dir / "cost_model.json"
    if args.skip_operator_calibration:
        cost_model_path.write_text(json.dumps({
            "schema": "cubutterfly-local-cost-model-v1",
            "status": "skipped",
            "device": profile["device"],
            "compute_capability": profile["compute_capability"],
            "reason": "operator calibration was skipped",
        }, indent=2) + "\n")
    else:
        run([
            python, str(SCRIPT_DIR / "stage_cost_model.py" if args.cost_model == "staged" else COST_MODEL_SCRIPT),
            "--profile", str(output_dir / "calibration_device.json" if args.cost_model == "staged" else profile_dir / "hardware_profile.json"),
            "--operator-calibration", str(output_dir / "operator_calibration.json"),
            "--output", str(cost_model_path),
        ])
        if args.cost_model == "staged":
            cost_model = json.loads(cost_model_path.read_text())
            write_checkpoint(cost_model_path, bind_static_model_identity(cost_model, operator_result))

    local_selector_header = build_dir / "generated" / "generated_local_selector.hpp"
    local_selector_points = write_local_selector_header(profile, operator_result, local_selector_header) if args.embed_legacy_selector else 0
    # Dynamic registry is cumulative across semantic cells and GPU capacities.
    # A skipped calibration does not erase any previously verified records.
    registry = default_path()
    registry_points = 0 if args.skip_operator_calibration else promote(registry, profile, operator_result)

    manifest = {
        "schema": "cubutterfly-local-calibration-v1",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "device": profile["device"],
        "compute_capability": profile["compute_capability"],
        "build_dir": str(build_dir),
        "profile": str(profile_dir / "hardware_profile.json"),
        "hardware_model": str(profile_dir / static_hardware_profile.FILENAME),
        "unfolding_rank": str(output_dir / "unfolding_rank.csv"),
        "unfolding_rank_rows": rank_rows,
        "operator_calibration": str(output_dir / "operator_calibration.json") if not args.skip_operator_calibration else "",
        "operator_calibration_summary": str(output_dir / "operator_calibration.csv") if not args.skip_operator_calibration else "",
        "operator_recommendations": str(output_dir / "operator_recommendations.json") if not args.skip_operator_calibration else "",
        "cost_model": str(cost_model_path),
        "schedule_calibration": str(output_dir / "schedule_calibration.json") if (output_dir / "schedule_calibration.json").exists() else "",
        "stage_calibration_mode": getattr(args, "stage_calibration", "skip"),
        "composition_calibration": composition_calibration,
        "stage_service_profile": str(output_dir / "stage_service_profile.json") if profile.get("stage_service") else "",
        "stage_point_sampling": str(output_dir / "stage_point_sampling.json") if stage_sampling else "",
        "stage_point_sampling_summary": stage_sampling or {},
        "stage_inventory_complete": (profile.get("stage_service", {}).get("inventory_complete")
                                      if stage_sampling else None),
        "stage_service_calibration_status": (profile.get("stage_service", {}).get("calibration_status")
                                              if profile.get("stage_service") else None),
        "search_coverage": str(output_dir / "search_coverage.json") if not args.skip_operator_calibration else "",
        "global_memory_bytes": profile["global_memory_bytes"],
        "theory_alignment": "partial; see docs/theory_alignment_audit.md",
        "operator_result": operator_result,
        "operator_families": operator_result.get("operator_families", []),
        "local_selector_header": str(local_selector_header),
        "local_selector_points": local_selector_points,
        "registry_path": str(registry),
        "registry_promoted_records": registry_points,
        "runtime_auto_select": "hardware-registry-with-portable-fallback",
        "status": "ready-with-measured-mappings" if registry_points else "ready-with-unmeasured-fallback",
    }
    (output_dir / "calibration_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"local calibration: {manifest['status']}")
    print(f"profile: {manifest['profile']}")
    print(f"unfolding rank: {manifest['unfolding_rank']} ({rank_rows} rows)")
    print(f"calibration manifest: {output_dir / 'calibration_manifest.json'}")
    print(f"cost model: {cost_model_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except InterruptedError as error:
        print(f"interrupted: {error}", file=sys.stderr)
        raise SystemExit(130)
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
