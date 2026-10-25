#!/usr/bin/env python3
"""Fixed 64-case xD C2C population with guarded, resumable execution."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
POPULATION = "xd-v11-fixed64"
SCHEMA = 1
REPORT_PROTOCOL = "20-call CUDA graph; 7 replays; planning, warmup, checking excluded"
_MISSING = object()


class ResumeError(ValueError):
    """Raised when a resume source cannot be safely paired with this run."""


def cases() -> list[dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}

    def add(shape: tuple[int, ...], batch: int, precision: str,
            inverse: bool, group: str) -> None:
        name = f'{"x".join(map(str, shape))}_b{batch}_{precision}_{"inv" if inverse else "fwd"}'
        if name not in rows:
            rows[name] = dict(id=name, shape=list(shape), batch=batch,
                              precision=precision, inverse=inverse, groups=[])
        rows[name]["groups"].append(group)

    original = [((16, 16), 16), ((64, 64), 16), ((256, 256), 16), ((1024, 1024), 1),
                ((8, 8, 8), 16), ((16, 32, 32), 16), ((64, 64, 64), 16),
                ((128, 128, 128), 1)]
    for shape, batch in original:
        for precision in ("fp32", "fp64"):
            add(shape, batch, precision, False, "original16")
    for shapes, batches in [([(64, 64), (256, 256)], [1, 16, 64]),
                            ([(32, 32, 32), (64, 64, 64)], [1, 4, 16]),
                            ([(128, 128, 128)], [1, 4])]:
        for shape in shapes:
            for batch in batches:
                for precision in ("fp32", "fp64"):
                    for inverse in (False, True):
                        add(shape, batch, precision, inverse, "factorial-extension")
    return list(rows.values())


def _canonical(value: Any) -> str:
    if value is _MISSING:
        return "<missing>"
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalize_uuid(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.startswith(("GPU-", "MIG-")):
        return text.lower()
    if len(text) >= 8 and all(character in "0123456789abcdefABCDEF-" for character in text):
        return ("GPU-" + text).lower()
    return text.lower()


def _query_gpu_uuids() -> list[str]:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader,nounits"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
    except OSError:
        return []
    if result.returncode:
        return []
    return [uuid for uuid in (_normalize_uuid(line) for line in result.stdout.splitlines()) if uuid]


def _select_gpu_uuid(visible: str | None, physical: list[str]) -> str | None:
    tokens = [token.strip() for token in visible.split(",")] if visible else []
    first = tokens[0] if tokens else None
    if visible is not None and first in (None, "", "-1"):
        return None
    if first and first != "-1":
        explicit = _normalize_uuid(first) if not first.isdigit() else None
        if explicit is not None:
            return explicit
        try:
            index = int(first)
        except ValueError:
            index = -1
        if 0 <= index < len(physical):
            return physical[index]
        return None
    return physical[0] if physical else None


def _runtime_identity(*, require_gpu: bool) -> dict[str, Any]:
    """Resolve the physical UUID and execution policy for this process."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    physical = _query_gpu_uuids() if require_gpu else []
    selected = _select_gpu_uuid(visible, physical)
    if require_gpu and selected is None:
        raise ResumeError(
            "cannot resolve GPU UUID; nvidia-smi is unavailable or CUDA_VISIBLE_DEVICES "
            "selects no device"
        )
    return {
        "gpu_uuid": selected,
        "cuda_visible_devices": visible,
        "cuda_visible_uuid": selected if visible is not None else None,
        "compile_mode": os.environ.get("CUBUTTERFLY_COMPILE_MODE"),
    }


def _make_manifest(args: argparse.Namespace, rows: list[dict[str, Any]],
                   runtime: Mapping[str, Any]) -> dict[str, Any]:
    library = args.library.resolve()
    manifest: dict[str, Any] = dict(
        schema=SCHEMA, cases=rows, population=POPULATION, library=str(library),
        library_sha256=_sha256(library),
        baselines=["cufft-native"] + (["vkfft"] if args.vkfft_library else []),
        gpu_uuid=runtime.get("gpu_uuid"),
        cuda_visible_devices=runtime.get("cuda_visible_devices"),
        cuda_visible_uuid=runtime.get("cuda_visible_uuid"),
        compile_mode=runtime.get("compile_mode"), timing_protocol=REPORT_PROTOCOL,
    )
    if args.vkfft_library:
        manifest["vkfft_library_sha256"] = _sha256(args.vkfft_library.resolve())
    return manifest


def _load_object(path: Path, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ResumeError(f"cannot read resume {label} {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise ResumeError(f"resume {label} is not a JSON object: {path}")
    return value


def _validate_source_manifest(source: Mapping[str, Any], current: Mapping[str, Any]) -> None:
    for key in ("schema", "population", "cases", "library_sha256", "baselines",
                "vkfft_library_sha256"):
        if _canonical(source.get(key, _MISSING)) != _canonical(current.get(key, _MISSING)):
            raise ResumeError(
                f"resume manifest mismatch for {key}: "
                f"source={source.get(key)!r}, current={current.get(key)!r}"
            )
    # Old interrupted manifests predate these fields; their reports still
    # undergo the per-file GPU and compile-policy checks below.
    for key in ("gpu_uuid", "cuda_visible_devices", "cuda_visible_uuid", "compile_mode"):
        if key in source and source.get(key) != current.get(key):
            raise ResumeError(
                f"resume environment mismatch for {key}: "
                f"source={source.get(key)!r}, current={current.get(key)!r}"
            )


def _report_gpu_uuid(report: Mapping[str, Any]) -> str | None:
    value = report.get("gpu_uuid")
    if value is None and isinstance(report.get("gpu"), Mapping):
        gpu = report["gpu"]
        value = gpu.get("gpu_uuid", gpu.get("uuid"))
    return _normalize_uuid(value)


def _contract_matches(case: Mapping[str, Any], report: Mapping[str, Any]) -> bool:
    shape = report.get("shape")
    if not isinstance(shape, list):
        return False
    return (
        shape == list(case.get("shape", ()))
        and report.get("batch") == case.get("batch")
        and report.get("precision") == case.get("precision")
        and report.get("inverse") == case.get("inverse")
    )


def _validate_report_identity(report: Mapping[str, Any], case: Mapping[str, Any],
                              manifest: Mapping[str, Any],
                              runtime: Mapping[str, Any]) -> None:
    expected_library = manifest.get("library_sha256")
    if report.get("library_sha256") != expected_library:
        raise ResumeError(
            f"case {case['id']} library SHA mismatch: expected {expected_library}, "
            f"got {report.get('library_sha256')}"
        )
    expected_gpu = _normalize_uuid(runtime.get("gpu_uuid"))
    actual_gpu = _report_gpu_uuid(report)
    if expected_gpu is None or actual_gpu != expected_gpu:
        raise ResumeError(
            f"case {case['id']} gpu UUID mismatch: expected {expected_gpu}, got {actual_gpu}"
        )
    expected_mode = runtime.get("compile_mode")
    if report.get("compile_mode", _MISSING) != expected_mode:
        raise ResumeError(
            f"case {case['id']} compile_mode mismatch: expected {expected_mode!r}, "
            f"got {report.get('compile_mode', _MISSING)!r}"
        )
    expected_vkfft = manifest.get("vkfft_library_sha256")
    if expected_vkfft is None:
        return
    rows = report.get("rows")
    if not isinstance(rows, list):
        return
    for row in rows:
        if not isinstance(row, Mapping) or row.get("requested") != "vkfft":
            continue
        metadata = row.get("baseline_metadata")
        actual = metadata.get("library_sha256") if isinstance(metadata, Mapping) else None
        if actual != expected_vkfft:
            raise ResumeError(
                f"case {case['id']} VkFFT library SHA mismatch: expected {expected_vkfft}, "
                f"got {actual}"
            )


def _complete_report(report: Mapping[str, Any], case: Mapping[str, Any],
                     manifest: Mapping[str, Any], runtime: Mapping[str, Any]) -> bool:
    if not _contract_matches(case, report):
        return False
    _validate_report_identity(report, case, manifest, runtime)
    if report.get("status") != "passed" or report.get("protocol") != REPORT_PROTOCOL:
        return False
    rows = report.get("rows")
    if not isinstance(rows, list):
        return False
    expected = ["selected", *manifest.get("baselines", [])]
    by_name: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if isinstance(row, Mapping) and isinstance(row.get("requested"), str):
            by_name.setdefault(row["requested"], row)
    if any(name not in by_name for name in expected):
        return False
    for name in expected:
        row = by_name[name]
        if row.get("status") != "passed" or row.get("input_preserved") is not True:
            return False
        if not isinstance(row.get("correctness"), Mapping):
            return False
        if not isinstance(row.get("graph_correctness"), Mapping):
            return False
        graph = row.get("graph")
        if not isinstance(graph, Mapping):
            return False
        if graph.get("graph_used") is not True:
            return False
        try:
            calls = int(graph.get("graph_capture_calls"))
            replays = int(graph.get("graph_replays"))
        except (TypeError, ValueError):
            return False
        if calls != 20 or replays != 7:
            return False
        samples = graph.get("samples_ms")
        if not isinstance(samples, list) or len(samples) != 7:
            return False
        if any(not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0
               for value in samples):
            return False
        median = graph.get("median_ms")
        if (not isinstance(median, (int, float)) or not math.isfinite(median)
                or median <= 0):
            return False
    return True


def _reusable_case(path: Path, case: Mapping[str, Any], manifest: Mapping[str, Any],
                   runtime: Mapping[str, Any], cutoff: float) -> bool:
    try:
        if path.stat().st_mtime >= cutoff:
            return False
    except OSError:
        return False
    try:
        report = _load_object(path, "case report")
    except ResumeError:
        return False
    return _complete_report(report, case, manifest, runtime)


def _parse_cutoff(value: str) -> tuple[float, str]:
    text = value.strip()
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ResumeError(f"--resume-before must be ISO-8601 with timezone: {value!r}") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ResumeError(f"--resume-before must include a timezone: {value!r}")
    return parsed.timestamp(), text


def _diagnostic_command(args: argparse.Namespace, manifest: Mapping[str, Any],
                        case: Mapping[str, Any], output: Path) -> list[str]:
    command = [sys.executable, str(ROOT / "scripts" / "diagnose_xd_fft.py"),
               "--library", str(args.library.resolve()), "--shape", *map(str, case["shape"]),
               "--batch", str(case["batch"]), "--precision", case["precision"],
               "--lowerings", "selected", *manifest["baselines"], "--output", str(output)]
    if case["inverse"]:
        command.append("--inverse")
    if args.vkfft_library:
        command += ["--vkfft-library", str(args.vkfft_library.resolve())]
    return command


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", required=True, type=Path)
    parser.add_argument("--vkfft-library", type=Path)
    parser.add_argument("--output", type=Path, default=os.getenv("XD_REPORT_DIR"))
    parser.add_argument("--manifest-only", action="store_true")
    parser.add_argument("--resume-from", type=Path,
                        help="previous fixed64 output directory to reuse")
    parser.add_argument("--resume-before",
                        help="only reuse reports strictly older than this ISO-8601 cutoff")
    args = parser.parse_args(argv)
    if args.output is None:
        parser.error("--output or XD_REPORT_DIR is required")
    if (args.resume_from is None) != (args.resume_before is None):
        parser.error("--resume-from and --resume-before must be supplied together")
    if args.manifest_only and args.resume_from is not None:
        parser.error("--manifest-only cannot be combined with resume")
    return args


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    rows = cases()
    assert len(rows) == 64
    if not args.library.is_file():
        raise ResumeError(f"xD library not found: {args.library}")
    if args.vkfft_library is not None and not args.vkfft_library.is_file():
        raise ResumeError(f"VkFFT library not found: {args.vkfft_library}")

    runtime = _runtime_identity(require_gpu=not args.manifest_only)
    manifest = _make_manifest(args, rows, runtime)
    source_dir: Path | None = None
    cutoff: float | None = None
    cutoff_text: str | None = None
    if args.resume_from is not None:
        source_dir = args.resume_from.expanduser().resolve()
        if source_dir.is_file() and source_dir.name == "manifest.json":
            source_dir = source_dir.parent
        if source_dir == args.output.expanduser().resolve():
            raise ResumeError("resume output must be a fresh directory, not --resume-from itself")
        source_manifest = _load_object(source_dir / "manifest.json", "manifest")
        _validate_source_manifest(source_manifest, manifest)
        cutoff, cutoff_text = _parse_cutoff(args.resume_before)

    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if args.manifest_only:
        print(f"fixed population: {len(rows)} cases")
        return 0

    failures: list[dict[str, Any]] = []
    reused: list[dict[str, Any]] = []
    rerun_ids: list[str] = []
    for index, case in enumerate(rows):
        case_output = output / (case["id"] + ".json")
        source_report = source_dir / case_output.name if source_dir is not None else None
        if source_report is not None and cutoff is not None and _reusable_case(
                source_report, case, manifest, runtime, cutoff):
            shutil.copy2(source_report, case_output)
            reused.append({"id": case["id"], "source": str(source_report),
                           "cutoff": cutoff_text})
            print(f"{index + 1}/{len(rows)} {case['id']} reused", flush=True)
            continue

        rerun_ids.append(case["id"])
        command = _diagnostic_command(args, manifest, case, case_output)
        print(f"{index + 1}/{len(rows)} {case['id']}", flush=True)
        with (output / (case["id"] + ".log")).open("w", encoding="utf-8") as log:
            result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            failures.append(dict(id=case["id"], returncode=result.returncode))

    completion: dict[str, Any] = dict(completed=len(rows), failures=failures)
    if source_dir is not None:
        completion.update(reused_ids=[entry["id"] for entry in reused],
                          resume_from=str(source_dir), resume_before=cutoff_text)
        (output / "resume.json").write_text(
            json.dumps({"source": str(source_dir), "cutoff": cutoff_text,
                        "reused": reused,
                        "reused_ids": [entry["id"] for entry in reused],
                        "rerun_ids": rerun_ids}, indent=2) + "\n", encoding="utf-8")
    (output / "completion.json").write_text(
        json.dumps(completion, indent=2) + "\n", encoding="utf-8")
    # Preserve complete coverage including unsupported baselines. The summary
    # must explicitly count failures and cannot silently shrink the population.
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ResumeError as exc:
        print(f"benchmark_xd_expanded: {exc}", file=sys.stderr)
        raise SystemExit(2)
