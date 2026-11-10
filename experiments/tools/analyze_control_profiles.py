#!/usr/bin/env python3
"""CPU-only audit of the paper controls and the two exact Nsight traces.

The Nsight reports are exported to SQLite before analysis.  This tool never
launches the benchmark or any CUDA process.  Nsight kernel timings are kept
separate from the normal controlled timing rows; the former are a trace
diagnostic and are not substituted into the latter.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import itertools
import json
import math
import re
import sqlite3
import statistics
import subprocess
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_SCHEMA = "cubutterfly-control-profile-analysis-v1"
PROFILE_SCHEMA = "cubutterfly-controlled-profile-v1"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    raw = path.read_bytes()
    return json.loads(raw), {
        "path": str(path.resolve()),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False) + "\n")


def relative_path(path, root):
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return None


def file_record(path, root, role=None):
    path = Path(path).resolve()
    result = {
        "path": str(path),
        "repo_relative": relative_path(path, root),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }
    if role:
        result["role"] = role
    return result


def finite(value):
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def stats(values):
    values = [float(value) for value in values if finite(value) is not None]
    if not values:
        return {"count": 0, "sum": None, "min": None, "median": None, "max": None}
    return {
        "count": len(values),
        "sum": sum(values),
        "min": min(values),
        "median": statistics.median(values),
        "max": max(values),
    }


def ns_stats(values):
    value = stats(values)
    return value


def ratio_summary(values):
    values = [float(value) for value in values if finite(value) is not None]
    if not values:
        return {
            "count": 0,
            "treatment_faster": 0,
            "treatment_slower": 0,
            "equal": 0,
            "min": None,
            "median": None,
            "max": None,
        }
    return {
        "count": len(values),
        "treatment_faster": sum(value > 1.0 for value in values),
        "treatment_slower": sum(value < 1.0 for value in values),
        "equal": sum(value == 1.0 for value in values),
        "min": min(values),
        "median": statistics.median(values),
        "max": max(values),
    }


def normalized_uuid(value):
    if value is None:
        return None
    text = str(value).strip().lower()
    return text[4:] if text.startswith("gpu-") else text


def safe_json(value):
    return json.loads(json.dumps(value, ensure_ascii=True, allow_nan=False))


def sqlite_table_exists(connection, name):
    row = connection.execute(
        "select 1 from sqlite_master where type='table' and name=?", (name,)
    ).fetchone()
    return row is not None


def sqlite_string_ids(connection):
    if not sqlite_table_exists(connection, "StringIds"):
        return {}
    return {
        int(row[0]): row[1]
        for row in connection.execute("select id,value from StringIds")
    }


def metadata_rows(connection, table):
    if not sqlite_table_exists(connection, table):
        return []
    return connection.execute("select name,value from " + table).fetchall()


def metadata_protocol(capture):
    arguments = {}
    for key, value in capture:
        match = re.fullmatch(r"PROCESS_0:ARGUMENT_(\d+)", str(key))
        if match:
            arguments[int(match.group(1))] = str(value)
    warmup = None
    repeat = None
    for index, value in arguments.items():
        if value == "--warmup" and index + 1 in arguments:
            try:
                warmup = int(arguments[index + 1])
            except ValueError:
                pass
        if value == "--repeat" and index + 1 in arguments:
            try:
                repeat = int(arguments[index + 1])
            except ValueError:
                pass
    command = capture_value(capture, "PROCESS_0:COMMAND")
    app_command = ([command] if command else []) + [arguments[index] for index in sorted(arguments)]
    return {
        "warmup": warmup,
        "repeat": repeat,
        "arguments": [arguments[index] for index in sorted(arguments)],
        "command": app_command,
    }


def capture_value(rows, key, default=None):
    for name, value in rows:
        if name == key:
            return value
    return default


def metadata_values(rows, key):
    return [value for name, value in rows if name == key]


def parse_kernel_stage(name):
    if not name:
        return None
    values = [int(value) for value in re.findall(r"\(int\)(\d+)", name)]
    if len(values) < 3:
        return None
    return {
        "logN": values[0],
        "first_stage": values[1],
        "stage_count": values[2],
        "label": "stage%d-%d" % (values[1], values[1] + values[2]),
    }


def union_length(intervals):
    total = 0
    current_start = None
    current_end = None
    for start, end in sorted(intervals):
        if end < start:
            end = start
        if current_start is None:
            current_start, current_end = start, end
        elif start > current_end:
            total += current_end - current_start
            current_start, current_end = start, end
        else:
            current_end = max(current_end, end)
    if current_start is not None:
        total += current_end - current_start
    return total


def positive_gaps(intervals):
    cursor = None
    gaps = []
    for start, end in sorted(intervals):
        if cursor is not None and start > cursor:
            gaps.append(start - cursor)
        cursor = end if cursor is None else max(cursor, end)
    return gaps


def stream_gaps(kernels):
    gaps = {}
    for stream_id in sorted(set(kernel["streamId"] for kernel in kernels)):
        rows = sorted(
            [kernel for kernel in kernels if kernel["streamId"] == stream_id],
            key=lambda kernel: (kernel["start"], kernel["end"]),
        )
        gaps[stream_id] = [
            max(0, right["start"] - left["end"])
            for left, right in zip(rows, rows[1:])
        ]
    return gaps


def api_category(name):
    name = name or ""
    if "cudaLaunch" in name:
        return "launch"
    if "cudaEvent" in name or "cudaStream" in name:
        return "event_stream"
    if "cudaMalloc" in name or "cudaFree" in name or "cudaMemcpy" in name:
        return "memory"
    if name.startswith("cu"):
        return "driver"
    return "other"


def duration_ns(rows):
    return sum(max(0, int(row["end"]) - int(row["start"])) for row in rows)


def summarize_gaps(gaps):
    return {
        "count": len(gaps),
        "total_ns": sum(gaps),
        "min_ns": min(gaps) if gaps else None,
        "median_ns": statistics.median(gaps) if gaps else None,
        "max_ns": max(gaps) if gaps else None,
        "values_ns": list(gaps),
    }


def clipped_duration(row, start, end):
    left = max(int(row["start"]), int(start))
    right = min(int(row["end"]), int(end))
    return max(0, right - left)


def api_subset_summary(rows):
    by_category = defaultdict(list)
    by_name = defaultdict(list)
    for row in rows:
        by_category[api_category(row.get("name"))].append(row)
        by_name[row.get("name") or "<unknown>"].append(row)
    return {
        "count": len(rows),
        "duration_ns": duration_ns(rows),
        "by_category": {
            category: {
                "count": len(category_rows),
                "duration_ns": duration_ns(category_rows),
            }
            for category, category_rows in sorted(by_category.items())
        },
        "by_name": {
            name: {"count": len(name_rows), "duration_ns": duration_ns(name_rows)}
            for name, name_rows in sorted(by_name.items(), key=lambda pair: -duration_ns(pair[1]))
        },
    }


def invocation_metrics(kernels, runtime_by_correlation, index, total_kernel_count):
    kernels = sorted(kernels, key=lambda kernel: (kernel["start"], kernel["end"]))
    starts = [kernel["start"] for kernel in kernels]
    ends = [kernel["end"] for kernel in kernels]
    first_start, last_end = min(starts), max(ends)
    intervals = [(kernel["start"], kernel["end"]) for kernel in kernels]
    union_ns = union_length(intervals)
    total_ns = sum(kernel["end"] - kernel["start"] for kernel in kernels)
    pair_rows = []
    for left, right in itertools.combinations(kernels, 2):
        overlap = max(0, min(left["end"], right["end"]) - max(left["start"], right["start"]))
        if overlap:
            pair_rows.append({
                "left_row": left["row_index"],
                "right_row": right["row_index"],
                "left_stream": left["streamId"],
                "right_stream": right["streamId"],
                "overlap_ns": overlap,
            })
    launch_rows = []
    launch_delays = []
    for kernel in kernels:
        api = runtime_by_correlation.get(kernel.get("correlationId"))
        if api is not None:
            launch_rows.append(api)
            launch_delays.append(kernel["start"] - api["end"])
    gaps_by_stream = stream_gaps(kernels)
    flat_stream_gaps = [gap for gaps in gaps_by_stream.values() for gap in gaps]
    global_gaps = positive_gaps(intervals)
    stage_counts = defaultdict(int)
    for kernel in kernels:
        stage = parse_kernel_stage(kernel.get("demangled"))
        stage_counts[stage["label"] if stage else "unknown"] += 1
    return {
        "index": index,
        "kind": "warmup" if index == 0 else "timed-repeat",
        "ordinal": index,
        "kernel_rows": [kernel["row_index"] for kernel in kernels],
        "kernel_count": len(kernels),
        "stream_ids": sorted(set(kernel["streamId"] for kernel in kernels)),
        "stage_counts": dict(sorted(stage_counts.items())),
        "start_ns": first_start,
        "end_ns": last_end,
        "span_ns": last_end - first_start,
        "kernel_duration_sum_ns": total_ns,
        "kernel_union_ns": union_ns,
        "global_gap": summarize_gaps(global_gaps),
        "same_stream_gap": summarize_gaps(flat_stream_gaps),
        "same_stream_gap_by_stream": {
            str(stream): summarize_gaps(gaps)
            for stream, gaps in sorted(gaps_by_stream.items())
        },
        "overlap": {
            "pair_count": len(pair_rows),
            "pair_overlap_ns": sum(row["overlap_ns"] for row in pair_rows),
            "same_stream_pair_count": sum(row["left_stream"] == row["right_stream"] for row in pair_rows),
            "same_stream_overlap_ns": sum(
                row["overlap_ns"]
                for row in pair_rows
                if row["left_stream"] == row["right_stream"]
            ),
            "cross_stream_pair_count": sum(row["left_stream"] != row["right_stream"] for row in pair_rows),
            "cross_stream_overlap_ns": sum(
                row["overlap_ns"]
                for row in pair_rows
                if row["left_stream"] != row["right_stream"]
            ),
            "active_overlap_ns": total_ns - union_ns,
            "pair_rows": pair_rows,
        },
        "launch_api": {
            "count": len(launch_rows),
            "duration_ns": duration_ns(launch_rows),
            "launch_to_kernel_delay": ns_stats(launch_delays),
        },
        "expected_kernel_count_total": total_kernel_count,
    }


def load_sqlite_profile(record_path, rep_path, sqlite_path, root):
    record, record_source = read_json(record_path)
    if record.get("schema") != PROFILE_SCHEMA:
        raise ValueError("unexpected profile record schema: %s" % record_path)

    connection = sqlite3.connect(str(sqlite_path))
    connection.row_factory = sqlite3.Row
    strings = sqlite_string_ids(connection)
    capture_rows = metadata_rows(connection, "META_DATA_CAPTURE")
    export_rows = metadata_rows(connection, "META_DATA_EXPORT")
    metadata = metadata_protocol(capture_rows)
    metadata_uuid = capture_value(
        capture_rows,
        "PROFILING_SESSION_UUID",
        capture_value(capture_rows, "PROFILING_UUID"),
    )

    kernels = []
    for row_index, row in enumerate(
        connection.execute("select rowid as row_index,* from CUPTI_ACTIVITY_KIND_KERNEL order by start,end,rowid")
    ):
        kernel = dict(row)
        kernel["row_index"] = row_index
        kernel["demangled"] = strings.get(kernel.get("demangledName"), "")
        kernel["short"] = strings.get(kernel.get("shortName"), "")
        kernels.append(kernel)
    runtimes = []
    for row in connection.execute(
        "select rowid as row_index,* from CUPTI_ACTIVITY_KIND_RUNTIME order by start,end,rowid"
    ):
        runtime = dict(row)
        runtime["name"] = strings.get(runtime.get("nameId"), "")
        runtimes.append(runtime)
    runtime_by_correlation = {
        runtime.get("correlationId"): runtime
        for runtime in runtimes
        if "cudaLaunch" in runtime.get("name", "")
    }

    if not kernels:
        raise ValueError("no CUDA kernels in exported profile: %s" % sqlite_path)
    device_ids = sorted(set(kernel["deviceId"] for kernel in kernels))
    stream_ids = sorted(set(kernel["streamId"] for kernel in kernels))
    device_rows = []
    if sqlite_table_exists(connection, "TARGET_INFO_GPU"):
        device_rows = [dict(row) for row in connection.execute("select * from TARGET_INFO_GPU")]
    target_device = next((row for row in device_rows if row.get("id") == device_ids[0]), None)
    context_rows = []
    if sqlite_table_exists(connection, "TARGET_INFO_CUDA_CONTEXT_INFO"):
        context_rows = [dict(row) for row in connection.execute("select * from TARGET_INFO_CUDA_CONTEXT_INFO")]
    process_rows = []
    if sqlite_table_exists(connection, "PROCESSES"):
        process_rows = [dict(row) for row in connection.execute("select * from PROCESSES")]
    process_id = context_rows[0].get("processId") if context_rows else None
    process = next((row for row in process_rows if row.get("pid") == process_id), None)

    expected_protocol = record.get("protocol", {})
    metadata_protocol_values = {
        "warmup": metadata.get("warmup"),
        "repeat": metadata.get("repeat"),
    }
    protocol_issues = []
    for field in ("warmup", "repeat"):
        if expected_protocol.get(field) != metadata_protocol_values.get(field):
            protocol_issues.append(
                "%s record=%r metadata=%r"
                % (field, expected_protocol.get(field), metadata_protocol_values.get(field))
            )
    invocation_count = expected_protocol.get("warmup", 0) + expected_protocol.get("repeat", 0)
    if not invocation_count or len(kernels) % invocation_count:
        protocol_issues.append(
            "kernel count %d is not divisible by invocation count %d"
            % (len(kernels), invocation_count)
        )
        kernels_per_invocation = len(kernels)
    else:
        kernels_per_invocation = len(kernels) // invocation_count
    invocations = []
    for index in range(invocation_count or 1):
        begin = index * kernels_per_invocation
        end = min(len(kernels), (index + 1) * kernels_per_invocation)
        if begin >= len(kernels):
            break
        invocations.append(
            invocation_metrics(kernels[begin:end], runtime_by_correlation, index, len(kernels))
        )
    all_intervals = [(kernel["start"], kernel["end"]) for kernel in kernels]
    all_total_ns = sum(kernel["end"] - kernel["start"] for kernel in kernels)
    all_union_ns = union_length(all_intervals)
    all_pair_rows = []
    for left, right in itertools.combinations(kernels, 2):
        overlap = max(0, min(left["end"], right["end"]) - max(left["start"], right["start"]))
        if overlap:
            all_pair_rows.append((left, right, overlap))
    first_kernel_start = min(kernel["start"] for kernel in kernels)
    last_kernel_end = max(kernel["end"] for kernel in kernels)
    span_api_rows = [
        row
        for row in runtimes
        if row["end"] > first_kernel_start and row["start"] < last_kernel_end
    ]
    clipped_api = []
    for row in span_api_rows:
        copy_row = dict(row)
        copy_row["start"] = max(row["start"], first_kernel_start)
        copy_row["end"] = min(row["end"], last_kernel_end)
        clipped_api.append(copy_row)

    timed_start_index = int(expected_protocol.get("warmup", 0) or 0)
    timed_invocations = invocations[timed_start_index:]
    timed_kernels = kernels[timed_start_index * kernels_per_invocation :]
    timed_launch_rows = [
        runtime_by_correlation.get(kernel.get("correlationId"))
        for kernel in timed_kernels
    ]
    timed_launch_rows = [row for row in timed_launch_rows if row is not None]
    timed_intervals = [(kernel["start"], kernel["end"]) for kernel in timed_kernels]
    nsys_device = {
        key: target_device.get(key)
        for key in (
            "id", "name", "uuid", "busLocation", "totalMemory", "computeMajor", "computeMinor",
            "smCount", "memoryBandwidth", "clockRate", "cuDevice",
        )
        if target_device is not None and key in target_device
    }
    identity = record.get("identity", {})
    identity_device = identity.get("device", {}) if isinstance(identity, dict) else {}
    identity_uuid = identity_device.get("uuid", identity.get("gpu_uuid"))
    nsys_uuid = nsys_device.get("uuid")
    profile_result = {
        "record": record_source,
        "report": file_record(rep_path, root, "Nsight Systems report"),
        "sqlite_export": file_record(sqlite_path, root, "nsys export --type sqlite output"),
        "profile_record_source": {
            "declared_source": record.get("source"),
            "declared_source_sha256": record.get("source_sha256"),
            "declared_source_sha256_matches": (
                Path(record["source"]).exists()
                and sha256(Path(record["source"])) == record.get("source_sha256")
            ) if record.get("source") else False,
        },
        "metadata": {
            "application_command": metadata.get("command"),
            "arguments": metadata.get("arguments"),
            "warmup": metadata.get("warmup"),
            "repeat": metadata.get("repeat"),
            "profiling_uuid": metadata_uuid,
            "export": dict(export_rows),
            "process": process,
        },
        "device": {
            "nsys_device_id": device_ids[0],
            "kernel_device_ids": device_ids,
            "target_info": nsys_device,
            "uuid": "GPU-" + nsys_uuid if nsys_uuid else None,
            "record_identity_uuid": identity_uuid,
            "uuid_match": normalized_uuid(nsys_uuid) == normalized_uuid(identity_uuid),
            "context_info": context_rows,
        },
        "scope": {
            "case_id": record.get("case_id"),
            "variant_id": record.get("variant_id"),
            "mapping": safe_json(record.get("mapping", {})),
            "protocol_record": safe_json(expected_protocol),
            "protocol_metadata": metadata_protocol_values,
            "protocol_issues": protocol_issues,
            "kernels_per_invocation": kernels_per_invocation,
            "invocation_count": len(invocations),
            "warmup_invocations": min(timed_start_index, len(invocations)),
            "timed_invocations": max(0, len(invocations) - timed_start_index),
            "normal_timing_accepted": bool(record.get("normal_timing_accepted")),
        },
        "kernels": {
            "total_count": len(kernels),
            "timed_count": len(timed_kernels),
            "streams": stream_ids,
            "stream_count": len(stream_ids),
            "short_name_counts": dict(sorted(
                ((name, sum(kernel["short"] == name for kernel in kernels))
                 for name in sorted(set(kernel["short"] for kernel in kernels)))
            )),
            "demangled_name_counts": dict(sorted(
                ((name, sum(kernel["demangled"] == name for kernel in kernels))
                 for name in sorted(set(kernel["demangled"] for kernel in kernels)))
            )),
            "stage_counts": dict(sorted(
                (label, sum(
                    (parse_kernel_stage(kernel.get("demangled")) or {"label": "unknown"})["label"] == label
                    for kernel in kernels
                ))
                for label in sorted(set(
                    (parse_kernel_stage(kernel.get("demangled")) or {"label": "unknown"})["label"]
                    for kernel in kernels
                ))
            )),
            "total_duration_ns": all_total_ns,
            "union_active_ns": all_union_ns,
            "span_ns": last_kernel_end - first_kernel_start,
            "global_gap": summarize_gaps(positive_gaps(all_intervals)),
            "overlap": {
                "pair_count": len(all_pair_rows),
                "pair_overlap_ns": sum(row[2] for row in all_pair_rows),
                "same_stream_pair_count": sum(row[0]["streamId"] == row[1]["streamId"] for row in all_pair_rows),
                "same_stream_overlap_ns": sum(
                    row[2] for row in all_pair_rows if row[0]["streamId"] == row[1]["streamId"]
                ),
                "cross_stream_pair_count": sum(row[0]["streamId"] != row[1]["streamId"] for row in all_pair_rows),
                "cross_stream_overlap_ns": sum(
                    row[2] for row in all_pair_rows if row[0]["streamId"] != row[1]["streamId"]
                ),
                "active_overlap_ns": all_total_ns - all_union_ns,
            },
            "invocations": invocations,
        },
        "api_overhead": {
            "all_runtime": api_subset_summary(runtimes),
            "kernel_span_intersection": api_subset_summary(clipped_api),
            "launch_all": api_subset_summary([row for row in runtimes if api_category(row.get("name")) == "launch"]),
            "launch_timed": api_subset_summary(timed_launch_rows),
            "launch_warmup": api_subset_summary([
                runtime_by_correlation.get(kernel.get("correlationId"))
                for kernel in kernels[:timed_start_index * kernels_per_invocation]
                if runtime_by_correlation.get(kernel.get("correlationId")) is not None
            ]),
            "timed_kernel_duration_ns": sum(
                invocation["kernel_duration_sum_ns"] for invocation in timed_invocations
            ),
            "timed_kernel_span_ns": (
                max((invocation["end_ns"] for invocation in timed_invocations), default=0)
                - min((invocation["start_ns"] for invocation in timed_invocations), default=0)
            ),
            "timed_kernel_union_ns": union_length(timed_intervals),
            "timed_global_gap": summarize_gaps(positive_gaps(timed_intervals)),
        },
        "kernel_rows": [
            {
                "row_index": kernel["row_index"],
                "start_ns": kernel["start"],
                "end_ns": kernel["end"],
                "duration_ns": kernel["end"] - kernel["start"],
                "stream_id": kernel["streamId"],
                "correlation_id": kernel["correlationId"],
                "short_name": kernel["short"],
                "demangled_name": kernel["demangled"],
                "stage": parse_kernel_stage(kernel.get("demangled")),
                "grid": [kernel["gridX"], kernel["gridY"], kernel["gridZ"]],
                "block": [kernel["blockX"], kernel["blockY"], kernel["blockZ"]],
                "registers_per_thread": kernel["registersPerThread"],
            }
            for kernel in kernels
        ],
    }
    connection.close()
    return profile_result


def csv_rows(path):
    # NCU prepends human-readable connection lines before the CSV header.
    lines = path.read_text().splitlines()
    header_index = next(
        (index for index, line in enumerate(lines) if line.startswith('"ID","Process ID"')),
        None,
    )
    if header_index is None:
        return
    for row in csv.DictReader(io.StringIO("\n".join(lines[header_index:]))):
        if row.get("Metric Name") and row.get("Kernel Name"):
            yield row


def parse_ncu(path, root, metadata_path=None):
    result = {"source": file_record(path, root, "NCU CSV counters"), "rows": 0, "launch_ids": [], "metrics": {}}
    rows = list(csv_rows(path))
    result["rows"] = len(rows)
    result["launch_ids"] = sorted(set(row.get("ID") for row in rows if row.get("ID") is not None))
    grouped = defaultdict(list)
    for row in rows:
        stage = parse_kernel_stage(row.get("Kernel Name"))
        stage_label = stage["label"] if stage else row.get("Kernel Name")
        try:
            value = float(row.get("Metric Value", ""))
        except ValueError:
            continue
        grouped[(stage_label, row.get("Metric Name"), row.get("Metric Unit"))].append(value)
    result["metrics"] = {
        "%s|%s|%s" % key: {
            "stage": key[0],
            "metric": key[1],
            "unit": key[2],
            "samples": stats(values),
        }
        for key, values in sorted(grouped.items())
    }
    if metadata_path is not None and Path(metadata_path).exists():
        value, source = read_json(Path(metadata_path))
        result["metadata"] = {"source": source, "record": value}
    else:
        result["metadata"] = None
    result["interpretation"] = (
        "Descriptive counters only. NCU counter collection/replay can serialize launches; "
        "these rows cannot establish same-call or cross-stream overlap."
    )
    return result


def extract_normal_variant(control_path, case_id, variant_id, root=ROOT):
    if not control_path.exists():
        return None
    document, _ = read_json(control_path)
    case = next((case for case in document.get("cases", []) if case.get("id") == case_id), None)
    if case is None:
        return None
    variant = next((variant for variant in case.get("variants", []) if variant.get("id") == variant_id), None)
    if variant is None:
        return None
    values = []
    for trial in variant.get("trials", []):
        value = finite(trial.get("kernel_ms"))
        if value is not None and value > 0:
            values.append(value)
    return {
        "source": file_record(control_path, root, "normal controlled timing source"),
        "case_id": case_id,
        "variant_id": variant_id,
        "workload": safe_json(case.get("workload", {})),
        "protocol": safe_json(document.get("protocol", {})),
        "trials_kernel_ms": values,
        "kernel_ms": stats(values),
        "status": variant.get("status"),
        "preflight_status": (variant.get("preflight") or {}).get("status"),
    }


def comparison_row(case, comparison):
    ratio = finite(comparison.get("median_treatment_over_baseline"))
    return {
        "case_id": case.get("id"),
        "workload": safe_json(case.get("workload", {})),
        "baseline": comparison.get("baseline"),
        "treatment": comparison.get("treatment"),
        "speedup_baseline_over_treatment": ratio,
        "actual_changed_axes": safe_json(comparison.get("actual_changed_axes", comparison.get("changed_axes", []))),
        "status": comparison.get("status"),
        "performance_accepted": comparison.get("performance_accepted"),
    }


def summarize_control_document(path, root, platform, study_name):
    document, source = read_json(path)
    comparisons = []
    variant_count = 0
    correct_variant_count = 0
    unsupported_comparisons = 0
    for case in document.get("cases", []):
        variant_count += len(case.get("variants", []))
        for variant in case.get("variants", []):
            trial_statuses = [trial.get("status") for trial in variant.get("trials", [])]
            if variant.get("status") in ("verified", "correct") or (
                trial_statuses and all(status in ("measured", "verified", "correct") for status in trial_statuses)
            ):
                correct_variant_count += 1
        for comparison in case.get("comparisons", []):
            if comparison.get("status") == "measured":
                comparisons.append(comparison_row(case, comparison))
            else:
                unsupported_comparisons += 1
    ratios = [row["speedup_baseline_over_treatment"] for row in comparisons if row["speedup_baseline_over_treatment"] is not None]
    groups = defaultdict(list)
    for row in comparisons:
        key = "+".join(row["actual_changed_axes"])
        groups[key].append(row["speedup_baseline_over_treatment"])
    return {
        "platform": platform,
        "study": study_name,
        "source": source,
        "source_file": file_record(path, root, "controlled timing journal"),
        "schema": document.get("schema"),
        "status": document.get("status"),
        "identity": safe_json(document.get("identity", {})),
        "protocol": safe_json(document.get("protocol", {})),
        "case_count": len(document.get("cases", [])),
        "variant_count": variant_count,
        "correct_variant_count": correct_variant_count,
        "measured_comparison_count": len(comparisons),
        "unsupported_comparison_count": unsupported_comparisons,
        "comparison_summary": ratio_summary(ratios),
        "comparison_groups": {
            key: ratio_summary(values) for key, values in sorted(groups.items())
        },
        "comparisons": comparisons,
        "limitations": safe_json(document.get("limitations", [])),
    }


def summarize_optional_document(path, root, name):
    if not path.exists():
        return {
            "name": name,
            "status": "missing",
            "expected_path": str(path.resolve()),
        }
    document, source = read_json(path)
    comparisons = [
        comparison_row(case, comparison)
        for case in document.get("cases", [])
        for comparison in case.get("comparisons", [])
        if comparison.get("status") == "measured"
    ]
    return {
        "name": name,
        "status": document.get("status", "present"),
        "source": source,
        "source_file": file_record(path, root, "optional paper study"),
        "identity": safe_json(document.get("identity", {})),
        "protocol": safe_json(document.get("protocol", {})),
        "case_count": len(document.get("cases", [])),
        "variant_count": sum(len(case.get("variants", [])) for case in document.get("cases", [])),
        "measured_comparison_count": len(comparisons),
        "comparison_summary": ratio_summary([
            row["speedup_baseline_over_treatment"] for row in comparisons
        ]),
        "comparisons": comparisons,
    }


def aggregate_factorial(document_path, root, name=None):
    result = summarize_optional_document(
        document_path, root, name or "a100-80gb-fft-factorial-v2"
    )
    if result.get("status") == "missing":
        return result
    groups = defaultdict(list)
    for row in result.get("comparisons", []):
        groups["+".join(row["actual_changed_axes"])].append(row["speedup_baseline_over_treatment"])
    result["comparison_groups"] = {
        key: ratio_summary(values) for key, values in sorted(groups.items())
    }
    return result


def aggregate_generation(document_path, root):
    return summarize_optional_document(document_path, root, "a100-80gb-generation-witness")


def summarize_scalar_organization(path, root):
    """Keep the scalar organization control separate from the core factorial."""
    result = summarize_optional_document(path, root, "a100-40gb-scalar-organization")
    if result.get("status") == "missing":
        return result
    document, _ = read_json(path)
    resource_rows = []
    for case in document.get("cases", []):
        variant_rows = {}
        for variant in case.get("variants", []):
            resources = ((variant.get("preflight") or {}).get("resources") or {})
            groups = resources.get("execution_groups") or []
            lane_rows = []
            for group in groups:
                lanes = finite(group.get("data_space"))
                threads = finite(group.get("threads"))
                if lanes is None or threads is None or threads <= 0:
                    continue
                lane_rows.append({
                    "stage_count": group.get("stage_count"),
                    "active_butterfly_lanes": int(lanes),
                    "threads_per_cta": int(threads),
                    "fraction": lanes / threads,
                })
            variant_rows[variant.get("id")] = {
                "stage_partition": safe_json(
                    ((variant.get("preflight") or {}).get("resolved_mapping") or {}).get("stage_partition", [])
                ),
                "active_lane_groups": lane_rows,
                "max_active_butterfly_lanes": max(
                    [row["active_butterfly_lanes"] for row in lane_rows] or [None]
                ),
                "threads_per_cta": max([row["threads_per_cta"] for row in lane_rows] or [None]),
            }
        resource_rows.append({"case_id": case.get("id"), "variants": variant_rows})
    result["resource_control"] = resource_rows
    result["headline_policy"] = (
        "Diagnostic fixed scalar-lowering utilization control. Stagewise uses one active "
        "butterfly lane per 128-thread CTA while resident6 uses 32/128 in its six-stage group; "
        "the large ratios are not a fair stage-per-kernel baseline and cannot be attributed "
        "solely to HBM reduction."
    )
    return result


def source_list_for_report(document, root):
    sources = []
    for item in document.get("provenance", []):
        if item.get("path"):
            sources.append(item)
    return sources


def us(value):
    return float(value) / 1000.0 if value is not None else None


def ms(value):
    return float(value) / 1000000.0 if value is not None else None


def fmt(value, digits=3):
    if value is None:
        return "n/a"
    return ("%%.%df" % digits) % float(value)


def plot_profiles(profiles, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(profiles), 1, figsize=(11, 4.8 * len(profiles)), squeeze=False)
    colors = {"stage0-6": "#1565c0", "stage6-12": "#e65100", "unknown": "#616161"}
    for axis, profile in zip(axes[:, 0], profiles):
        rows = profile["kernel_rows"]
        reference = min(row["start_ns"] for row in rows)
        streams = sorted(set(row["stream_id"] for row in rows))
        y_by_stream = {stream: index for index, stream in enumerate(streams)}
        for row in rows:
            start = us(row["start_ns"] - reference)
            width = us(row["duration_ns"])
            stage = (row.get("stage") or {}).get("label", "unknown")
            axis.barh(
                y_by_stream[row["stream_id"]], width, left=start, height=0.55,
                color=colors.get(stage, colors["unknown"]), edgecolor="black", linewidth=0.35,
            )
        invocations = profile["kernels"]["invocations"]
        for invocation in invocations[1:]:
            axis.axvline(us(invocation["start_ns"] - reference), color="#777777", linestyle="--", linewidth=0.8)
        labels = ["stream %d" % stream for stream in streams]
        axis.set_yticks(list(range(len(streams))))
        axis.set_yticklabels(labels)
        axis.set_ylabel("GPU stream")
        axis.set_xlabel("time from first kernel (us)")
        axis.set_title(
            "%s / %s: %d kernels, %d streams; dashed lines separate invocations"
            % (profile["scope"]["case_id"], profile["scope"]["variant_id"],
               profile["kernels"]["total_count"], profile["kernels"]["stream_count"])
        )
        legend_handles = [
            axis.bar([], [], color=colors["stage0-6"], label="stage 0-6"),
            axis.bar([], [], color=colors["stage6-12"], label="stage 6-12"),
        ]
        axis.legend(handles=legend_handles, loc="upper right", frameon=True)
        axis.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    paths = []
    for suffix in ("svg", "pdf"):
        path = output_dir / ("profile_overlap_timeline." + suffix)
        fig.savefig(path, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths


def plot_controls(control_documents, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    platforms = sorted(set(document["platform"] for document in control_documents))
    fig, axes = plt.subplots(1, len(platforms), figsize=(12, 5), squeeze=False)
    operators = {"fft": "#1565c0", "fwht": "#2e7d32", "ntt": "#e65100"}
    for axis, platform in zip(axes[0], platforms):
        docs = [document for document in control_documents if document["platform"] == platform]
        study_names = [document["study"] for document in docs]
        x_by_study = {name: index for index, name in enumerate(study_names)}
        for document in docs:
            base_x = x_by_study[document["study"]]
            rows = document["comparisons"]
            for offset, row in enumerate(rows):
                workload = row.get("workload", {})
                operator = workload.get("operator", "other")
                ratio = row.get("speedup_baseline_over_treatment")
                if ratio is None:
                    continue
                jitter = ((offset % 5) - 2) * 0.055
                axis.scatter(
                    base_x + jitter, ratio, s=28, alpha=0.78,
                    color=operators.get(operator, "#616161"), edgecolor="white", linewidth=0.35,
                )
        axis.axhline(1.0, color="#333333", linewidth=0.9, linestyle="--")
        axis.set_yscale("log")
        axis.set_xticks(range(len(study_names)))
        axis.set_xticklabels([name.replace("-", "\n") for name in study_names])
        axis.set_ylabel("baseline / treatment speedup")
        axis.set_title("%s controls" % platform)
        axis.grid(axis="y", which="both", alpha=0.22)
    handles = [
        axes[0, 0].scatter([], [], color=color, label=operator)
        for operator, color in operators.items()
    ]
    handles.append(axes[0, 0].axhline(1.0, color="#333333", linestyle="--", label="equal"))
    axes[0, 0].legend(handles=handles, loc="best", frameon=True)
    fig.suptitle("Existing fixed-core controls; ratio > 1 means treatment is faster", y=1.02)
    fig.tight_layout()
    paths = []
    for suffix in ("svg", "pdf"):
        path = output_dir / ("control_comparison_speedups." + suffix)
        fig.savefig(path, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths


def build_report(document, output_path):
    output_dir = output_path.parent
    profiles = document["profiles"]
    controls = document["controls"]
    lines = []
    lines.append("# CPU audit of paper controls and exact profiles")
    lines.append("")
    lines.append(
        "This report is CPU-only. It exports the existing Nsight Systems reports to SQLite and "
        "does not launch a benchmark, query a GPU, or modify the runtime. Normal controlled "
        "timing and profiler timing are separate cohorts."
    )
    lines.append("")
    lines.append("## Scope and protocol")
    lines.append("")
    lines.append(
        "The exact profiler scope is FFT, fp32, N=4096 (logN=12), batch=64, forward, "
        "out-of-place, normalization=none, shared-iterative scalar/radix2, stage partition "
        "[6,6], on the A100 80GB UUID `%s`. Metadata in both reports confirms `warmup=1` and "
        "`repeat=2`; one invocation is the warmup and two are timed repeats."
        % profiles[0]["device"]["uuid"]
    )
    lines.append("")
    lines.append("The application commands reconstructed from `META_DATA_CAPTURE` are:")
    lines.append("")
    for profile in profiles:
        lines.append(
            "- `%s`: session UUID `%s`; `%s`"
            % (
                profile["variant"],
                profile["metadata"].get("profiling_uuid") or "unavailable",
                " ".join(profile["metadata"].get("application_command") or []),
            )
        )
    lines.append("")
    lines.append(
        "The ratio used below is `baseline median kernel_ms / treatment median kernel_ms`; "
        "therefore values above 1 mean that the treatment is faster."
    )
    lines.append("")
    lines.append("## Provenance")
    lines.append("")
    lines.append("Every input and generated SQLite export is listed with its absolute path and SHA-256 in `control_profile_analysis.json`.")
    lines.append("")
    lines.append("| role | path | SHA-256 |")
    lines.append("| --- | --- | --- |")
    seen = set()
    for source in document.get("provenance", []):
        key = source.get("path")
        if not key or key in seen:
            continue
        seen.add(key)
        lines.append("| %s | `%s` | `%s` |" % (source.get("role", "input"), key, source.get("sha256")))
    lines.append("")
    lines.append("## Nsight kernel evidence")
    lines.append("")
    lines.append("| variant | kernels (all/timed) | streams | kernel sum all (us) | timed kernel sum (us) | cross-stream overlap pairs (us) | global gaps all/timed (us) | launch API all/timed (us) |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for profile in profiles:
        kernels = profile["kernels"]
        api = profile["api_overhead"]
        timed_gap = api["timed_kernel_span_ns"] - api["timed_kernel_union_ns"]
        timed_overlap = sum(invocation["overlap"]["cross_stream_overlap_ns"] for invocation in kernels["invocations"][profile["scope"]["warmup_invocations"]:])
        lines.append(
            "| `%s` | %d / %d | %d | %s | %s | %d / %s | %s / %s | %s / %s |"
            % (
                profile["scope"]["variant_id"], kernels["total_count"], kernels["timed_count"],
                kernels["stream_count"], fmt(us(kernels["total_duration_ns"])),
                fmt(us(api["timed_kernel_duration_ns"])), kernels["overlap"]["cross_stream_pair_count"],
                fmt(us(kernels["overlap"]["cross_stream_overlap_ns"])),
                fmt(us(kernels["global_gap"]["total_ns"])), fmt(us(timed_gap)),
                fmt(us(api["launch_all"]["duration_ns"])), fmt(us(api["launch_timed"]["duration_ns"])),
            )
        )
    lines.append("")
    lines.append(
        "Bulk has exactly two kernels per invocation on one stream, and no same-stream or "
        "cross-stream overlap. Pipeline-tile16 has eight kernels per invocation on two streams. "
        "Across its three invocations, SQLite shows %d cross-stream overlap pairs totaling %s us; "
        "the two timed invocations account for %s us. This is direct kernel interval evidence, "
        "not an inference from NCU counters."
        % (
            profiles[1]["kernels"]["overlap"]["cross_stream_pair_count"],
            fmt(us(profiles[1]["kernels"]["overlap"]["cross_stream_overlap_ns"])),
            fmt(us(sum(invocation["overlap"]["cross_stream_overlap_ns"] for invocation in profiles[1]["kernels"]["invocations"][1:]))),
        )
    )
    lines.append("")
    lines.append("Per-invocation overlap and gap details:")
    lines.append("")
    lines.append("| variant/call | kind | kernels | streams | kernel sum (us) | span (us) | cross overlap pairs/us | active global gaps (us) |")
    lines.append("| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |")
    for profile in profiles:
        for invocation in profile["kernels"]["invocations"]:
            lines.append(
                "| `%s/%d` | %s | %d | %s | %s | %s | %d / %s | %s |"
                % (
                    profile["scope"]["variant_id"], invocation["ordinal"], invocation["kind"],
                    invocation["kernel_count"], ",".join(str(x) for x in invocation["stream_ids"]),
                    fmt(us(invocation["kernel_duration_sum_ns"])), fmt(us(invocation["span_ns"])),
                    invocation["overlap"]["cross_stream_pair_count"],
                    fmt(us(invocation["overlap"]["cross_stream_overlap_ns"])),
                    fmt(us(invocation["global_gap"]["total_ns"])),
                )
            )
    lines.append("")
    lines.append(
        "The timed profile kernel sums are diagnostic only. The profile API accounting is also "
        "shown separately because the full trace includes one-time setup: bulk includes a "
        "%s ms cudaMalloc call, while pipeline includes a %s ms cudaStreamCreate call. These "
        "setup costs are not added to normal kernel timing."
        % (
            fmt(ms(max(
                value["duration_ns"]
                for name, value in profiles[0]["api_overhead"]["all_runtime"]["by_name"].items()
                if "cudaMalloc" in name
            )), 3),
            fmt(ms(max(
                value["duration_ns"]
                for name, value in profiles[1]["api_overhead"]["all_runtime"]["by_name"].items()
                if "cudaStreamCreate" in name
            )), 3),
        )
    )
    lines.append("")
    lines.append("## Normal timing kept separate")
    lines.append("")
    lines.append("| variant | normal trials kernel_ms | median kernel_ms | source protocol |")
    lines.append("| --- | --- | ---: | --- |")
    for normal in document["normal_timing"]:
        lines.append(
            "| `%s` | `%s` | %s | warmup=%s, repeat=%s, trials=%s |"
            % (
                normal["variant_id"], ", ".join(fmt(value, 6) for value in normal["trials_kernel_ms"]),
                fmt(normal["kernel_ms"]["median"], 6), normal["protocol"].get("warmup"),
                normal["protocol"].get("repeat"), normal["protocol"].get("trials"),
            )
        )
    normal_ratio = document.get("normal_comparison", {}).get("baseline_over_treatment")
    lines.append("")
    lines.append(
        "For the normal batch64 control, bulk versus pipeline-tile16 is `%s` baseline/treatment "
        "speedup (pipeline is slower). This normal result is not replaced by either Nsight "
        "kernel sum or span."
        % fmt(normal_ratio, 6)
    )
    lines.append("")
    lines.append("## NCU counters")
    lines.append("")
    lines.append("NCU rows are descriptive resource/counter evidence only. NCU collection can serialize launches or replay them, so these counters cannot prove overlap.")
    lines.append("")
    lines.append("| profile | stage | metric | samples | median | unit |")
    lines.append("| --- | --- | --- | ---: | ---: | --- |")
    for ncu in document["ncu"]:
        for metric in ncu.get("metrics", {}).values():
            lines.append(
                "| `%s` | `%s` | `%s` | %d | %s | `%s` |"
                % (
                    ncu["variant"], metric["stage"], metric["metric"], metric["samples"]["count"],
                    fmt(metric["samples"]["median"], 4), metric["unit"],
                )
            )
    lines.append("")
    lines.append("## Existing controls")
    lines.append("")
    lines.append("The following rows are normal controlled timing journals, not profiler latency.")
    lines.append("")
    lines.append("| platform | study | cases | variants | measured comparisons | treatment faster (>1) | treatment slower (<1) | ratio range |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |")
    for control in controls:
        summary = control["comparison_summary"]
        lines.append(
            "| %s | `%s` | %d | %d | %d | %d | %d | %s..%s |"
            % (
                control["platform"], control["study"], control["case_count"], control["variant_count"],
                control["measured_comparison_count"], summary["treatment_faster"],
                summary["treatment_slower"], fmt(summary["min"], 4), fmt(summary["max"], 4),
            )
        )
    lines.append("")
    lines.append(
        "At 80GB, layout is negative for writer-aligned versus linear (all three ratios below 1), "
        "partition is positive for depth6 versus depth3 (all three above 1), and every batch "
        "pipeline comparison is slower than bulk in this journal. At 40GB, the same layout and "
        "partition directions hold; batch has one near-tie positive row (FFT batch16 pipeline-tile16) "
        "and 26 slower rows."
    )
    lines.append("")
    def append_factorial_block(title, factorial, core_caveat=False):
        lines.append("## %s" % title)
        lines.append("")
        if factorial.get("status") == "missing":
            lines.append("The factorial input is missing at `%s`." % factorial.get("expected_path"))
            lines.append("")
            return
        lines.append(
            "This factorial has %d cases, %d variants and %d measured comparisons. Grouped by "
            "the actual changed axes:" % (
                factorial["case_count"], factorial["variant_count"], factorial["measured_comparison_count"]
            )
        )
        lines.append("")
        lines.append("| changed axes | count | treatment faster | treatment slower | ratio range |")
        lines.append("| --- | ---: | ---: | ---: | --- |")
        for key, summary in factorial.get("comparison_groups", {}).items():
            lines.append(
                "| `%s` | %d | %d | %d | %s..%s |"
                % (key, summary["count"], summary["treatment_faster"], summary["treatment_slower"],
                   fmt(summary["min"], 4), fmt(summary["max"], 4))
            )
        if core_caveat:
            lines.append("")
            lines.append(
                "The imported cuFFTDx versus native register-tile rows straddle 1; no global "
                "core ordering is claimed. The simple schedule retains hierarchical factor "
                "decomposition, so this factorial is not a no-mixed-dataflow ablation."
            )
        lines.append("")

    def append_generation_block(title, generation):
        lines.append("## %s" % title)
        lines.append("")
        if generation.get("status") == "missing":
            lines.append("The generation witness input is missing at `%s`." % generation.get("expected_path"))
        else:
            summary = generation["comparison_summary"]
            lines.append(
                "This witness covers FFT/FWHT/NTT at N=16384, batch=3, with %d measured "
                "comparisons. Two-group treatment is faster than three-group baseline in "
                "%d/%d rows (speedup range %s..%s)."
                % (summary["count"], summary["treatment_faster"], summary["count"],
                   fmt(summary["min"], 4), fmt(summary["max"], 4))
            )
        lines.append("")

    append_factorial_block("80GB factorial v2", document["factorial"], core_caveat=True)
    append_generation_block("80GB generation witness", document["generation"])
    append_factorial_block("40GB factorial v2", document["optional_40gb"]["factorial"])
    append_generation_block("40GB generation witness", document["optional_40gb"]["generation"])

    scalar = document.get("scalar_organization", {})
    lines.append("## 40GB scalar organization control")
    lines.append("")
    if scalar.get("status") == "missing":
        lines.append("The scalar organization control is missing at `%s`." % scalar.get("expected_path"))
    else:
        scalar_summary = scalar.get("comparison_summary", {})
        lines.append(
            "This separate ordinary scalar-core control has %d cases, %d variants and %d "
            "measured comparisons; resident6 is faster in %d/%d rows (ratio range %s..%s)."
            % (scalar.get("case_count"), scalar.get("variant_count"), scalar.get("measured_comparison_count"),
               scalar_summary.get("treatment_faster"), scalar_summary.get("count"),
               fmt(scalar_summary.get("min"), 4), fmt(scalar_summary.get("max"), 4))
        )
        lines.append("")
        lines.append("| case | stagewise active lanes | resident6 active lanes | baseline/treatment speedup |")
        lines.append("| --- | --- | --- | ---: |")
        resource_by_case = {row["case_id"]: row["variants"] for row in scalar.get("resource_control", [])}
        for row in scalar.get("comparisons", []):
            case_id = row["case_id"]
            resources = resource_by_case.get(case_id, {})
            stagewise = resources.get("stagewise", {})
            resident6 = resources.get("resident6", {})
            lines.append(
                "| `%s` | %s/128 | %s/128 | %s |"
                % (case_id, stagewise.get("max_active_butterfly_lanes", "n/a"),
                   resident6.get("max_active_butterfly_lanes", "n/a"),
                   fmt(row.get("speedup_baseline_over_treatment"), 4))
            )
        lines.append("")
        lines.append(
            "This is a diagnostic fixed-lowering utilization control, not a fair "
            "stage-per-kernel baseline and not evidence that ordinary scalar code is 3--57x "
            "faster solely because of HBM reduction: stagewise uses 1/128 active butterfly "
            "lane in each CTA, whereas the six-stage resident group uses 32/128."
        )
    lines.append("")
    lines.append("## Missing evidence and limits")
    lines.append("")
    for item in document["missing_evidence"]:
        lines.append("- %s" % item)
    lines.append("")
    lines.append("## Figures")
    lines.append("")
    for figure in document.get("figures", []):
        lines.append("- `%s` (`%s`), SHA-256 `%s`." % (figure["repo_relative"] or figure["path"], figure["format"], figure["sha256"]))
    lines.append("")
    lines.append("## Reproduction")
    lines.append("")
    lines.append("`CUDA_VISIBLE_DEVICES='' python experiments/tools/analyze_control_profiles.py`")
    lines.append("")
    output_path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "results/paper_experiments_20260917/analysis",
    )
    parser.add_argument("--nsys", default="nsys", help="nsys executable used for CPU SQLite export")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    sqlite_dir = output_dir / "nsys_sqlite"
    sqlite_dir.mkdir(parents=True, exist_ok=True)

    base = root / "results/paper_experiments_20260917"
    profile_dir = base / "a100-80gb/profiling"
    record_specs = [
        ("bulk", profile_dir / "fft-batch64_bulk_nsys.json", profile_dir / "fft-batch64_bulk_nsys.nsys-rep"),
        ("pipeline-tile16", profile_dir / "fft-batch64_pipeline-tile16_nsys.json", profile_dir / "fft-batch64_pipeline-tile16_nsys.nsys-rep"),
    ]
    profiles = []
    provenance = []
    for variant, record_path, rep_path in record_specs:
        sqlite_path = sqlite_dir / (rep_path.stem + ".sqlite")
        if not record_path.exists() or not rep_path.exists():
            raise FileNotFoundError("missing exact profile input for %s" % variant)
        command = [
            args.nsys, "export", "--type", "sqlite", "--force-overwrite=true", "--quiet=true",
            "--output", str(sqlite_path), str(rep_path),
        ]
        completed = subprocess.run(command, cwd=str(root), text=True, capture_output=True)
        if completed.returncode:
            raise RuntimeError("nsys export failed for %s: %s" % (variant, completed.stderr.strip()))
        profile = load_sqlite_profile(record_path, rep_path, sqlite_path, root)
        profile["variant"] = variant
        profile["sqlite_export_command"] = command
        profiles.append(profile)
        provenance.extend([
            profile["record"], profile["report"], profile["sqlite_export"],
        ])

    normal_control_path = base / "a100-80gb/controls/batch.json"
    normal_timing = [
        extract_normal_variant(normal_control_path, "fft-batch64", "bulk", root),
        extract_normal_variant(normal_control_path, "fft-batch64", "pipeline-tile16", root),
    ]
    if any(value is None for value in normal_timing):
        raise ValueError("normal batch64 control rows are missing")
    normal_comparison = {
        "baseline": "bulk",
        "treatment": "pipeline-tile16",
        "baseline_over_treatment": normal_timing[0]["kernel_ms"]["median"] / normal_timing[1]["kernel_ms"]["median"],
        "source": normal_timing[0]["source"],
    }

    control_specs = [
        ("a100-80gb", "layout", base / "a100-80gb/controls/layout.json"),
        ("a100-80gb", "partition", base / "a100-80gb/controls/partition.json"),
        ("a100-80gb", "batch", base / "a100-80gb/controls/batch.json"),
        ("a100-40gb", "layout", base / "a100-40gb/controls/journals/layout_audited.json"),
        ("a100-40gb", "partition", base / "a100-40gb/controls/journals/partition_audited.json"),
        ("a100-40gb", "batch", base / "a100-40gb/controls/journals/batch_audited.json"),
    ]
    controls = []
    for platform, study, path in control_specs:
        if not path.exists():
            raise FileNotFoundError("missing control journal: %s" % path)
        document = summarize_control_document(path, root, platform, study)
        controls.append(document)
        provenance.extend([document["source_file"]])
    summary_40 = base / "a100-40gb/controls/controls_summary.json"
    if summary_40.exists():
        summary_value, source = read_json(summary_40)
        summary_record = file_record(summary_40, root, "40GB controls summary cross-check")
        summary_record["declared_hardware"] = safe_json(summary_value.get("hardware", {}))
        summary_record["declared_protocol"] = safe_json(summary_value.get("protocol", {}))
        summary_record["declared_counts"] = safe_json(summary_value.get("counts", {}))
        derived_40 = [control for control in controls if control["platform"] == "a100-40gb"]
        summary_record["derived_counts"] = {
            "cases": sum(control["case_count"] for control in derived_40),
            "variants": sum(control["variant_count"] for control in derived_40),
            "comparisons_measured": sum(control["measured_comparison_count"] for control in derived_40),
        }
        provenance.append(summary_record)
    else:
        summary_record = None

    factorial_path = base / "a100-80gb/fft_factorial_v2.json"
    factorial = aggregate_factorial(factorial_path, root)
    if factorial.get("source_file"):
        provenance.append(factorial["source_file"])
    factorial_summary_path = base / "a100-80gb/fft_factorial_v2_summary.json"
    if factorial_summary_path.exists():
        provenance.append(file_record(factorial_summary_path, root, "factorial derived summary cross-check"))
    generation_path = base / "a100-80gb/generation_witness.json"
    generation = aggregate_generation(generation_path, root)
    if generation.get("source_file"):
        provenance.append(generation["source_file"])

    optional_40_factorial = aggregate_factorial(
        base / "a100-40gb/fft_factorial_v2.json", root, "a100-40gb-fft-factorial-v2"
    )
    optional_40_generation = summarize_optional_document(
        base / "a100-40gb/generation_witness.json", root, "a100-40gb-generation-witness"
    )
    scalar_organization = summarize_scalar_organization(
        base / "a100-40gb/scalar_organization.json", root
    )
    for optional in (optional_40_factorial, optional_40_generation, scalar_organization):
        if optional.get("source_file"):
            provenance.append(optional["source_file"])

    ncu_specs = [
        ("bulk", profile_dir / "fft-batch64_bulk_ncu.csv", profile_dir / "fft-batch64_bulk_ncu.json"),
        ("pipeline-tile16", profile_dir / "fft_batch64_pipeline_tile16.ncu.csv", None),
    ]
    ncu = []
    for variant, csv_path, metadata_path in ncu_specs:
        if not csv_path.exists():
            continue
        value = parse_ncu(csv_path, root, metadata_path)
        value["variant"] = variant
        ncu.append(value)
        provenance.append(value["source"])
        if value.get("metadata"):
            provenance.append(value["metadata"]["source"])

    missing = [
        "No A100 40GB profiler report was supplied; 40GB controls are timing journals only and controls_summary.json records profiling as deferred.",
        "The pipeline NCU CSV has no companion NCU metadata JSON, so its NCU command and warmup/repeat cannot be independently reconstructed from that file.",
        "The two Nsight traces cover only FFT N=4096, batch=64 on one A100 80GB UUID; they do not establish cross-platform or whole-matrix behavior.",
    ]
    if optional_40_factorial.get("status") == "missing":
        missing.append("A100 40GB factorial-v2 JSON is absent in this snapshot.")
    if optional_40_generation.get("status") == "missing":
        missing.append("A100 40GB generation-witness JSON is absent in this snapshot.")
    if scalar_organization.get("status") == "missing":
        missing.append(
            "The prepared scalar-organization HBM-writeback versus CTA-residency control is absent; "
            "when present it must remain a separate ordinary-core control, not an FFT factorial row."
        )

    document = {
        "schema": ANALYSIS_SCHEMA,
        "scope": {
            "cpu_only": True,
            "gpu_experiment_started": False,
            "normal_timing_is_not_replaced": True,
            "nsys_export": "nsys export --type sqlite",
            "profile_case": "fft-batch64",
            "profile_workload": {
                "operator": "fft", "precision": "fp32", "logN": 12, "N": 4096, "batch": 64,
                "direction": "forward", "placement": "out-of-place", "normalization": "none",
            },
        },
        "profiles": profiles,
        "normal_timing": normal_timing,
        "normal_comparison": normal_comparison,
        "ncu": ncu,
        "controls": controls,
        "controls_summary_40gb": summary_record,
        "factorial": factorial,
        "generation": generation,
        "optional_40gb": {
            "factorial": optional_40_factorial,
            "generation": optional_40_generation,
        },
        "scalar_organization": scalar_organization,
        "missing_evidence": missing,
        "provenance": provenance,
        "figures": [],
    }
    # Deduplicate provenance while retaining first role assignment.
    deduped = []
    seen = set()
    for source in document["provenance"]:
        key = source.get("path")
        if key and key not in seen:
            deduped.append(source)
            seen.add(key)
    document["provenance"] = deduped

    if not args.no_plots:
        figure_paths = plot_profiles(profiles, output_dir)
        figure_paths.extend(plot_controls(controls, output_dir))
        document["figures"] = [
            dict(file_record(path, root, "analysis figure"), format=path.suffix.lstrip("."))
            for path in figure_paths
        ]
    output_json = output_dir / "control_profile_analysis.json"
    write_json(output_json, document)
    build_report(document, output_dir / "REPORT.md")
    print(json.dumps({
        "output": str(output_json),
        "report": str(output_dir / "REPORT.md"),
        "profiles": [
            {
                "variant": profile["variant"],
                "kernels": profile["kernels"]["total_count"],
                "streams": profile["kernels"]["stream_count"],
                "cross_stream_overlap_ns": profile["kernels"]["overlap"]["cross_stream_overlap_ns"],
            }
            for profile in profiles
        ],
        "gpu_experiment_started": False,
    }, indent=2))


if __name__ == "__main__":
    main()
