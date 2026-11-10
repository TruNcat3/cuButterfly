#!/usr/bin/env python3
"""Four bounded same-core, same-tile-capacity RNS stage-organization controls."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import statistics
import subprocess


ROOT = Path(__file__).resolve().parents[2]
GPU_UUID = "GPU-2b759faa-3e58-7106-10aa-ea6e387c3f8e"
RESULTS = ROOT / "results/e2e_rns_workload_20261018"
BINARY = RESULTS / "rns_mapping_bench_stage_mapping"
LIBRARY = RESULTS / "library_stage_mapping/libcuntt.so"
PRIMES = ("1152921504577486849", "576460756061519873")


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def command(argv, env=None):
    result = subprocess.run(argv, cwd=ROOT, env=env, capture_output=True, text=True)
    return dict(argv=argv, returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)


def occupancy():
    result = command(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
                      "--format=csv,noheader,nounits"])
    if result["returncode"]:
        raise RuntimeError("Cannot check GPU occupancy: " + result["stderr"])
    result["target_processes"] = [line for line in result["stdout"].splitlines()
                                  if line.split(",", 1)[0].strip() == GPU_UUID]
    return result


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_csv(path, rows):
    if rows:
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def mapping(log_n, inverse, new):
    n1_log = 9 if new else log_n // 2
    prefix_rows = 512 // (1 << (log_n - n1_log))
    suffix_rows = 512 // (1 << n1_log)
    rows, second_rows = (suffix_rows, prefix_rows) if inverse else (prefix_rows, suffix_rows)
    return dict(n1_log=str(n1_log), rows_per_block=str(rows), threads_per_block="256",
                compute_unit="radix2", cross_twiddle="first", physical_boundary_traversal="0",
                core_family="fused", second_rows_per_block=str(second_rows), second_threads_per_block="256")


def mapping_argument(fields):
    return ",".join(fields[key] for key in (
        "n1_log", "rows_per_block", "threads_per_block", "compute_unit", "cross_twiddle",
        "physical_boundary_traversal", "core_family", "second_rows_per_block",
        "second_threads_per_block")).replace("radix2", "2")


def matrix():
    cases = []
    for log_n in (12, 14):
        for inverse in (0, 1):
            old, new = mapping(log_n, inverse, False), mapping(log_n, inverse, True)
            cases.append(dict(log_n=log_n, batch=2, inverse=inverse, old_mapping=old, new_mapping=new,
                              argv=[str(BINARY), "--log-n", str(log_n), "--batch", "2", "--inverse", str(inverse),
                                    "--repetitions", "30", "--fixed-mapping", mapping_argument(old),
                                    "--compare-mapping", mapping_argument(new)]))
    return cases


def validate(result, case):
    if result["returncode"]:
        raise RuntimeError(f"Diagnostic failed with returncode {result['returncode']}: {result['stderr']}")
    rows = list(csv.DictReader(io.StringIO(result["stdout"])))
    assert len(rows) == 2, "Expected two timing rows per process"
    for index, (row, selected) in enumerate(zip(rows, (case["old_mapping"], case["new_mapping"]))):
        expected = dict(case="compare_mapping" if index else "fixed_mapping",
                        comparison="interleaved_mapping_control", log_n=str(case["log_n"]), batch="2",
                        inverse=str(case["inverse"]), ring="negacyclic",
                        input_order="bit_reversed" if case["inverse"] else "natural",
                        output_order="natural" if case["inverse"] else "bit_reversed",
                        q0=PRIMES[0], q1=PRIMES[1], repetitions="30", samples="10",
                        input_preserved="1", status="exact", **selected)
        assert set(row) == set(expected) | {"median_ms"}, "Unexpected CSV schema"
        assert all(row[key] == value for key, value in expected.items()), (row, expected)
        assert math.isfinite(float(row["median_ms"])) and float(row["median_ms"]) > 0
    differences = sorted(key for key in rows[0] if rows[0][key] != rows[1][key]
                         and key not in ("case", "median_ms"))
    assert differences == ["n1_log", "rows_per_block", "second_rows_per_block"], differences
    protocol, samples = [], []
    for line in result["stderr"].splitlines():
        if line.startswith("interleaved_mapping_control_protocol,"):
            protocol.append(dict(field.split("=", 1) for field in line.split(",")[1:]))
        elif line.startswith("interleaved_mapping_control_sample,"):
            samples.append(dict(field.split("=", 1) for field in line.split(",")[1:]))
    assert len(protocol) == 1 and float(protocol[0]["warmup_ms"]) >= 100
    assert protocol[0]["rounds"] == "10" and protocol[0]["repetitions"] == "30"
    assert int(protocol[0]["warmup_graph_launches_per_mapping"]) > 0
    assert len(samples) == 10 and [int(row["round"]) for row in samples] == list(range(10))
    assert [int(row["first_mapping"]) for row in samples] == [0, 1] * 5
    for index, row in enumerate(rows):
        median = statistics.median(float(sample[f"mapping{index}_ms"]) for sample in samples)
        assert math.isclose(median, float(row["median_ms"]), rel_tol=1e-5), "Round median mismatch"
    return rows, protocol[0], samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Run diagnostics on the authorized GPU only if idle")
    parser.add_argument("--output", type=Path, default=RESULTS / "stage_controls")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "manifest.json"
    if manifest_path.exists():
        raise RuntimeError(f"Existing manifest retained: {manifest_path}; use a fresh output directory")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = GPU_UUID
    env["LD_LIBRARY_PATH"] = str(LIBRARY.parent) + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
    artifacts = {str(path): sha256(path) for path in (BINARY, LIBRARY)}
    manifest = dict(started_utc=now(), status="prepared", gpu_uuid=GPU_UUID,
                    scope="Fixed radix2 organization control: split and matching rows change; each stage retains a 512-word tile, both thread counts stay 256",
                    limitation="Two approximately 60-bit prime domains; not CKKS application NTT parameters, latency, or selection regret",
                    measurement_protocol="Same process: joint warmup >=100ms, 10 alternating-order rounds, 30 repetitions per Graph, per-mapping median",
                    artifacts_sha256=artifacts, driver_sha256=sha256(Path(__file__)),
                    environment={key: env[key] for key in ("CUDA_VISIBLE_DEVICES", "LD_LIBRARY_PATH")},
                    planned_executions=matrix(), executions=[], paired_results=[])
    write_json(manifest_path, manifest)
    if not args.execute:
        print(f"Prepared four pairs; no GPU calls: {manifest_path}")
        return
    rows = []
    try:
        hardware = command(["nvidia-smi", "--query-gpu=index,uuid,name,memory.total", "--format=csv,noheader,nounits"])
        manifest.update(hardware=hardware, before=occupancy(), linkage=command(["ldd", str(BINARY)], env))
        assert hardware["returncode"] == 0 and any(GPU_UUID in line for line in hardware["stdout"].splitlines())
        assert manifest["linkage"]["returncode"] == 0 and str(LIBRARY) in manifest["linkage"]["stdout"]
        for case in manifest["planned_executions"]:
            check = occupancy()
            if check["target_processes"]:
                manifest.update(status="prepared_gpu_busy" if not manifest["executions"] else "stopped_gpu_busy", busy_check=check)
                break
            assert all(sha256(Path(path)) == value for path, value in artifacts.items()), "Artifact changed"
            label = f"log{case['log_n']}_b2_inv{case['inverse']}"
            started = now()
            result = command(case["argv"], env)
            (args.output / (label + ".stdout.csv")).write_text(result["stdout"])
            (args.output / (label + ".stderr.log")).write_text(result["stderr"])
            record = dict(**case, **{key: value for key, value in result.items() if key != "argv"},
                          started_utc=started, completed_utc=now(), before=check)
            manifest["executions"].append(record)
            write_json(manifest_path, manifest)
            parsed, protocol, samples = validate(result, case)
            record.update(protocol=protocol, round_samples=samples)
            rows.extend(parsed)
            old_ms, new_ms = (float(row["median_ms"]) for row in parsed)
            manifest["paired_results"].append(dict(log_n=case["log_n"], batch=2, inverse=case["inverse"],
                old_ms=old_ms, new_ms=new_ms, old_over_new=old_ms / new_ms,
                new_minus_old_us=(new_ms - old_ms) * 1000,
                differing_mapping_fields="n1_log,rows_per_block,second_rows_per_block", exact=True))
            manifest["status"] = "running"
            write_json(manifest_path, manifest)
            print(f"{label}: exact; old/new={old_ms/new_ms:.4f}", flush=True)
        else:
            assert len(manifest["paired_results"]) == 4 and len(rows) == 8
            assert all(sha256(Path(path)) == value for path, value in artifacts.items()), "Artifact changed"
            manifest["status"] = "complete"
    except Exception as error:
        manifest.update(status="failed", error=str(error))
        raise
    finally:
        manifest.update(completed_utc=now(), after=occupancy())
        write_json(manifest_path, manifest)
        write_csv(args.output / "executions.csv", rows)
        write_csv(args.output / "pairs.csv", manifest["paired_results"])
        report = ["# RNS stage organization controls", "", f"Status: {manifest['status']}. GPU: `{GPU_UUID}`.", "",
                  manifest["scope"] + ".", "", manifest["measurement_protocol"] + ".", "",
                  "Old split=logN/2; new split=9. Prefix/suffix rows are 512/2^localLog; inverse reverses "
                  "their physical execution order. Core, both thread counts, prime domains, input/output order, "
                  "cross placement, physical traversal and tile capacity are held fixed. The comparison changes "
                  "stage organization and its matching row counts together, not a single row-count parameter.", "",
                  manifest["limitation"] + ". All negative results are retained. A ratio above one favors the new mapping.", "",
                  "| logN | B | Inverse | Old us | New us | Old/new |", "|---:|---:|---:|---:|---:|---:|"]
        for pair in manifest["paired_results"]:
            report.append(f"| {pair['log_n']} | 2 | {pair['inverse']} | {pair['old_ms']*1000:.4f} | {pair['new_ms']*1000:.4f} | {pair['old_over_new']:.4f} |")
        report += ["", "[manifest.json](manifest.json) records hardware, environment, hashes, raw output, "
                   "return codes, exact parameter checks and round samples. [pairs.csv](pairs.csv) holds "
                   "paired ratios; [executions.csv](executions.csv) holds all eight validated rows.", ""]
        (args.output / "README.md").write_text("\n".join(report))


if __name__ == "__main__":
    main()
