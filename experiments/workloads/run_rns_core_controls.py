#!/usr/bin/env python3
"""Replay saved radix2/radix4 controls with their matching pre-stage-mapping ABI."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import sqlite3
import subprocess


ROOT = Path(__file__).resolve().parents[2]
GPU_UUID = "GPU-2b759faa-3e58-7106-10aa-ea6e387c3f8e"
BINARY = ROOT / "results/e2e_rns_workload_20261018/rns_mapping_bench_interleaved"
LIBRARY = ROOT / "results/e2e_rns_workload_20261018/library_radix4/libcuntt.so"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def command(argv, env=None):
    completed = subprocess.run(argv, cwd=ROOT, env=env, capture_output=True, text=True)
    return dict(argv=argv, returncode=completed.returncode,
                stdout=completed.stdout, stderr=completed.stderr)


def processes():
    result = command(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
                      "--format=csv,noheader,nounits"])
    if result["returncode"]:
        raise RuntimeError("Cannot check GPU occupancy: " + result["stderr"])
    result["target_processes"] = [line for line in result["stdout"].splitlines()
                                  if line.split(",", 1)[0].strip() == GPU_UUID]
    return result


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def pairs():
    representative = [(14, b, inverse) for b in (2, 6) for inverse in (0, 1)]
    remaining = [(log_n, b, inverse) for log_n in (12, 14, 16)
                 for b in (2, 6, 12) for inverse in (0, 1)
                 if (log_n, b, inverse) not in representative]
    return representative + remaining


def launch_geometry(output):
    source = ROOT / "results/e2e_rns_workload_20261018"
    distribution_path = source / "postchange_trace/workloads.json"
    database = source / "ckks_radix4_trace.sqlite"
    distribution = json.loads(distribution_path.read_text())
    connection = sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    grouped = {}
    for scene in distribution["trace_workloads"]:
        sequence = scene["sequence_per_ciphertext"]
        for backend, graph_id in (("native", scene["native_graph_id"]), ("cubutterfly", scene["cu_graph_id"])):
            launches = {}
            for row in connection.execute("""
                SELECT k.*,n.value AS name FROM CUPTI_ACTIVITY_KIND_KERNEL k
                JOIN StringIds n ON n.id=k.demangledName WHERE (k.graphNodeId >> 32)=?
                ORDER BY k.start
            """, (graph_id,)):
                if ("gpuntt::" in row["name"] if backend == "native" else "cuntt::rns_fused_detail::stage_kernel<" in row["name"]):
                    launches.setdefault(row["correlationId"], []).append(row)
            assert len(launches) == scene["observed_graph_launches"]
            for kernels in launches.values():
                assert len(kernels) == scene["calls_per_graph"] * 2
                for request_index in range(scene["calls_per_graph"]):
                    request = sequence[request_index % len(sequence)]
                    if request["B"] != 2:
                        continue
                    for stage_index in (0, 1):
                        row = kernels[2 * request_index + stage_index]
                        grid = tuple(row[k] for k in ("gridX", "gridY", "gridZ"))
                        block = tuple(row[k] for k in ("blockX", "blockY", "blockZ"))
                        key = (scene["scene"], request["inverse"], backend, stage_index, grid, block,
                               row["registersPerThread"], row["staticSharedMemory"], row["dynamicSharedMemory"], row["name"])
                        if key not in grouped:
                            warps = grid[0] * grid[1] * grid[2] * ((block[0] * block[1] * block[2] + 31) // 32)
                            grouped[key] = dict(scene=scene["scene"], B=2, inverse=request["inverse"],
                                                backend=backend, stage_index=stage_index, grid=grid, block=block,
                                                registers_per_thread=row["registersPerThread"],
                                                static_shared_bytes=row["staticSharedMemory"],
                                                dynamic_shared_bytes=row["dynamicSharedMemory"],
                                                kernel_name=row["name"], total_grid_warps=warps,
                                                mean_grid_warps_per_SM=warps / 108,
                                                grid_capacity_fraction_of_64_warps_per_SM=min(1, warps / (108 * 64)),
                                                observed_stage_events=0)
                        grouped[key]["observed_stage_events"] += 1
    connection.close()
    ncu_paths = sorted(source.glob("ncu_b2_dir*_radix*.csv"))
    ncu_active = []
    for path in ncu_paths:
        lines = path.read_text().splitlines()
        start = next(i for i, line in enumerate(lines) if line.startswith('"ID",'))
        for row in csv.DictReader(lines[start:]):
            if row.get("Metric Name") == "sm__warps_active.avg.pct_of_peak_sustained_active":
                ncu_active.append(dict(source=str(path.relative_to(ROOT)), kernel_id=row["ID"],
                                       grid=row["Grid Size"], block=row["Block Size"],
                                       active_warps_percent=float(row["Metric Value"])))
    payload = dict(
        scope="CPU-only launch-geometry audit, paired by validated request order; capacity bounds are not occupancy measurements",
        assumed_hardware=dict(SMs=108, maximum_resident_warps_per_SM=64),
        fixed_N14_B2_rows2_threads256=dict(grid_blocks=128, total_warps=1024,
                                          mean_warps_per_SM=1024 / 108,
                                          grid_capacity_fraction=1024 / (108 * 64)),
        input_sha256={str(path.relative_to(ROOT)): sha256(path) for path in [distribution_path, database, *ncu_paths]},
        ncu_fixed_B2_active_warps=ncu_active,
        application_B2_stage_geometry=sorted(grouped.values(), key=lambda row: (row["scene"], row["inverse"], row["backend"], row["stage_index"])))
    write_json(output / "launch_geometry.json", payload)
    lines = ["# B2 launch geometry", "",
             "Fixed N14/B2 rows2/threads256 launches 128 blocks: 128*256/32=1024 warps. "
             "Across 108 SMs this is 9.481 warps/SM, or 14.815% of 64 resident warps/SM. "
             "This whole-grid parallelism ceiling can explain low active-warp percentages without attributing "
             "them to register pressure. It does not establish compute or bandwidth saturation. "
             "Existing isolated NCU observations are retained in JSON with their source paths; "
             "they are not measurements of the application kernels below.", "",
             "The table uses the new application trace. Native/cu stages are paired by the validated request "
             "sequence. Dimensions are x*y*z; shared is static+dynamic bytes. "
             "The percentage is only total launched warps/(108*64), capped at 100%, not measured occupancy. "
             "Both backends have small grids. cuButterfly's inverse request launches more warps than native "
             "while remaining slower, so whole-grid parallelism alone does not explain the backend latency gap.", "",
             "| Scene/direction | Backend/pass | Grid | Block | Registers | Shared bytes | Grid warp capacity |",
             "|---|---|---|---|---:|---:|---:|"]
    for row in payload["application_B2_stage_geometry"]:
        lines.append(f"| {row['scene']}/{'I' if row['inverse'] else 'F'} | {row['backend']}/{row['stage_index']} | "
                     f"{'*'.join(map(str,row['grid']))} | {'*'.join(map(str,row['block']))} | "
                     f"{row['registers_per_thread']} | {row['static_shared_bytes']}+{row['dynamic_shared_bytes']} | "
                     f"{100*row['grid_capacity_fraction_of_64_warps_per_SM']:.3f}% |")
    lines += ["", "[launch_geometry.json](launch_geometry.json) retains exact kernel names, observed event counts, "
              "per-stage geometry, capacity calculations and source hashes.", ""]
    (output / "launch_geometry.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/e2e_rns_workload_20261018/core_controls_interleaved")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "manifest.json"
    if manifest_path.exists():
        raise RuntimeError(f"Existing manifest retained: {manifest_path}; choose a fresh output directory")
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = GPU_UUID
    environment["LD_LIBRARY_PATH"] = str(LIBRARY.parent) + (
        ":" + environment["LD_LIBRARY_PATH"] if environment.get("LD_LIBRARY_PATH") else "")
    hardware = command(["nvidia-smi", "--query-gpu=index,uuid,name,memory.total",
                        "--format=csv,noheader,nounits"])
    assert hardware["returncode"] == 0
    assert any(GPU_UUID in line for line in hardware["stdout"].splitlines())
    artifacts = {str(p): sha256(p) for p in (BINARY, LIBRARY)}
    linkage = command(["ldd", str(BINARY)], environment)
    assert linkage["returncode"] == 0 and str(LIBRARY) in linkage["stdout"], linkage
    matrix = []
    for log_n, batch, inverse in pairs():
        mapping = f"{log_n // 2},{2 if batch == 2 else 4},256,2,first,0,fused"
        matrix.append(dict(log_n=log_n, batch=batch, inverse=inverse,
                           argv=[str(BINARY), "--log-n", str(log_n), "--batch", str(batch),
                                 "--inverse", str(inverse), "--repetitions", "30",
                                 "--fixed-mapping", mapping, "--compare-radix", "1"]))
    assert len(matrix) == 18
    manifest = dict(started_utc=now(), status="prepared", gpu_uuid=GPU_UUID,
                    scope="Fixed-mapping mechanism control, two approximately 60-bit prime domains; not CKKS application latency or selection regret",
                    artifacts_sha256=artifacts, hardware=hardware, linkage=linkage,
                    measurement_protocol="Two Graphs in one process, jointly warmed >=100ms, 10 alternating-order rounds, median per core",
                    diagnostic_source_sha256=sha256(ROOT / "apps/cuntt_rns_mapping_bench.cu"),
                    build_argv=["nvcc", "-std=c++17", "-O3", "-arch=sm_80", "-Iinclude",
                                "apps/cuntt_rns_mapping_bench.cu", "-Lbuild-contract-20261017", "-lcuntt",
                                "-o", str(BINARY.relative_to(ROOT))],
                    environment={k: environment[k] for k in ("CUDA_VISIBLE_DEVICES", "LD_LIBRARY_PATH")},
                    before=processes(), planned_executions=matrix, executions=[], paired_results=[])
    write_json(manifest_path, manifest)
    if not args.execute or manifest["before"]["target_processes"]:
        manifest["status"] = "prepared_gpu_busy" if manifest["before"]["target_processes"] else "prepared"
        write_json(manifest_path, manifest)
        print(manifest["status"] + ": " + str(manifest_path), flush=True)
        return
    rows = []
    try:
        for case in matrix:
            assert all(sha256(Path(path)) == value for path, value in artifacts.items()), "Artifact changed"
            occupancy = processes()
            if occupancy["target_processes"]:
                manifest["status"] = "stopped_gpu_busy"
                manifest["busy_check"] = occupancy
                break
            label = f"log{case['log_n']}_b{case['batch']}_inv{case['inverse']}_interleaved"
            started = now()
            result = command(case["argv"], environment)
            (args.output / (label + ".stdout.csv")).write_text(result["stdout"])
            (args.output / (label + ".stderr.log")).write_text(result["stderr"])
            record = dict(**case, **{k: v for k, v in result.items() if k != "argv"},
                          started_utc=started, completed_utc=now(), before=occupancy)
            manifest["executions"].append(record)
            write_json(manifest_path, manifest)
            assert result["returncode"] == 0, f"{label}: returncode {result['returncode']}"
            parsed = list(csv.DictReader(io.StringIO(result["stdout"])))
            assert len(parsed) == 2, f"{label}: expected two timing rows"
            for radix, row in zip((2, 4), parsed):
                expected = dict(log_n=str(case["log_n"]), batch=str(case["batch"]),
                                inverse=str(case["inverse"]), n1_log=str(case["log_n"] // 2),
                                rows_per_block=str(2 if case["batch"] == 2 else 4),
                                threads_per_block="256", compute_unit=f"radix{radix}",
                                cross_twiddle="first", physical_boundary_traversal="0", core_family="fused",
                                repetitions="30", samples="10", input_preserved="1", status="exact",
                                comparison="interleaved_radix_control")
                assert all(row[k] == value for k, value in expected.items()), (label, row, expected)
                assert float(row["median_ms"]) > 0
                rows.append(row)
            protocol_rows = [dict(field.split("=", 1) for field in line.split(",")[1:])
                             for line in result["stderr"].splitlines() if line.startswith("compare_radix_protocol,")]
            sample_rows = [dict(field.split("=", 1) for field in line.split(",")[1:])
                           for line in result["stderr"].splitlines() if line.startswith("compare_radix_sample,")]
            assert len(protocol_rows) == 1 and float(protocol_rows[0]["warmup_ms"]) >= 100
            assert len(sample_rows) == 10 and int(protocol_rows[0]["rounds"]) == 10
            assert [int(row["first_radix"]) for row in sample_rows] == [2, 4] * 5
            record.update(protocol=protocol_rows[0], round_samples=sample_rows)
            a, b = parsed
            differences = sorted(k for k in a if a[k] != b[k] and k != "median_ms")
            assert differences == ["compute_unit"], differences
            manifest["paired_results"].append(dict(
                log_n=case["log_n"], batch=case["batch"], inverse=case["inverse"],
                radix2_ms=float(a["median_ms"]), radix4_ms=float(b["median_ms"]),
                radix2_over_radix4=float(a["median_ms"]) / float(b["median_ms"]),
                radix4_minus_radix2_us=(float(b["median_ms"]) - float(a["median_ms"])) * 1000,
                differing_mapping_fields="compute_unit", exact=True))
            print(f"{label}: exact; radix2/radix4={manifest['paired_results'][-1]['radix2_over_radix4']:.4f}", flush=True)
            manifest["status"] = "running"
            write_json(manifest_path, manifest)
        else:
            assert len(manifest["paired_results"]) == 18
            assert all(sha256(Path(path)) == value for path, value in artifacts.items()), "Artifact changed"
            manifest["status"] = "complete"
    except Exception as error:
        manifest.update(status="failed", error=str(error))
        raise
    finally:
        manifest.update(completed_utc=now(), after=processes())
        write_json(manifest_path, manifest)
        write_csv(args.output / "executions.csv", rows)
        write_csv(args.output / "pairs.csv", manifest["paired_results"])
        lines = ["# Fixed-mapping RNS core controls", "", f"Status: {manifest['status']}. GPU: `{GPU_UUID}`.", "",
                 "Each pair changes only compute_unit (radix2/radix4). Split=logN/2; rows=2 for B2 and 4 otherwise; "
                 "threads=256, cross=first, physical traversal=0, fused core. "
                 "Two Graphs are captured and CPU-exact checked outside timing in one process. Both cores jointly "
                 "warm for at least 100ms, then run 10 rounds alternating which core goes first. Each Graph contains "
                 "30 repetitions; each core reports its median over 10 rounds. Post-measurement CPU-exact output "
                 "and input preservation are required. Warmup duration and every round are retained in stderr and JSON.", "",
                 "This same-process interleaved protocol is the primary fixed-core control. The original "
                 "separate-process controls in ../core_controls/ have a cold-start and GPU-frequency-ramp "
                 "confounding risk and are retained as historical diagnostics; their quantitative ratios are superseded.", "",
                 "These are isolated fixed-mapping controls with two existing approximately 60-bit prime domains. "
                 "They do not replace the CKKS application 40/50-bit measurements or measure selector regret. "
                 "A ratio above one favors radix4; all negative results remain in the table.", "",
                 "| logN | B | Inverse | Radix2 us | Radix4 us | Radix2/radix4 |", "|---:|---:|---:|---:|---:|---:|"]
        for row in manifest["paired_results"]:
            lines.append(f"| {row['log_n']} | {row['batch']} | {row['inverse']} | {row['radix2_ms']*1000:.4f} | {row['radix4_ms']*1000:.4f} | {row['radix2_over_radix4']:.4f} |")
        lines += ["", "[manifest.json](manifest.json) records commands, environment, GPU identity and occupancy, "
                  "artifact SHA256 hashes, raw stdout/stderr, return codes, and timestamps. "
                  "[pairs.csv](pairs.csv) records paired ratios; [executions.csv](executions.csv) retains parsed timing rows. "
                  "[B2 launch geometry](launch_geometry.md) compares native/cu stage dimensions and resource counts.", ""]
        (args.output / "README.md").write_text("\n".join(lines))
        launch_geometry(args.output)
    print(str(manifest_path), flush=True)


if __name__ == "__main__":
    main()
