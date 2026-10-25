"""Check new NTT templates through the public inventory and exact mapping replay.

Run inside run_isolated_gpu.py. This is a bounded correctness check, not a
performance study. Each process has a timeout and every output is retained.
"""
import argparse
import csv
import io
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    bench = str(args.bench.resolve())
    report = []
    for bits, modulus, count in [(64, 576460756061519873, 16), (32, 2013265921, 26)]:
        base = [bench, "--logN", "20", "--batch", "2", "--word-bits", str(bits),
                "--modulus", str(modulus)]
        inventory = subprocess.run(base + ["--list-design-points"], check=True,
                                   capture_output=True, text=True, timeout=60)
        rows = [json.loads(line) for line in inventory.stdout.splitlines()]
        specialized = [r for r in rows if r.get("subgraph_mappings") and
                       r["subgraph_mappings"][0]["core"].startswith(
                           ("homogeneous-", "dataflow-radix4", "appt-"))]
        appt = [r for r in rows if r.get("subgraph_mappings") and
                r["subgraph_mappings"][0]["core"] == "appt-online-register-tail"]
        if len(appt) != 1 or appt[0].get("stage_partition") != [7, 7, 6]:
            raise AssertionError(f"word{bits}: APPT register-tail descriptor missing")
        unique = {json.dumps(r, sort_keys=True) for r in specialized}
        if len(specialized) != count or len(unique) != count:
            raise AssertionError(f"word{bits}: expected {count} unique specialized templates, "
                                 f"got {len(specialized)} records/{len(unique)} unique")
        (args.output / f"inventory_word{bits}.json").write_text(
            json.dumps(specialized, indent=2) + "\n")
        for inverse in [False, True]:
            for index, mapping in enumerate(specialized):
                label = f"word{bits}_{'inverse' if inverse else 'forward'}_{index:02d}"
                command = base + (["--inverse"] if inverse else []) + [
                    "--mapping-json", json.dumps(mapping), "--verify", "--warmup", "0",
                    "--repeat", "1", "--csv"]
                row = {"case": label, "word_bits": bits, "inverse": inverse,
                       "requested_mapping": mapping, "status": "failed"}
                try:
                    result = subprocess.run(command, capture_output=True, text=True, timeout=60)
                    (args.output / f"{label}.log").write_text(result.stdout + result.stderr)
                    row["returncode"] = result.returncode
                    if result.returncode == 0:
                        measured = list(csv.DictReader(io.StringIO(result.stdout)))
                        if len(measured) != 1:
                            raise AssertionError("missing unique CSV result")
                        measured = measured[0]
                        row["actual_mapping"] = json.loads(measured["mapping_json"])
                        row["correct"] = measured["correct"]
                        row["verified_batches"] = measured["verified_batches"]
                        row["kernel_ms_diagnostic_only"] = measured["kernel_ms"]
                        if measured["correct"] == "1" and measured["verified_batches"] == "2":
                            row["status"] = "passed"
                except (subprocess.TimeoutExpired, AssertionError, ValueError, KeyError) as error:
                    row["error"] = str(error)
                report.append(row)
                (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
                print(f"{label}: {row['status']}", flush=True)
    failed = [r["case"] for r in report if r["status"] != "passed"]
    print(f"NTT specialized public templates: {len(report) - len(failed)}/{len(report)} passed")
    if failed:
        raise SystemExit("failed: " + ", ".join(failed))


if __name__ == "__main__":
    main()
