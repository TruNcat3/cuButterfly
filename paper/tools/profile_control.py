#!/usr/bin/env python3
"""Profile an already verified controlled mapping, outside normal timing."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from run_comprehensive_suite import require_exclusive_gpu
from run_controlled_experiment import _timed_command, _sha256
from prepare_experiments import write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--tools", nargs="+", choices=["ncu", "nsys"], required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    document = json.loads(args.journal.read_text())
    expected = document["identity"]["device"]["uuid"]
    if os.environ.get("CUDA_VISIBLE_DEVICES", "").lower() != expected.lower():
        raise ValueError("visible GPU must equal the source journal UUID")
    case = next(c for c in document["cases"] if c["id"] == args.case)
    variant = next(v for v in case["variants"] if v["id"] == args.variant)
    if variant.get("preflight", {}).get("status") != "accepted":
        raise ValueError("profiling requires an accepted full-batch preflight")
    binary = Path(case["binary"])
    expected_binary = document["identity"]["binaries"][binary.name]["sha256"]
    if _sha256(binary) != expected_binary:
        raise ValueError("benchmark binary changed since controlled measurement")
    protocol = dict(document["protocol"], warmup=1, repeat=2)
    command = _timed_command(case, variant, protocol)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for tool in args.tools:
        prefix = args.output_dir / f"{args.case}_{args.variant}_{tool}"
        record_path = prefix.with_suffix(".json")
        if record_path.exists():
            raise ValueError(f"profiling record exists: {record_path}; select a new output directory")
        if tool == "ncu":
            invocation = [tool, "--target-processes", "all", "--kernel-name-base", "demangled",
                          "--kernel-name", "regex:shared_iterative_kernel", "--launch-count", "4",
                          "--metrics", "sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed,launch__registers_per_thread",
                          "--csv", "--log-file", str(prefix.with_suffix(".csv"))] + command
        else:
            invocation = [tool, "profile", "--trace=cuda,nvtx", "--sample=none", "--cpuctxsw=none",
                          "--output", str(prefix)] + command
        require_exclusive_gpu()
        result = subprocess.run(invocation, text=True, capture_output=True)
        require_exclusive_gpu()
        write(record_path, dict(schema="cubutterfly-controlled-profile-v1", source=str(args.journal.resolve()),
              source_sha256=_sha256(args.journal), identity=document["identity"], case_id=args.case,
              variant_id=args.variant, mapping=variant["preflight"]["resolved_mapping"], protocol=protocol,
              command=invocation, returncode=result.returncode, stdout=result.stdout, stderr=result.stderr,
              normal_timing_accepted=False))
        print(tool, "exit", result.returncode, "record", record_path, flush=True)
        if result.returncode:
            raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
