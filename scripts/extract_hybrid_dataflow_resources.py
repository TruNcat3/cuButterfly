#!/usr/bin/env python3
import argparse
import csv
import re
import subprocess
from pathlib import Path


KERNEL = re.compile(
    r"hybrid_dataflow_kernelI(?P<word>[jm])Lj(?P<flow>\d+)ELj(?P<stage>\d+)E"
    r"Lj(?P<td>\d+)ELj(?P<ur>\d+)ELj(?P<rb>\d+)ELj(?P<ti>\d+)E")
RESOURCE = re.compile(r"REG:(?P<registers>\d+) STACK:(?P<stack>\d+).*LOCAL:(?P<local>\d+)")


def main():
    parser = argparse.ArgumentParser(description="Extract HybridDataflow cubin resources")
    parser.add_argument("binary", type=Path)
    parser.add_argument("--cuobjdump", default="cuobjdump")
    parser.add_argument("--output", "-o", required=True, type=Path)
    args = parser.parse_args()
    result = subprocess.run([args.cuobjdump, "--dump-resource-usage", str(args.binary)],
                            check=True, capture_output=True, text=True)
    rows = {}
    pending = None
    for line in result.stdout.splitlines():
        match = KERNEL.search(line)
        if match:
            pending = match.groupdict()
            continue
        resource = RESOURCE.search(line)
        if pending is None or not resource:
            continue
        row = {
            "word_bits": 32 if pending["word"] == "j" else 64,
            "flow_tile_log_n": int(pending["flow"]),
            "stage_space": int(pending["stage"]),
            "data_time": int(pending["td"]),
            "role_stages": int(pending["ur"]),
            "target_ctas_per_sm": int(pending["rb"]),
            "token_interleave": int(pending["ti"]),
            "registers_per_thread": int(resource["registers"]),
            "stack_bytes": int(resource["stack"]),
            "local_bytes": int(resource["local"]),
        }
        key = tuple(row[field] for field in (
            "word_bits", "flow_tile_log_n", "stage_space", "data_time",
            "role_stages", "target_ctas_per_sm", "token_interleave"))
        rows[key] = row
        pending = None
    if not rows:
        raise ValueError("no HybridDataflow resource records found")
    fields = list(next(iter(rows.values())))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows[key] for key in sorted(rows))


if __name__ == "__main__":
    main()
