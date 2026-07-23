#!/usr/bin/env python3
import argparse
import csv
import json
import math
import pathlib
import sys


def limiting_rate(capabilities, us, ud, word_bytes, barrier_model):
    spatial_cells = us * ud
    rates = {
        "compute": capabilities["equivalent_butterflies_per_second"] / spatial_cells,
        "boundary": capabilities["global_feedback_bytes_per_second"] / (4 * ud * word_bytes),
    }
    if us > 1:
        rates["interstage"] = capabilities["interstage_shared_bytes_per_second"] / (4 * ud * (us - 1) * word_bytes)
        if barrier_model == "cta":
            rates["synchronization"] = capabilities["cta_barriers_per_second"] / (us - 1)
    bottleneck = min(rates, key=rates.get)
    return rates[bottleneck], bottleneck, rates


def main():
    parser = argparse.ArgumentParser(description="Rank fixed-compute NTT unfolding points using calibrated GPU capabilities.")
    parser.add_argument("--capabilities", required=True, type=pathlib.Path)
    parser.add_argument("--logN", required=True, type=int)
    parser.add_argument("--spatial-budget", required=True, type=int, dest="budget")
    parser.add_argument("--word-bytes", required=True, type=int, dest="word_bytes")
    parser.add_argument("--stage-space", nargs="*", type=int, dest="stage_spaces")
    parser.add_argument("--barrier-model", choices=("none", "cta"), default="cta")
    parser.add_argument("--output", "-o", type=pathlib.Path)
    args = parser.parse_args()

    if args.logN <= 0 or args.budget <= 0 or args.word_bytes <= 0:
        parser.error("logN, spatial budget, and word bytes must be positive")

    profile = json.loads(args.capabilities.read_text())
    capabilities = profile["capabilities"]
    stage_spaces = args.stage_spaces or [factor for factor in range(1, args.logN + 1) if args.budget % factor == 0]
    records = []
    for us in stage_spaces:
        if us <= 0 or args.budget % us != 0:
            continue
        ud = args.budget // us
        ts = math.ceil(args.logN / us)
        td = math.ceil(((1 << args.logN) // 2) / ud)
        body_steps = ts * td
        rate, bottleneck, rates = limiting_rate(capabilities, us, ud, args.word_bytes, args.barrier_model)
        records.append(
            {
                "device": profile["device"],
                "logN": args.logN,
                "spatial_budget_C": args.budget,
                "stage_space_Us": us,
                "data_space_Ud": ud,
                "stage_time_Ts": ts,
                "data_time_Td": td,
                "stage_utilization": args.logN / (us * ts),
                "ideal_body_steps": body_steps,
                "compute_step_rate": rates["compute"],
                "boundary_step_rate": rates["boundary"],
                "interstage_step_rate": rates.get("interstage", ""),
                "synchronization_step_rate": rates.get("synchronization", ""),
                "limiting_capability": bottleneck,
                "calibrated_body_us": body_steps / rate * 1.0e6,
                "barrier_model": args.barrier_model,
            }
        )
    records.sort(key=lambda record: record["calibrated_body_us"])

    output = args.output.open("w", newline="") if args.output else sys.stdout
    try:
        writer = csv.DictWriter(output, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    finally:
        if args.output:
            output.close()


if __name__ == "__main__":
    main()
