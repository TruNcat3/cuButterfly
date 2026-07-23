#!/usr/bin/env python3
import argparse
import csv
import math
import sys


def main():
    parser = argparse.ArgumentParser(description="Analyze the four-way stage/data space-time unfolding of an NTT.")
    parser.add_argument("--logN", type=int, required=True)
    parser.add_argument("--stage-space", type=int, required=True, dest="stage_space")
    parser.add_argument("--data-space", type=int, required=True, dest="data_space")
    parser.add_argument("--word-bytes", type=int, required=True, dest="word_bytes")
    parser.add_argument("--csv", action="store_true")
    args = parser.parse_args()

    if args.logN <= 0 or args.stage_space <= 0 or args.data_space <= 0 or args.word_bytes <= 0:
        parser.error("all numeric arguments must be positive")

    n = 1 << args.logN
    butterflies_per_stage = n // 2
    ts = math.ceil(args.logN / args.stage_space)
    td = math.ceil(butterflies_per_stage / args.data_space)
    eta_stage = args.logN / (args.stage_space * ts)
    eta_data = butterflies_per_stage / (args.data_space * td)
    state_level_bytes = 2 * n * args.word_bytes * ts

    record = {
        "logN": args.logN,
        "N": n,
        "stage_space_Us": args.stage_space,
        "stage_time_Ts": ts,
        "data_space_Ud": args.data_space,
        "data_time_Td": td,
        "stage_utilization": eta_stage,
        "data_utilization": eta_data,
        "ideal_body_cycles": ts * td,
        "spatial_butterfly_cells": args.stage_space * args.data_space,
        "boundary_coefficient_words_per_cycle": 2 * args.data_space,
        "boundary_read_write_bytes_per_cycle": 4 * args.data_space * args.word_bytes,
        "interstage_coefficient_words_per_cycle": 2 * args.data_space * (args.stage_space - 1),
        "twiddle_words_per_cycle_upper_bound": args.stage_space * args.data_space,
        "feedback_coefficient_words_per_cycle": 2 * args.data_space,
        "resident_state_bytes": n * args.word_bytes,
        "state_level_bytes_all_folds": state_level_bytes,
        "butterflies_per_state_level_byte": (args.logN * butterflies_per_stage) / state_level_bytes,
    }

    if args.csv:
        writer = csv.DictWriter(sys.stdout, fieldnames=list(record))
        writer.writeheader()
        writer.writerow(record)
        return

    for name, value in record.items():
        print(f"{name}: {value}")


if __name__ == "__main__":
    main()
