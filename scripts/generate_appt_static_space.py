#!/usr/bin/env python3
"""Generate V100 APPT static-layout and online-writer design points."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path


def parse_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",")]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--word-bits", type=int, choices=(32, 64), required=True)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--fragment-widths", default="8,16,32")
    parser.add_argument("--writer-tiles", default="1,2,4")
    parser.add_argument("--producer-weights", default="7,8,9")
    parser.add_argument("--tail-weights", default="10,12,14")
    parser.add_argument("--writer-weights", default="1,2,3,4")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    points = []
    for fragment, tiles, producer, tail, writer, output_order in itertools.product(
        parse_ints(args.fragment_widths), parse_ints(args.writer_tiles),
        parse_ints(args.producer_weights), parse_ints(args.tail_weights),
        parse_ints(args.writer_weights), ("natural", "appt-static"),
    ):
        if output_order == "appt-static" and (tiles != 1 or writer != 1):
            continue
        role_sum = producer + tail + (writer if output_order == "natural" else 0)
        points.append({
            "stage_partition": [7, 7, 6],
            "physical_core": "appt-online-register-tail",
            "word_bits": args.word_bits,
            "batch": args.batch,
            "input_order": "natural",
            "output_order": output_order,
            "fragment_width": fragment,
            "writer_tiles_per_cta": tiles,
            "role_weights": {
                "producer": producer, "tail": tail, "writer": writer,
            },
            "normalized_role_shares": {
                "producer": producer / role_sum,
                "tail": tail / role_sum,
                "writer": writer / role_sum if output_order == "natural" else 0.0,
            },
            "layout_id": f"appt-ntt-log20-7x7x6-xor-fw{fragment}-v1",
        })
    document = {
        "schema": "cubutterfly-appt-static-space-v1",
        "device_contract": "sm70-v100",
        "shape": {"logN": 20, "batch": args.batch,
                  "word_bits": args.word_bits},
        "points": points,
    }
    rendered = json.dumps(document, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
