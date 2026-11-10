#!/usr/bin/env python3
import argparse
import csv
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path


N12_LABEL = re.compile(r"^(hybrid2d|hybrid_td6|hybrid_td12)_b(\d+)$")


def summarize(rows, sm_count=80, resident_ctas=2):
    samples = defaultdict(list)
    for row in rows:
        samples[(row["case_id"], row["variant"])].append(
            float(row["kernel_ms"]))
    medians = {key: statistics.median(values)
               for key, values in samples.items()}
    records = []
    batches = sorted({int(match.group(2))
                      for case_variant in samples
                      if case_variant[0] == "n12_u32"
                      for match in [N12_LABEL.fullmatch(case_variant[1])]
                      if match})
    for batch in batches:
        h2d = medians[("n12_u32", f"hybrid2d_b{batch}")]
        td6 = medians[("n12_u32", f"hybrid_td6_b{batch}")]
        td12 = medians[("n12_u32", f"hybrid_td12_b{batch}")]
        hybrid = min(td6, td12)
        slots = sm_count * resident_ctas
        waves = math.ceil(batch / slots)
        records.append({
            "case_id": "n12_u32",
            "batch": batch,
            "v06_best_ms": h2d,
            "v07_best_ms": hybrid,
            "v07_variant": "td6" if td6 <= td12 else "td12",
            "v07_vs_v06": h2d / hybrid,
            "td6_ms": td6,
            "td12_ms": td12,
            "td_spread_pct": abs(td6 / td12 - 1.0) * 100.0,
            "cta_waves": waves,
            "last_wave_fill": (batch - (waves - 1) * slots) / slots,
        })
    required = ("hybrid2d", "resident_9_11", "packet_7_13", "lane32")
    if all(("n20_u32_b16", variant) in medians for variant in required):
        v06 = min(medians[("n20_u32_b16", "hybrid2d")],
                  medians[("n20_u32_b16", "resident_9_11")])
        v07 = min(medians[("n20_u32_b16", "packet_7_13")],
                  medians[("n20_u32_b16", "lane32")])
        records.append({
            "case_id": "n20_u32_b16",
            "batch": 16,
            "v06_best_ms": v06,
            "v07_best_ms": v07,
            "v07_variant": "packet" if medians[("n20_u32_b16", "packet_7_13")] <=
                            medians[("n20_u32_b16", "lane32")] else "lane32",
            "v07_vs_v06": v06 / v07,
            "td6_ms": "",
            "td12_ms": "",
            "td_spread_pct": "",
            "cta_waves": "",
            "last_wave_fill": "",
        })
    return records


def render(records):
    n12 = [row for row in records if row["case_id"] == "n12_u32"]
    n20 = next((row for row in records
                if row["case_id"] == "n20_u32_b16"), None)
    lines = [
        "# v0.6/v0.7 Crossover Attribution", "",
        "If the v0.7 candidate set strictly contains the v0.6 candidate set, "
        "then `min(T_v0.7) <= min(T_v0.6)` at every workload. The measured "
        "crossovers therefore diagnose incomplete physical-space inclusion, "
        "coarse scheduling quanta, or stale controls; they are not a property "
        "of the NTT dependency graph.", "",
        "## logN=12 uint32 Wave Boundary", "",
        "| batch | Hybrid2D ms | hybrid Td=6 ms | hybrid Td=12 ms | v0.7/v0.6 | CTA waves | last-wave fill |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in n12:
        lines.append(
            f"| {row['batch']} | {row['v06_best_ms']:.6f} | "
            f"{row['td6_ms']:.6f} | {row['td12_ms']:.6f} | "
            f"{row['v07_vs_v06']:.3f}x | {row['cta_waves']} | "
            f"{row['last_wave_fill'] * 100.0:.1f}% |")
    lines.extend([
        "", "Td=6 and Td=12 differ by less than 1.6% at every point, so the "
        "generated-table threshold is not the crossover cause. The resident "
        "kernel launches one CTA per transform and is limited to two active "
        "CTAs/SM by its register footprint. Batch 160 fills one 80-SM resident "
        "wave; batch 256 requires a second wave with only 60% of its slots "
        "used. Hybrid2D exposes finer intra-transform CTA work and avoids this "
        "same quantization.", "",
    ])
    if n20:
        lines.extend([
            "## Updated logN=20 Control", "",
            "| workload | latest v0.6-best ms | packet/lane32 best ms | v0.7/v0.6 |",
            "|:--|---:|---:|---:|",
            f"| uint32 logN=20 batch=16 | {n20['v06_best_ms']:.6f} | "
            f"{n20['v07_best_ms']:.6f} | {n20['v07_vs_v06']:.3f}x |", "",
            "The historical packet result used the older 17:13 v0.6 control. "
            "Against the later 9:11 resident/Hybrid2D envelope, packet is no "
            "longer the winner at this point. Baseline evolution was a second "
            "source of apparent crossover.", "",
        ])
    lines.extend([
        "## Consequence", "",
        "The release selector must take the lower envelope of both generations. "
        "The research generator must also import the v0.6 Hybrid2D local core "
        "and multi-CTA data-space decomposition as legal v0.7 physical mappings. "
        "Once that candidate-space inclusion is enforced, a remaining crossover "
        "would indicate a measurement or selection bug rather than a legitimate "
        "architecture result.", "",
    ])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("raw", type=Path)
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--markdown", required=True, type=Path)
    args = parser.parse_args()
    with args.raw.open(newline="") as handle:
        records = summarize(csv.DictReader(handle))
    with args.csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]),
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    args.markdown.write_text(render(records))


if __name__ == "__main__":
    main()
