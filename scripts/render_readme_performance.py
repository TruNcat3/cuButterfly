#!/usr/bin/env python3
"""Render the README's paired speedups from immutable, CPU-readable evidence.

Run from any directory with the project's Python environment. No live results,
GPU, network, or benchmark execution is needed.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures"
SOURCES = ["experiments/snapshots/current_refresh/snapshot.json",
           "experiments/snapshots/current_refresh/fp16_comparison.json"]
MAIN = "main66"
CYCLIC = "cyclic90"
FP16 = "fp16_public_default9_20261105"
GROUPS = [
    (MAIN, "fft", "fp32", "cufft", "FFT FP32 / cuFFT"),
    (MAIN, "fft", "fp32", "vkfft", "FFT FP32 / VkFFT"),
    (MAIN, "fft", "fp64", "cufft", "FFT FP64 / cuFFT"),
    (MAIN, "fft", "fp64", "vkfft", "FFT FP64 / VkFFT"),
    (MAIN, "fwht", "fp32", "daofht", "FWHT FP32 / DaoFHT\n(supplied-output)"),
    (MAIN, "ntt", "word64", "gpu-ntt", "NTT word64 / GPU-NTT"),
    (CYCLIC, "ntt", "word32", "gpu-ntt", "NTT word32 / GPU-NTT"),
    (CYCLIC, "ntt", "word64", "gpu-ntt", "NTT word64 / GPU-NTT"),
    (FP16, "fft", "fp16", "cufft", "FFT FP16 / cuFFT"),
]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    snapshot, low = [json.loads((ROOT / p).read_text()) for p in SOURCES]
    pools = {
        MAIN: [p for p in snapshot["operators"]["pairs"] if p["cohort"] == MAIN],
        CYCLIC: snapshot["cyclic"]["pairs"], FP16: low["pairs"],
    }
    records, summaries = [], []
    seen = set()
    for cohort, operator, precision, baseline, label in GROUPS:
        candidates = [p for p in pools[cohort]
                      if (p["operator"], p["precision"], p["baseline"])
                      == (operator, precision, baseline)]
        ratios = []
        for p in candidates:
            own, external, stored = p.get("own_us"), p.get("baseline_us"), p.get("speedup")
            valid = all(v is not None and math.isfinite(v) and v > 0
                        for v in (own, external, stored))
            ratio = external / own if valid else None
            if valid and not math.isclose(ratio, stored, rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError(f"Stored ratio mismatch: {p}")
            case_id = p.get("case_id", p.get("case"))
            key = (cohort, operator, precision, baseline, case_id)
            if key in seen:
                raise ValueError(f"Duplicate pair: {key}")
            seen.add(key)
            if valid:
                ratios.append(ratio)
            records.append(dict(cohort=cohort, operator=operator, precision=precision,
                                baseline=baseline, case_id=case_id,
                                output_contract="historical supplied-output" if operator == "fwht" else "see frozen source semantics",
                                shape=json.dumps(p.get("shape", p.get("semantics", {}).get("shape")), separators=(",", ":")),
                                batch=p.get("batch", p.get("B")),
                                inverse=p.get("inverse", p.get("semantics", {}).get("inverse")),
                                own_us=own, baseline_us=external, speedup=ratio,
                                included=valid, status=p.get("status"),
                                reason=p.get("reason", ""), source=p.get("source")))
        values = np.array(ratios)
        summaries.append(dict(cohort=cohort, operator=operator, precision=precision,
                              baseline=baseline, label=label, candidates=len(candidates),
                              paired=len(values), unpaired=len(candidates)-len(values),
                              gm=float(np.exp(np.mean(np.log(values)))),
                              faster=int(sum(values > 1)), equal=int(sum(values == 1)),
                              slower=int(sum(values < 1)), minimum=float(min(values)),
                              q1=float(np.quantile(values, .25)), median=float(np.median(values)),
                              q3=float(np.quantile(values, .75)), maximum=float(max(values))))
    assert [sum(s["paired"] for s in summaries if s["cohort"] == c)
            for c in (MAIN, CYCLIC, FP16)] == [49, 54, 9]
    OUT.mkdir(exist_ok=True)
    csv_path = OUT / "readme_performance.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)

    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 11,
                         "svg.fonttype": "none", "svg.hashsalt": "readme-performance-v1"}):
        fig, axes = plt.subplots(3, 1, figsize=(11.5, 8.8), dpi=100, sharex=True,
                                 gridspec_kw={"height_ratios": [6, 2, 1]})
        fig.subplots_adjust(left=.315, right=.79, top=.82, bottom=.24, hspace=.6)
        fig.text(.035, .951, "Paired performance across operators", size=23, weight="bold")
        fig.text(.035, .914, "NVIDIA A100 80GB PCIe · frozen measured cohorts", size=13, color="#444444")
        fig.text(.315, .865, "Slower than baseline  ←     →  Faster than baseline", size=11)
        fig.text(.815, .865, "GM        Faster / n", size=11, weight="bold")
        colors = ["#0072B2", "#0072B2", "#D55E00"]
        sections = [summaries[:6], summaries[6:8], summaries[8:]]
        titles = ["Main operator cohort · 49 matched pairs",
                  "Cyclic NTT extension · 54 matched pairs / 90 requests",
                  "Independent FP16 DEFAULT · 9 matched pairs · 2026-11-05"]
        for ax, section, color, title in zip(axes, sections, colors, titles):
            ax.set_title(title, loc="left", fontsize=11, weight="bold", pad=10)
            for i, summary in enumerate(section):
                values = [r["speedup"] for r in records if r["included"] and
                          (r["cohort"], r["operator"], r["precision"], r["baseline"])
                          == tuple(summary[k] for k in ("cohort", "operator", "precision", "baseline"))]
                ax.boxplot([values], positions=[i], vert=False, widths=.48, whis=(0, 100),
                           showfliers=False, patch_artist=True,
                           boxprops=dict(facecolor="white", edgecolor=color, linewidth=1.5),
                           whiskerprops=dict(color=color), capprops=dict(color=color),
                           medianprops=dict(color="#222222", linewidth=1.7))
                # Fixed jitter reveals coincident observations without changing ratios.
                jitter = np.random.default_rng(410+i).uniform(-.17, .17, len(values))
                ax.scatter(values, i+jitter, s=23, c=color, edgecolors="white", linewidths=.4, zorder=3)
                ax.text(1.05, i, f'{summary["gm"]:.3f}×    {summary["faster"]:2d}/{summary["paired"]}',
                        transform=ax.get_yaxis_transform(), va="center", fontsize=11, fontfamily="DejaVu Sans Mono")
            ax.set_yticks(range(len(section)), [s["label"] for s in section])
            ax.tick_params(axis="y", length=0, pad=14)
            ax.set_ylim(len(section)-.55, -.55)
            ax.set_xscale("log")
            ax.set_xlim(.65, 3.05)
            ax.axvline(1, color="#333333", linestyle="--", linewidth=1.2)
            ax.xaxis.set_major_locator(FixedLocator([.75, 1, 1.5, 2, 3]))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}×"))
            ax.xaxis.set_minor_locator(NullLocator())
            ax.grid(axis="x", color="#dddddd", linewidth=.7)
            ax.set_axisbelow(True)
            for spine in ("top", "right", "left"):
                ax.spines[spine].set_visible(False)
            ax.spines["bottom"].set_color("#bbbbbb")
        axes[-1].set_xlabel("Speedup = baseline / cuButterfly complete-call time (log scale)", labelpad=10)
        notes = ["Each dot is one matched request. GM: geometric mean; faster counts are nominal, not significance.",
                 "Boxes: workload IQR; whiskers: min–max; these are not confidence intervals. All losses are retained.",
                 "Missing / inapplicable pairs are excluded, never zero. Floating-point results are error-report-only.",
                 "FWHT uses the historical supplied-output contract; later native-return repairs are separate."]
        for i, note in enumerate(notes):
            fig.text(.035, .126-i*.026, note, fontsize=10.5, color="#444444")
        for suffix in ("svg", "png"):
            path = OUT / f"readme_performance.{suffix}"
            fig.savefig(path, metadata={"Date": None} if suffix == "svg" else {})
            if suffix == "svg":
                path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines())+"\n")
        plt.close(fig)

    main_requests = [r for r in snapshot["operators"]["requests"] if r["cohort"] == MAIN]
    missing_groups = []
    for op, precision in sorted({(r["operator"], r["precision"]) for r in main_requests}):
        if not any(s["operator"] == op and s["precision"] == precision for s in summaries[:6]):
            missing_groups.append(dict(operator=op, precision=precision,
                                       requests=sum(r["operator"] == op and r["precision"] == precision for r in main_requests),
                                       reason="No matched external baseline in main66"))
    manifest = dict(schema="readme-paired-performance-v1", hardware="NVIDIA A100 80GB PCIe",
                    snapshot_campaign=snapshot["campaign"],
                    cohort_dates={MAIN: "campaign 2026-10-25; frozen evidence", CYCLIC: "2026-10-29 recovery + 2026-11-02 baseline supplement", FP16: "2026-11-05"},
                    sources={p: digest(ROOT/p) for p in SOURCES},
                    generator={"path": str(Path(__file__).relative_to(ROOT)), "sha256": digest(Path(__file__))},
                    versions={"matplotlib": matplotlib.__version__, "numpy": np.__version__},
                    pixels=[1150, 880], axis={"scale": "log", "limits": [.65, 3.05]},
                    ratio="baseline_us / own_us; verified against stored speedup (rtol=atol=1e-12)",
                    aggregation="One vote per matched request within each cohort/operator/precision/baseline; no pooled GM",
                    interpretation="Nominal faster counts, not statistical significance; float error-report-only",
                    box="IQR; median line; whiskers min/max; points all requests; not confidence intervals",
                    summary=summaries, matched_pairs=112,
                    candidate_pair_records=len(records), unpaired_in_displayed_groups=sum(not r["included"] for r in records),
                    full_main66_pair_status_counts=dict(Counter(p["status"] for p in pools[MAIN])),
                    displayed_unpaired_status_counts=dict(Counter(r["status"] for r in records if not r["included"])),
                    displayed_unpaired_reasons=dict(Counter(r["reason"] or r["status"] for r in records if not r["included"])),
                    main66_requests_without_external_comparison=len(main_requests)-len({r["case_id"] for r in records if r["included"] and r["cohort"]==MAIN}),
                    omitted_main66_operator_precisions=missing_groups,
                    omitted_other_cohorts=dict(Counter(p["cohort"] for p in snapshot["operators"]["pairs"] if p["cohort"] != MAIN)),
                    exclusions=["ntt5 replay and fft_grouped2 remeasurements excluded to preserve main66 denominator",
                                "Cyclic90 is independent; its 36 unmatched requests are not plotted",
                                "BF16 experiments deferred; no imputed comparisons",
                                "Later FWHT native-return repair and A100 40GB FP16 supplement are separate",
                                "No PRO6000 or application performance included"],
                    outputs={str(p.relative_to(ROOT)): digest(p) for p in
                             [csv_path, OUT/"readme_performance.svg", OUT/"readme_performance.png"]})
    (OUT/"readme_performance.json").write_text(json.dumps(manifest, indent=2)+"\n")
    for s in summaries:
        print(f'{s["cohort"]} {s["label"].replace(chr(10), " ")}: GM {s["gm"]:.6f}; {s["faster"]}/{s["paired"]}; range {s["minimum"]:.6f}–{s["maximum"]:.6f}; unpaired {s["unpaired"]}')


if __name__ == "__main__":
    main()
