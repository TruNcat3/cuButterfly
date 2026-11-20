from pathlib import Path
import json, hashlib, math, statistics
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
ROOT=Path(__file__).resolve().parent
def gm(values):
    return math.exp(statistics.mean(math.log(v) for v in values)) if values else None

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def axis_style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#DDDDDD", linewidth=.5)
    ax.set_axisbelow(True)

def save(fig, name, output, manifest, description):
    paths = []
    for ext in ("pdf", "svg", "png"):
        path = output / f"current_refresh_{name}.{ext}"
        fig.savefig(path, dpi=300, facecolor="white", transparent=False)
        paths.append(path)
    manifest["figures"][name] = {
        "description": description,
        "width_inches": float(fig.get_figwidth()),
        "height_inches": float(fig.get_figheight()),
        "files": {str(p.relative_to(output)): digest(p) for p in paths},
    }
    plt.close(fig)

def shown_applications(data):
    return [p for p in data["applications"]["pairs"] if p.get("speedup") is not None]

def strongest_ckks_pairs(data):
    """Check the frozen per-request faster-native baseline before plotting."""
    key = lambda r: (r["scenario"], int(r["N"]), int(r["batch"]), r["method"])
    candidates = {}
    for row in shown_applications(data):
        if row["cohort"] == "ckks_auto36":
            candidates.setdefault(key(row), []).append(row)
    chosen = data["applications"]["strongest_native_pairs"]
    if len(chosen) != 36 or len({key(row) for row in chosen}) != 36:
        raise ValueError("Expected 36 distinct CKKS requests with faster-native references")
    for row in chosen:
        options = candidates[key(row)]
        if {r["baseline"] for r in options} != {"HEonGPU_native_loop", "HEonGPU_native_stage_batch"}:
            raise ValueError("CKKS request is missing a native execution alternative")
        best = min(options, key=lambda r: r["baseline_ms"])
        if not all(math.isclose(r["own_ms"], row["own_ms"], rel_tol=1e-9) for r in options):
            raise ValueError("CKKS alternatives must share the same cuButterfly measurement")
        if row["selected_baseline"] != best["baseline"] or not math.isclose(row["speedup"], best["baseline_ms"] / row["own_ms"], rel_tol=1e-9):
            raise ValueError("Frozen faster-native CKKS ratio does not match its measurements")
    return chosen

def application_plot(data, output, manifest):
    from paper_plot_style import RC
    with plt.rc_context(RC):
        all_pairs = shown_applications(data)
        # Matched batching is an integration intervention, not an upstream API.
        # Keep those records in the snapshot and controlled-experiment table.
        controls = [r for r in all_pairs if r["baseline"] == "HEonGPU_native_stage_batch"]
        pairs = [r for r in all_pairs if r["baseline"] != "HEonGPU_native_stage_batch"]
        historical_path = ROOT / "inputs/application_historical_complete_call.json"
        historical = json.loads(historical_path.read_text())
        if historical.get("status") != "historical-pinned-build" or len(historical.get("records", [])) != 4:
            raise ValueError("the four historical complete-call records are required explicitly")
        if any(type(r.get("performance_eligible")) is not bool for r in historical["records"]):
            raise ValueError("historical records require explicit numerical-evidence eligibility")
        eligible_history = [r for r in historical["records"] if r["performance_eligible"]]
        if [r["workload"] for r in eligible_history] != ["FNO 64² / B8", "FNO 128² / B4"]:
            raise ValueError("historical figure layout requires the two audited FNO records")
        fig, ax = plt.subplots(figsize=(3.45, 3.15))
        fig.subplots_adjust(left=.15, right=.985, bottom=.565, top=.83)
        fft = [r for r in pairs if r["cohort"] in ("poisson_graph", "fft_filter_graph")]
        order = [("poisson_graph", 1024, 4), ("poisson_graph", 2048, 4), ("poisson_graph", 2048, 16),
                 ("fft_filter_graph", 1024, 16), ("fft_filter_graph", 2048, 16)]
        for idx, (cohort, n, batch) in enumerate(order):
            for r in fft:
                edge = r.get("N") or r.get("shape", [None])[0]
                if isinstance(edge, list):
                    edge = edge[0]
                if (r["cohort"], int(edge), int(r["batch"])) != (cohort, n, batch):
                    continue
                cupy = "cupy" in r["baseline"].lower()
                color, marker, offset = ("#007B59", "^", .13) if cupy else ("#A34D00", "s", -.13)
                ax.scatter(idx + offset, r["speedup"], color=color, marker=marker, s=12,
                           facecolors="none" if cohort == "fft_filter_graph" else color, zorder=4)
        ckks = strongest_ckks_pairs(data)
        for i, batch in enumerate((1, 8, 32)):
            vals = [r["speedup"] for r in ckks if int(r["batch"]) == batch]
            if vals:
                center = 5 + i
                offsets = []
                for value in vals:
                    candidates = [0, -.08, .08, -.16, .16, -.21, .21]
                    offset_x = min(candidates, key=lambda candidate: sum(
                        abs(candidate - previous_x) < .075 and abs(value - previous_y) < .04
                        for previous_x, previous_y in offsets))
                    offsets.append((offset_x, value))
                ax.scatter([center + x for x, _ in offsets], vals, color="#A34D00",
                           s=7, marker="o", linewidths=.3, zorder=3)
                ax.scatter(center, gm(vals), s=19, color="#A34D00", edgecolors="black",
                           linewidth=.5, marker="D", zorder=5)
        bs = [r for r in pairs if r["cohort"] == "ckks_bootstrap"]
        for i, batch in enumerate((1, 8)):
            for r in bs:
                if int(r["batch"]) == batch:
                    center = 8 + i
                    ax.scatter(center, r["speedup"], s=14, color="#0072B2", marker="s", zorder=3)
                    ax.annotate(f"{r['speedup']:.3f}", (center, r["speedup"]),
                                xytext=(0, 7 + i * 7), textcoords="offset points",
                                ha="center", fontsize=6.5)
        # Historical complete-call records are a separate pinned-build cohort.
        historical_points = []
        for i, record in enumerate(eligible_history):
            center = 10 + i
            workload = record["workload"]
            color, marker = (("#246a9e", "D") if workload.startswith("FNO")
                             else ("#30856d", "X"))
            value = float(record["speedup"])
            ax.scatter(center, value, s=18, color=color, marker=marker,
                       facecolors="none", linewidths=.75, zorder=4)
            historical_points.append(dict(workload=workload, speedup=value))
        ax.set_xlim(-.5, 11.5)
        ax.set_ylim(.9, 1.32)
        ax.set_xticks(range(12),
                      ["1k²\nB4", "2k²\nB4", "2k²\nB16", "1k²\nB16", "2k²\nB16",
                       "B1", "B8", "B32", "B1", "B8", "64²\nB8", "128²\nB4"], fontsize=6.5)
        ax.set_yticks([.9, 1, 1.1, 1.2, 1.3])
        ax.set_ylabel("Baseline / cuButterfly", fontsize=7)
        ax.set_xlabel("")
        for boundary in (2.5, 4.5, 7.5, 9.5):
            ax.axvline(boundary, color="#BBBBBB", linewidth=.45)
        ax.axvspan(9.5, 11.5, color="#f0f4f6", zorder=0)
        for x, title in ((1, "Poisson"), (3.5, "Filter"), (6, "CKKS"), (8.5, "Bootstrap"),
                         (10.5, "FNO (hist.)")):
            ax.text(x, 1.04, title, transform=ax.get_xaxis_transform(),
                    ha="center", fontsize=6.5)
        axis_style(ax)
        ax.axhline(1, color="#555555", linestyle="--", linewidth=.8)
        ax.tick_params(axis="y", labelsize=6.5, pad=2)
        handles = [Line2D([], [], color=color, marker=marker, linestyle="none",
                          markersize=3, label=label)
                   for color, marker, label in (
                       ("#A34D00", "s", "Torch/cuFFT"), ("#007B59", "^", "CuPy"),
                       ("#A34D00", "o", "Faster native"), ("#0072B2", "s", "Native bootstrap"),
                       ("#246a9e", "D", "FNO historical"),
                       ("#777777", "D", "Padding diagnostic"),
                       ("#AA3377", "D", "QuaRot/Dao"))]
        handles[4].set_markerfacecolor("none")
        handles[5].set_markerfacecolor("none")
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.53, 1.005),
                   ncol=3, frameon=False, fontsize=6.5, handletextpad=.3,
                   columnspacing=.7)
        supplement = data["application_closeout"]
        lower = fig.add_axes([.15, .13, .835, .265])
        fused = data.get("quarot_fused_followup", {}).get("records", [])
        fused_categories = [("QuaRot fused", 1), ("QuaRot fused", 4)] if fused else []
        categories = [("QuaRot", 1), ("QuaRot", 4), *fused_categories, ("filter", 1024), ("filter", 2048),
                      ("bootstrap", 1), ("bootstrap", 8), ("FNO", 64), ("FNO", 128)]
        for i, (kind, key) in enumerate(categories):
            selected = ([dict(row, role="external") for row in fused if row["batch"] == key]
                        if kind == "QuaRot fused" else [])
            for row in supplement["records"]:
                if row["timing_mode"] != "graph":
                    continue
                scenario = row["scenario"]
                matches = ((kind == "QuaRot" and "Llama" in scenario and row["batch"] == key)
                           or (kind == "filter" and f"filter {key}^2" in scenario)
                           or (kind == "bootstrap" and "bootstrap" in scenario and row["batch"] == key)
                           or (kind == "FNO" and row.get("size") == key))
                if matches:
                    selected.append(row)
            # Every process is shown. Offsets separate repeats; they are not uncertainty intervals.
            for j, row in enumerate(selected):
                diagnostic = row["role"] == "diagnostic"
                color, marker = (("#777777", "D") if diagnostic else
                                 ("#AA3377", "D") if kind in ("QuaRot", "QuaRot fused") else
                                 ("#007B59", "^") if kind == "filter" else
                                 ("#0072B2", "s") if kind == "bootstrap" else ("#A34D00", "s"))
                lower.scatter(i + (j - (len(selected) - 1) / 2) * .09, row["speedup"],
                              marker=marker, s=15,
                              facecolors="none" if diagnostic else color, edgecolors=color, linewidths=.7)
        lower.axhline(1, color="#555555", linestyle="--", linewidth=.8)
        tail = len(categories) - .5
        for boundary in ((1.5, 3.5, 5.5, 7.5) if fused else (1.5, 3.5, 5.5)):
            lower.axvline(boundary, color="#BBBBBB", linewidth=.45)
        lower.axvspan(tail - 2, tail, color="#f0f4f6", zorder=0)
        if fused:
            lower.axvspan(1.5, 3.5, color="#edf7f2", zorder=0)
        families = ([(.5, "QuaRot"), (2.5, "Fused"), (4.5, "Filter"), (6.5, "Bootstrap"), (8.5, "FNO")]
                    if fused else [(.5, "QuaRot"), (2.5, "Filter"), (4.5, "Bootstrap"), (6.5, "FNO")])
        for x, label in families:
            lower.text(x, 1.06, label, transform=lower.get_xaxis_transform(), ha="center", fontsize=6.5)
        lower.set_xlim(-.5, tail)
        lower.set_ylim(.93, 1.12)
        lower.set_yticks([.95, 1, 1.05, 1.10])
        lower.set_xticks(range(len(categories)), ["B1", "B4", *(["B1", "B4"] if fused else []),
                                   "1k²\nB16", "2k²\nB16", "B1", "B8", "64²\nB4", "128²\nB4"], fontsize=6.5)
        lower.tick_params(axis="y", labelsize=6.5)
        lower.set_ylabel("Baseline / cuButterfly", fontsize=7)
        axis_style(lower)
        save(fig, "applications", output, manifest,
             "Top: snapshot comparisons, CKKS against the faster recorded native execution per request, "
             "and two numerically checked historical FNO points. "
             "Two earlier QuaRot records are archived but excluded for invalid random-state preparation and absent output checks. "
             "Bottom: all 14 audited closure ratios (four QuaRot losses, four CuPy filter repeats, "
             "four separately marked same-padding filter diagnostics, two bootstrap repeats) and two historical "
             "FNO hotpath Graph aggregate ratios, plus separately marked fused QuaRot follow-up processes when available. "
             "FNO eager rows remain separate in frozen data/table. "
             "Panels use different labeled ratio ranges; no cohorts or builds are pooled.")
        manifest["figures"]["applications"].update(
            renderer_sha256=digest(Path(__file__)),
            pair_count=len(pairs), ckks_pairs=len(ckks), ckks_requests_per_control=36,
            ckks_baseline="minimum measured native-loop/stage-batch latency per matched request",
            ckks_gm=gm([r["speedup"] for r in ckks]),
            ckks_by_batch={str(batch): dict(requests=len([r for r in ckks if int(r["batch"]) == batch]),
                                          gm=gm([r["speedup"] for r in ckks if int(r["batch"]) == batch]))
                           for batch in (1, 8, 32)},
            ckks_selected_pairs=ckks,
            historical_records=len(historical_points), historical_source=str(historical_path.relative_to(ROOT)),
            historical_archived_records=len(historical["records"]),
            historical_exclusions=[dict(workload=r["workload"], reason=r["exclusion_reason"],
                                        source=r["source"], source_sha256=r["source_sha256"])
                                   for r in historical["records"] if not r["performance_eligible"]],
            historical_source_sha256=digest(historical_path),
            matched_integration_pairs_in_controlled_table=len(controls),
            supplemental_graph_rows=16, supplemental_eager_rows_in_table=2,
            fused_quarot_processes=len(fused),
            supplement_source="inputs/applications.json",
            supplement_sources=supplement["sources"],
            statistic="Snapshot CKKS diamonds are GMs of 12 ratios. Supplemental closure dots retain every process; FNO dots are medians of two process medians. No confidence intervals or cross-cohort means.",
            layout="Application-family grouped point charts; separate ratio ranges and shaded pinned-history/fused cohorts. All raw observations retained.",
            font_family="Liberation Serif", font_size_pt=7.0, tick_legend_size_pt=6.5,
            comparator_encoding={"Torch/cuFFT": "orange square", "CuPy": "green triangle",
                                 "HEonGPU faster native": "orange circle", "HEonGPU native bootstrap": "blue square",
                                 "QuaRot/Dao": "magenta diamond (composed and fused)",
                                 "historical FNO": "hollow blue diamond",
                                 "Torch/cuFFT same-padding diagnostic": "hollow grey diamond"})
