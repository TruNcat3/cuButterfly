#!/usr/bin/env python3
"""Render the complete nine-case fixed-core diagnostic from a frozen CPU snapshot.

Default: python paper/tools/render_factor_organization.py
Explicit refresh: add --source-dir results/factor_graph_20261025/core_organization_waitfix_gpu1/release
Refresh verifies the prepared artifacts; it never runs GPU work or compilation.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT = ROOT / "paper/evidence/factor_organization_20261025.json"
STEM = "factor_organization_20261025"
ORGANIZATIONS = ("t1-d0", "t4-d0", "t4-d1", "t4-d2")
IDS = tuple(f"{core}-{org}" for core in ("native", "cufftdx") for org in ORGANIZATIONS) + ("cufft",)
SCOPE = ("Explicit matched-organization mechanism diagnostic; not AUTO selection or an external "
         "provider's native organization. NativeFactor is the auditable repository register-tile "
         "implementation, not a VkFFT core transplant. cuFFT is a complete-library comparator.")


def digest_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def document(path):
    path = resolve(path)
    raw = path.read_text()
    return {"path": str(path.relative_to(ROOT)), "sha256": digest_bytes(raw.encode()), "raw_json": raw}


def refresh(source_dir):
    source_dir = resolve(source_dir)
    status_path = source_dir.parent / "status.json"
    status = json.loads(status_path.read_text())
    if status["state"] != "complete" or Path(status["release"]).resolve() != source_dir.resolve():
        raise ValueError("source status does not identify this completed release")
    command = status["command"]
    prepared = resolve(command[command.index("--prepared") + 1])
    binary = resolve(command[command.index("--binary") + 1])
    preparation = json.loads((source_dir / "preparation.json").read_text())
    suite = json.loads(prepared.read_text())
    config = ROOT / "configs/experiments/factor_graph_core_organization_20261025.json"
    paths = {"config": config, "prepared": prepared, "binary": binary,
             "library": binary.parent / "libcuntt.so", "hardware": Path(preparation["hardware_model"]),
             "reference": Path(suite["reference_file"])}
    if not preparation["complete"]:
        raise ValueError("incomplete preparation")
    checks = []
    for name, path in paths.items():
        actual = sha(path)
        if actual != preparation[name + "_sha256"]:
            raise ValueError(f"prepared artifact hash differs: {name}: {path}")
        checks.append({"kind": name, "path": str(path), "sha256": actual})
    for module in preparation["modules"]:
        if "module" in module:
            if sha(module["module"]) != module["sha256"]:
                raise ValueError(f"module hash differs: {module['id']}")
            checks.append({"kind": "module", "id": module["id"], "path": module["module"],
                           "sha256": module["sha256"]})
    sys.path.insert(0, str(ROOT / "scripts"))
    from run_factor_graph_experiment import check_control_groups
    results = json.loads((source_dir / "results.json").read_text())
    controls = json.loads((source_dir / "control_groups.json").read_text())
    if check_control_groups(suite, results["results"]) != controls:
        raise ValueError("observed control report does not reproduce")
    if check_control_groups(suite) != preparation["control_groups"]:
        raise ValueError("prepared control report does not reproduce")
    sources = {name: document(path) for name, path in {
        "results": source_dir / "results.json", "preparation": source_dir / "preparation.json",
        "control_groups": source_dir / "control_groups.json", "status": status_path,
        "prepared_suite": prepared, "prepared_manifest": prepared.with_suffix(".manifest.json"),
        "config": config, "hardware": paths["hardware"]}.items()}
    if sources["preparation"]["sha256"] != sources["prepared_manifest"]["sha256"]:
        raise ValueError("release preparation differs from original manifest")
    code_hashes = [{"path": path, "sha256": sha(ROOT / path)} for path in
                   ("scripts/run_factor_graph_experiment.py", "src/fft_factor_native.cuh",
                    "src/fft_factor_streamed.cuh")]
    snapshot = {"schema": "factor-organization-frozen-v1", "scope": SCOPE,
                "sources": sources, "verified_preparation_artifacts": checks,
                "source_code_at_freeze": code_hashes,
                "control_validation": {"checker": "scripts.run_factor_graph_experiment.check_control_groups",
                                       "all_groups_verified": True, "report": controls}}
    validate(snapshot)
    SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
    SNAPSHOT.write_text(json.dumps(snapshot, indent=2) + "\n")
    return snapshot


def validate(snapshot):
    sources = snapshot["sources"]
    for name, source in sources.items():
        if digest_bytes(source["raw_json"].encode()) != source["sha256"]:
            raise ValueError(f"frozen source hash mismatch: {name}")
    preparation = json.loads(sources["preparation"]["raw_json"])
    artifacts = snapshot["verified_preparation_artifacts"]
    for name in ("config", "prepared", "binary", "library", "hardware", "reference"):
        matches = [item for item in artifacts if item["kind"] == name]
        if len(matches) != 1 or matches[0]["sha256"] != preparation[name + "_sha256"]:
            raise ValueError(f"frozen preparation provenance mismatch: {name}")
    for name, kind in (("config", "config"), ("prepared_suite", "prepared"), ("hardware", "hardware")):
        if sources[name]["sha256"] != preparation[kind + "_sha256"]:
            raise ValueError(f"frozen document differs from preparation: {name}")
    modules = {m["id"]: m["sha256"] for m in preparation["modules"] if "module" in m}
    verified_modules = {m["id"]: m["sha256"] for m in artifacts if m["kind"] == "module"}
    if len(modules) != 8 or modules != verified_modules:
        raise ValueError("frozen module verification coverage differs")
    if (sources["preparation"]["sha256"] != sources["prepared_manifest"]["sha256"] or
            snapshot["control_validation"]["report"] != json.loads(sources["control_groups"]["raw_json"])):
        raise ValueError("frozen preparation/control reports differ")
    data = json.loads(sources["results"]["raw_json"])
    config = data["config"]
    if (config["operator"], config["precision"], config["logN"], config["batch"],
            config["inverse"], config["trials"], config["repeat"], config["warmup"]) != (
            "fft", "fp64", 16, 16, False, 7, 30, 5):
        raise ValueError("unexpected workload or timing protocol")
    rows = {row["id"]: row for row in data["results"]}
    if len(data["results"]) != 9 or set(rows) != set(IDS) or not data["all_outputs_finite"]:
        raise ValueError("nine-case finite-output coverage is incomplete")
    if not snapshot["control_validation"]["all_groups_verified"]:
        raise ValueError("control validation is incomplete")
    for row in rows.values():
        trials = row["trial_ms"]
        if len(trials) != 7 or any(not 0 < x < float("inf") for x in trials):
            raise ValueError("expected seven finite positive timing trials per case")
        if (statistics.median(trials), min(trials), max(trials)) != (
                row["median_ms"], row["min_ms"], row["max_ms"]):
            raise ValueError("reported timing summary differs from trials")
        if row["nonfinite"] or row["verified_batches"] != 16:
            raise ValueError("CPU verification coverage is incomplete")
    return data, rows


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def render(snapshot):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    data, rows = validate(snapshot)
    generated = ROOT / "paper/generated"
    figures = ROOT / "paper/figures"
    generated.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    table = []
    for case_id in IDS:
        row = rows[case_id]
        entry = {"case_id": case_id, "median_us": row["median_ms"] * 1000,
                 "min_us": row["min_ms"] * 1000, "max_us": row["max_ms"] * 1000,
                 "max_abs_cpu": row["max_abs_cpu"], "relative_l2_cpu": row["relative_l2_cpu"]}
        entry.update({f"trial_{i + 1}_us": x * 1000 for i, x in enumerate(row["trial_ms"])})
        table.append(entry)
    write_csv(generated / f"{STEM}_latency.csv", table)
    pairs = []
    for core in ("native", "cufftdx"):
        for reference in ("t1-d0", "t4-d0"):
            baseline = rows[f"{core}-{reference}"]
            for org in ORGANIZATIONS:
                target = rows[f"{core}-{org}"]
                pairs.append({"core": core, "reference": baseline["id"], "target": target["id"],
                              "reference_median_us": baseline["median_ms"] * 1000,
                              "target_median_us": target["median_ms"] * 1000,
                              "speedup_ratio_of_medians": baseline["median_ms"] / target["median_ms"],
                              "latency_reduction_percent": 100 * (1 - target["median_ms"] / baseline["median_ms"])})
    write_csv(generated / f"{STEM}_paired.csv", pairs)
    style = {"font.family": "DejaVu Sans", "font.size": 7, "axes.labelsize": 7,
             "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.5,
             "pdf.fonttype": 42, "ps.fonttype": 42, "axes.spines.top": False,
             "axes.spines.right": False, "axes.linewidth": .6, "savefig.facecolor": "white"}
    with plt.rc_context(style):
        fig, ax = plt.subplots(figsize=(3.4, 2.3))
        fig.subplots_adjust(left=.15, right=.98, bottom=.21, top=.72)
        for core, label, color, marker, offset in (
                ("native", "NativeFactor (register-tile)", "#0072B2", "o", -.045),
                ("cufftdx", "cuFFTDx block", "#A64B00", "s", .045)):
            chosen = [rows[f"{core}-{org}"] for org in ORGANIZATIONS]
            medians = [r["median_ms"] * 1000 for r in chosen]
            low = [(r["median_ms"] - r["min_ms"]) * 1000 for r in chosen]
            high = [(r["max_ms"] - r["median_ms"]) * 1000 for r in chosen]
            ax.errorbar([i + offset for i in range(4)], medians, yerr=[low, high],
                        color=color, marker=marker, markersize=3.5, linewidth=.9,
                        elinewidth=.8, capsize=2, label=label)
        comparator = rows["cufft"]
        ax.axhspan(comparator["min_ms"] * 1000, comparator["max_ms"] * 1000,
                   color=".7", alpha=.5, linewidth=0)
        ax.axhline(comparator["median_ms"] * 1000, color=".25", linestyle="--",
                   linewidth=.9, label="cuFFT (complete library)")
        ax.set(xticks=range(4), xticklabels=[x.replace("-", "") for x in ORGANIZATIONS],
               xlim=(-.3, 3.3), ylim=(0, 155), yticks=(0, 50, 100, 150),
               xlabel="Organization (tiles / CTA, prefetch depth)", ylabel="Graph latency (µs)")
        ax.grid(axis="y", color=".9", linewidth=.5)
        ax.set_axisbelow(True)
        fig.suptitle("Fixed-core organization diagnostic", fontsize=8, y=.98)
        fig.text(.5, .865, "N = 65,536, B = 16, FP64; A100 80GB", ha="center", fontsize=6.5)
        handles, labels = ax.get_legend_handles_labels()
        order = [labels.index(label) for label in
                 ("NativeFactor (register-tile)", "cuFFTDx block", "cuFFT (complete library)")]
        fig.legend([handles[i] for i in order], [labels[i] for i in order],
                   loc="upper center", bbox_to_anchor=(.56, .85),
                   frameon=False, ncol=1, handlelength=1.8, labelspacing=.22, borderpad=0)
        fig.text(.5, .025, "Median; whiskers / gray band: min-max of 7 timing trials", ha="center", fontsize=6)
        for suffix in ("pdf", "png"):
            metadata = {"Creator": "render_factor_organization.py"} if suffix == "pdf" else {}
            if suffix == "pdf":
                metadata.update({"CreationDate": None, "ModDate": None})
            fig.savefig(figures / f"{STEM}.{suffix}", dpi=600, metadata=metadata)
        plt.close(fig)
    caption = ("Fixed-core matched-organization diagnostic for forward, natural-order, unnormalized "
               "FP64 FFT (N=65536, batch=16) on NVIDIA A100 80GB PCIe. Each point is the median "
               "of seven timing trials of the complete CUDA Graph; whiskers show trial min-max, "
               "not confidence intervals or independent replicates. The cuFFT complete-library "
               "median is dashed with its trial min-max shaded. tXdY denotes X data tiles per CTA "
               "and prefetch depth Y. Both factor cores keep partition [8,8], EPT=16, columns=4, "
               "and all other mapping fields fixed within core. " + SCOPE)
    metadata = {"schema": "factor-organization-render-v1", "caption": caption, "scope": SCOPE,
                "snapshot": str(SNAPSHOT.relative_to(ROOT)), "snapshot_sha256": sha(SNAPSHOT),
                "sources": {k: {"path": v["path"], "sha256": v["sha256"]}
                            for k, v in snapshot["sources"].items()},
                "figure_inches": [3.4, 2.3], "png_dpi": 600, "matplotlib": matplotlib.__version__,
                "python": sys.version.split()[0], "estimator": "median(trial_ms) * 1000",
                "spread": "trial min-max; not a CI; trials are not independent replicates",
                "paired_formula": "reference median / target median; within the same core and cohort",
                "exclusions": [], "all_cases": list(IDS),
                "outputs": {str(p.relative_to(ROOT)): sha(p) for p in
                            [figures / f"{STEM}.pdf", figures / f"{STEM}.png",
                             generated / f"{STEM}_latency.csv", generated / f"{STEM}_paired.csv"]}}
    (generated / f"{STEM}_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Rendered {len(rows)} cases / 63 timing trials from {SNAPSHOT.relative_to(ROOT)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, help="explicitly verify live CPU artifacts and refresh frozen evidence")
    args = parser.parse_args()
    snapshot = refresh(args.source_dir) if args.source_dir else json.loads(SNAPSHOT.read_text())
    render(snapshot)


if __name__ == "__main__":
    main()
