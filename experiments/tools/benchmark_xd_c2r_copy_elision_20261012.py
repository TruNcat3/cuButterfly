#!/usr/bin/env python3
"""C2R copy-elision validation runner.

The default action is CPU-only manifest preparation and contract checking.  A
GPU run is opt-in and executes old/new shared libraries in separate processes,
using the frozen inverse mapping as an explicit public Plan policy.
"""
from __future__ import annotations

import argparse
import hashlib
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
FROZEN = ROOT / "results/paper_targeted_20261012/xd_application_breakdown_v1/default_guard/release/result.json"
OUT = ROOT / "results/paper_targeted_20261012/c2r_copy_elision_20261012"
CASES = [((1024, 1024), 1), ((1024, 1024), 4), ((2048, 2048), 1),
         ((2048, 2048), 4), ((64, 64, 64), 1), ((64, 64, 64), 4),
         ((128, 128, 128), 1), ((128, 128, 128), 4)]


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def prepare(out: Path) -> Path:
    data = json.loads(FROZEN.read_text())
    # The report's ``manifest_sha256`` identifies the upstream input manifest,
    # not the report file itself; bind this runner to the report's own digest.
    by_case = {(tuple(row["shape"]), row["batch"]): row for row in data["direct"]}
    rows = []
    for shape, batch in CASES:
        row = by_case[(shape, batch)]
        inverse = next(p for p in row["plans"] if p["direction"] == "inverse")
        contract = inverse["mapping"]["contract"]
        if contract["fft_type"] != 2 or contract["inverse"] != 1:
            raise RuntimeError("frozen row is not C2R")
        rows.append({"shape": list(shape), "batch": batch,
                     "input_sha256": row["input_sha256"],
                     "mapping": inverse["mapping"],
                     "descriptor": inverse["descriptor"],
                     "algorithm": inverse["algorithm"],
                     "source": {"path": str(FROZEN.relative_to(ROOT)),
                                "field": f"/direct/{CASES.index((shape,batch))}/plans/inverse",
                                "sha256": digest(FROZEN)}})
    manifest = {"schema": "xd-c2r-copy-elision-20261012-v1",
                "frozen_report": str(FROZEN.relative_to(ROOT)),
                "frozen_report_sha256": digest(FROZEN),
                "library_separation": "one subprocess per library; never import old and new in one process",
                "cases": rows,
                "correctness_contract": {
                    "fp32_full_output": True, "preserve_input": True,
                    "atol": None, "rtol": None,
                    "note": "record complete error distributions; no new float acceptance threshold"},
                "extra_correctness": [
                    {"shape": [8], "batch": 2, "kind": "rank1/noaxis"},
                    {"shape": [8, 16], "batch": 2, "kind": "rank2/strided"},
                    {"shape": [8, 8, 8], "batch": 1, "kind": "rank3/fused"},
                    {"shape": [8, 8, 8], "batch": 1, "kind": "rank3/packed-policy"},
                    {"shape": [8, 16], "batch": 2, "kind": "fp64/axis-order-forward"},
                    {"shape": [16, 8, 8], "batch": 1, "kind": "fp64/axis-order-inverse"}],
                "timing_contract": {
                    "cases": "only the eight frozen direct points",
                    "eager": "resident CUDA events, output buffer reused",
                    "graph": "capture repeated calls after warmup; output buffer contract identical",
                    "warmup": 10, "repeats": 20}}
    out.mkdir(parents=True, exist_ok=True)
    target = out / "manifest.json"
    if target.exists() and target.read_bytes() != (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode():
        raise RuntimeError(f"refusing to overwrite existing manifest: {target}")
    target.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return target


def cpu_check(manifest: Path) -> dict:
    d = json.loads(manifest.read_text())
    if digest(FROZEN) != d["frozen_report_sha256"]:
        raise RuntimeError("frozen report changed")
    assert len(d["cases"]) == 8
    assert all(c["mapping"]["contract"]["fft_type"] == 2 for c in d["cases"])
    assert all(c["mapping"]["contract"]["normalize_inverse"] is False for c in d["cases"])
    assert all(c["mapping"]["physical"]["output_permutation"] == [] for c in d["cases"])
    assert d["cases"][-1]["mapping"]["physical"]["axis_order"] == [1, 0]
    return {"status": "cpu-contract-pass", "cases": len(d["cases"]),
            "extra_correctness": len(d["extra_correctness"]),
            "frozen_report_sha256": d["frozen_report_sha256"]}


def worker(args: argparse.Namespace) -> None:
    """Run one library only. Kept separate from the parent to prevent symbol mixing."""
    import torch
    sys.path.insert(0, str(ROOT / "python"))
    from cubutterfly import (Library, Plan, PlanOptions, Transform, DataType,
                             ComputeType, FftType, ExecutionMode, AlgorithmPolicy)
    d = json.loads(args.manifest.read_text())
    mapping_override = json.loads(args.mapping_file.read_text()) if args.mapping_file else {}
    device = torch.device("cuda", args.device)
    report = {"library": str(args.library), "library_sha256": digest(args.library),
              "device": str(device), "rows": [], "status": "running"}
    for case in d["cases"]:
        shape, batch = tuple(case["shape"]), case["batch"]
        mapping_key = f"{','.join(map(str, shape))}:b{batch}"
        # Small deterministic worker payload is used for smoke validation; the
        # eight frozen points use the same tensor contract but may be skipped by
        # a caller lacking the requested GPU memory.
        n = math.prod(shape)
        if n > args.max_elements:
            continue
        gen = torch.Generator(device=device).manual_seed(20261012 + n + batch)
        spectrum = torch.randn((batch, *shape[:-1], shape[-1] // 2 + 1), device=device,
                               dtype=torch.complex64, generator=gen)
        original = spectrum.clone()
        transform = Transform(storage=DataType.FP32, compute=ComputeType.FP32,
            extents=list(shape), batch=batch, inverse=True, normalize_inverse=False,
            fft_type=FftType.C2R, fft_output_scale=1.0,
            execution_mode=ExecutionMode.XD)
        if mapping_key in mapping_override:
            options = PlanOptions(policy=AlgorithmPolicy.EXPLICIT,
                                  mapping=mapping_override[mapping_key],
                                  execution_mode=ExecutionMode.XD)
            selection_policy = "explicit old mapping replay"
        else:
            options = PlanOptions(policy=AlgorithmPolicy.DEFAULT,
                                  execution_mode=ExecutionMode.XD)
            selection_policy = "default (old/new independently selected)"
        plan = Plan(Library(args.library), transform, options)
        workspace = None
        workspace_size = plan.workspace_size()
        if workspace_size:
            workspace = torch.empty(workspace_size, dtype=torch.uint8, device=device)
            plan.set_workspace(workspace, workspace_size)
        out = torch.empty((batch, *shape), device=device, dtype=torch.float32)
        plan.execute(spectrum, out)
        torch.cuda.synchronize(device)
        row = {"shape": list(shape), "batch": batch,
            "input_preserved_max_abs": float((spectrum-original).abs().max().item()),
            "output_finite": bool(torch.isfinite(out).all().item()),
            "output_sha256": hashlib.sha256(out.detach().cpu().numpy().tobytes()).hexdigest(),
            "output_max_abs": float(out.abs().max().item()),
            "mapping_axis_order": case["mapping"]["physical"]["axis_order"],
            "selected_mapping": plan.mapping_json(),
            "selection_policy": selection_policy}
        # Timing is deliberately confined to the original eight rows and uses
        # the same reused output buffer for eager and graph calls.
        for _ in range(d["timing_contract"]["warmup"]):
            plan.execute(spectrum, out)
        torch.cuda.synchronize(device)
        start, stop = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        for _ in range(d["timing_contract"]["repeats"]):
            plan.execute(spectrum, out)
        stop.record(); stop.synchronize()
        row["eager_median_ms"] = start.elapsed_time(stop) / d["timing_contract"]["repeats"]
        try:
            graph = torch.cuda.CUDAGraph()
            torch.cuda.synchronize(device)
            graph.capture_begin()
            plan.execute(spectrum, out)
            graph.capture_end()
            graph.replay(); torch.cuda.synchronize(device)
            start, stop = torch.cuda.Event(True), torch.cuda.Event(True)
            start.record()
            for _ in range(d["timing_contract"]["repeats"]): graph.replay()
            stop.record(); stop.synchronize()
            row["graph_median_ms"] = start.elapsed_time(stop) / d["timing_contract"]["repeats"]
            row["graph_status"] = "captured"
        except Exception as exc:
            row["graph_status"] = "unavailable"
            row["graph_error_type"] = type(exc).__name__
            row["graph_error"] = str(exc)
        report["rows"].append(row)
    report["status"] = "pass" if all(r["input_preserved_max_abs"] == 0 for r in report["rows"]) else "input-mutated"
    args.result.write_text(json.dumps(report, indent=2) + "\n")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=OUT)
    p.add_argument("--old-library", type=Path)
    p.add_argument("--new-library", type=Path)
    p.add_argument("--manifest", type=Path)
    p.add_argument("--run", action="store_true", help="explicitly launch GPU workers")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--max-elements", type=int, default=1 << 20)
    p.add_argument("--worker", action="store_true")
    p.add_argument("--library", type=Path)
    p.add_argument("--result", type=Path)
    p.add_argument("--mapping-file", type=Path)
    args = p.parse_args()
    if args.worker:
        if not args.library or not args.result or not args.manifest:
            p.error("worker requires --library --result --manifest")
        worker(args); return 0
    manifest = args.manifest or prepare(args.output)
    print(json.dumps(cpu_check(manifest), indent=2))
    if not args.run:
        return 0
    if not args.old_library or not args.new_library:
        p.error("--run requires both --old-library and --new-library")
    mapping_file = args.output / "old_mappings.json"
    for label, library in (("old", args.old_library), ("new", args.new_library)):
        if not library.is_file():
            p.error(f"missing {label} library: {library}")
        result = args.output / f"{label}.json"
        # Preserve the guard's UUID-visible device. Inside run_isolated_gpu the
        # selected UUID is the only visible device, so worker index 0 remains
        # the intended physical GPU; replacing it with host index 0 would
        # silently benchmark the wrong card.
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", str(args.device))
        worker_command = [sys.executable, __file__, "--worker", "--manifest", str(manifest),
                        "--library", str(library), "--result", str(result),
                        "--device", str(args.device), "--max-elements", str(args.max_elements)]
        if label == "new":
            worker_command += ["--mapping-file", str(mapping_file)]
        subprocess.run(worker_command,
                       check=True, env={**os.environ, "CUDA_VISIBLE_DEVICES": visible})
        if label == "old":
            old_report = json.loads(result.read_text())
            mapping_file.write_text(json.dumps({
                f"{','.join(map(str,row['shape']))}:b{row['batch']}": row['selected_mapping']
                for row in old_report['rows']}, indent=2) + "\n")
    old = json.loads((args.output / "old.json").read_text())
    new = json.loads((args.output / "new.json").read_text())
    if len(old["rows"]) != len(new["rows"]):
        raise RuntimeError("old/new produced different case coverage")
    comparisons = []
    for before, after in zip(old["rows"], new["rows"]):
        if before["shape"] != after["shape"] or before["batch"] != after["batch"]:
            raise RuntimeError("old/new case order mismatch")
        comparisons.append({"shape": before["shape"], "batch": before["batch"],
                            "output_exact": before["output_sha256"] == after["output_sha256"],
                            "old_eager_ms": before["eager_median_ms"],
                            "new_eager_ms": after["eager_median_ms"],
                            "old_graph_status": before["graph_status"],
                            "new_graph_status": after["graph_status"]})
    (args.output / "comparison.json").write_text(json.dumps({
        "status": "pass" if all(row["output_exact"] for row in comparisons) else "output-mismatch",
        "comparison_policy": "exact output bytes and input preservation; no tolerance",
        "rows": comparisons}, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
