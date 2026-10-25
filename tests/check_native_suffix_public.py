#!/usr/bin/env python3
"""Replay prepared native suffix mappings through the installed-style public API."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--library", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if args.prepare_only:
        sys.path.insert(0, str(ROOT / "scripts"))
        from compile_module import compile_mapping
        from research_compile_requests import compile_request
        prepared = json.loads((args.prepared / "prepared.json").read_text())
        rows = []
        for case in prepared["cases"]:
            if not case["name"].startswith("main-") or not case["name"].endswith("fp64"):
                continue
            request = compile_request(case["mapping"])
            module = compile_mapping(request, root=ROOT,
                mathdx=ROOT / "external/mathdx/nvidia/mathdx",
                nvcc=Path("/usr/local/cuda/bin/nvcc"), sm=80,
                cache=args.prepared.resolve() / "public_modules")
            rows.append(dict(name=case["name"], mapping=case["mapping"],
                             request=request, module=str(module)))
        (args.prepared / "public_prepared.json").write_text(json.dumps(rows, indent=2)+"\n")
        return
    if not args.library:
        parser.error("--library is required for GPU replay")
    import cupy as cp
    import numpy as np
    from cubutterfly import Library, Plan, PlanOptions, Transform, DataType, Error

    prepared = json.loads((args.prepared / "prepared.json").read_text())
    cases = json.loads((args.prepared / "public_prepared.json").read_text())
    cache = args.prepared / "public_modules"
    modules_before = sorted(str(path) for path in cache.glob("*/module.so"))
    library = Library(str(args.library.resolve()))
    rows = []
    for case in cases:
        mapping = case["mapping"]
        n, batch = 1 << mapping["logN"], mapping["batch"]
        assets = prepared["references"][f"n{n}-b{batch}-fp64"]
        signal = np.load(assets["input"]["path"])
        for inverse in (False, True):
            reference = np.load(assets["inverse" if inverse else "forward"]["path"])
            source = cp.asarray(signal)
            output = cp.empty_like(source)
            transform = Transform(storage=DataType.COMPLEX_FP64, extents=[n], batch=batch,
                                  inverse=inverse, normalize_inverse=True)
            with Plan(library, transform, PlanOptions(mapping=json.dumps(mapping))) as plan:
                workspace = cp.empty(plan.workspace_size(), dtype=cp.uint8)
                plan.set_workspace(workspace.data.ptr, workspace.nbytes)
                plan.execute(source.data.ptr, output.data.ptr)
                cp.cuda.get_current_stream().synchronize()
                actual = cp.asnumpy(output)
                resolved = json.loads(plan.mapping_json())
                # The public API returns the actual flat butterfly mapping.
                assert resolved["suffix_core"] == "native", resolved
                assert resolved["execution_group_mappings"][1]["core"] == "register-tile"
                assert resolved["suffix_ept"] == 32
                assert resolved["suffix_threads"] == mapping["suffix_threads"]
                rows.append(dict(case=case["name"], inverse=inverse,
                    finite=bool(np.isfinite(actual).all()),
                    relative_l2=float(np.linalg.norm(actual-reference)/np.linalg.norm(reference)),
                    max_abs=float(np.max(np.abs(actual-reference))), mapping=resolved))
                assert rows[-1]["finite"]
            # Missing imported cores retain their explicit classification.
            bad = dict(mapping, suffix_core="cufftdx-block")
            try:
                with Plan(library, transform, PlanOptions(mapping=json.dumps(bad))):
                    raise AssertionError("missing cuFFTDx mapping silently changed core")
            except Error as error:
                assert error.status == 3 and "provider-implementation-missing" in str(error), str(error)
        print("public-checked", case["name"], flush=True)
    modules_after = sorted(str(path) for path in cache.glob("*/module.so"))
    assert modules_before == modules_after, "public replay unexpectedly needed an unprepared module"
    (args.prepared / "public_run.json").write_text(json.dumps(dict(
        library=str(args.library.resolve()),
        library_sha256=hashlib.sha256(args.library.read_bytes()).hexdigest(),
        rows=rows, prepared_module_cache_unchanged=True,
        floating_policy="Same-precision CPU reference errors only; no new threshold"), indent=2)+"\n")


if __name__ == "__main__":
    main()
