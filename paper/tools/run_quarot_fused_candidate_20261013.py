#!/usr/bin/env python3
"""Measure the validated short FP16 FWHT provider in the complete QuaRot path.

This probe deliberately keeps the current random-weight application runner and
library unchanged.  It compares Dao, the public cuButterfly adapter, and the
standalone fused N=32/64 provider after a zero-error full-model gate.  The
provider is a candidate integration point, not silently promoted into the
public selector by this experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import statistics
import sys
from contextlib import nullcontext
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "paper/experiments/e13"
PROVIDER_MODULE = ROOT / "results/paper_targeted_20261012/fused_short_fwht_candidate/fused_short_fwht.so"


def load(name: str, path: Path):
    location = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(location)
    sys.modules[name] = module
    location.loader.exec_module(module)
    return module


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--provider-module", type=Path, default=PROVIDER_MODULE)
    parser.add_argument("--batch-size", type=int, choices=(1, 4), default=1)
    parser.add_argument("--prefill", type=int, default=512)
    parser.add_argument("--decode", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--calls", type=int, default=3)
    parser.add_argument("--validation-warmup", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261003)
    parser.add_argument("--process-trial-index", type=int, default=0,
                        help="Rotate backend order across independent processes")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if min(args.batch_size, args.prefill, args.decode, args.warmup, args.calls) <= 0:
        parser.error("batch/prefill/decode/warmup/calls must be positive")
    if args.validation_warmup < 0 or args.process_trial_index < 0:
        parser.error("validation-warmup and process-trial-index must be nonnegative")
    if not args.library.is_file() or not args.provider_module.is_file():
        parser.error("library and provider module must exist")
    if os.environ.get("XD_REPORT_DIR") and not args.output.is_absolute():
        args.output = Path(os.environ["XD_REPORT_DIR"]) / args.output

    # The runner imports the pinned random-weight model and owns the exact
    # validation/timing protocol used by the current QuaRot evidence.
    runner = load("quarot_current_runner", BASE / "run_quarot_random_end_to_end.py")
    adapter_module = load("quarot_current_adapter", BASE / "quarot_adapter.py")
    provider_module = load("fused_short_fwht_provider", ROOT / "python/cubutterfly/fused_short_fwht.py")
    from transformers import __version__ as transformers_version
    import torch
    from quantized_llama import modeling_llama

    runner._install_sdpa_fallback()
    runner._patch_transformers_cache_compat()
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    args_ns = type("Args", (), {
        "batch_size": args.batch_size,
        "prefill": args.prefill,
        "decode": args.decode,
        "warmup": args.warmup,
        "calls": args.calls,
        "seed": args.seed,
        "attention_backend": "torch_sdpa",
        "atol": 0.0,
        "rtol": 0.0,
    })()

    runner._seed(args.seed)
    generator = torch.Generator(device=device).manual_seed(args.seed + 1)
    input_ids = torch.randint(0, 32000, (args.batch_size, args.prefill),
                              generator=generator, device=device, dtype=torch.long)
    next_token = torch.randint(0, 32000, (args.batch_size, 1),
                                generator=generator, device=device, dtype=torch.long)
    model = runner._build_model(modeling_llama, device, args.seed)
    before = runner.state_hash(model)
    for _ in range(args.validation_warmup):
        runner._validation_outputs(model, input_ids, next_token, args_ns)
    reference = runner._validation_outputs(model, input_ids, next_token, args_ns)

    public = adapter_module.QuaRotFWHTAdapter(
        args.library, device=device, compute_contract="fp32_composed", reuse_buffers=True)
    provider = provider_module.FusedShortFWHT(args.provider_module)
    candidate_class = load(
        "quarot_candidate_class", ROOT / "paper/tools/quarot_fused_candidate_20261012.py"
    ).candidate_adapter_class(adapter_module.QuaRotFWHTAdapter, provider)
    candidate = candidate_class(
        args.library, device=device, compute_contract="fp32_composed", reuse_buffers=True)
    variants = {"dao": None, "public_cubutterfly": public, "fused_candidate": candidate}
    backend_names = tuple(variants)
    offset = args.process_trial_index % len(backend_names)
    order = backend_names[offset:] + backend_names[:offset]
    report = {
        "schema": "cubutterfly-quarot-fused-candidate-v2",
        "status": "running",
        "library": str(args.library.resolve()),
        "library_sha256": digest(args.library),
        "provider_module": str(args.provider_module.resolve()),
        "provider_module_sha256": digest(args.provider_module),
        "provider_descriptor": provider_module.descriptor(),
        "protocol": {"batch_size": args.batch_size, "prefill": args.prefill,
                      "decode": args.decode, "seed": args.seed,
                      "validation_warmup": args.validation_warmup,
                      "warmup": args.warmup, "calls": args.calls,
                      "atol": 0.0, "rtol": 0.0, "attention_backend": "torch_sdpa"},
        "gpu": torch.cuda.get_device_name(device),
        "gpu_uuid": str(torch.cuda.get_device_properties(device).uuid),
        "gpu_memory_bytes": torch.cuda.get_device_properties(device).total_memory,
        "torch": torch.__version__,
        "transformers": transformers_version,
        "probe_sha256": digest(Path(__file__)),
        "candidate_adapter_sha256": digest(ROOT / "paper/tools/quarot_fused_candidate_20261012.py"),
        "provider_python_sha256": digest(ROOT / "python/cubutterfly/fused_short_fwht.py"),
        "provider_source_sha256": digest(provider_module.SOURCE),
        "runner_sha256": digest(BASE / "run_quarot_random_end_to_end.py"),
        "adapter_sha256": digest(BASE / "quarot_adapter.py"),
        "model_state_sha256": before,
        "public_search_integrated": False,
        "variant_order": order,
        "process_trial_index": args.process_trial_index,
        "timing_contract": {
            "independent_process_trials": 1,
            "plan_compile_warmup_excluded": True,
            "host_adapter_conversion_and_model_included": True,
            "quality_claim": False,
            "statistical_superiority_established": False,
        },
        "numerical_checks": {},
        "measurements": {},
    }

    try:
        for name, adapter in variants.items():
            context = (adapter_module.patched_quarot_hadamard(adapter=adapter)
                       if adapter is not None else nullcontext())
            with context:
                actual = runner._validation_outputs(model, input_ids, next_token, args_ns)
            report["numerical_checks"][name] = {
                key: runner.compare_outputs(reference[key], actual[key], 0.0, 0.0)
                for key in reference
            }
        if not all(check["pass"] for checks in report["numerical_checks"].values()
                   for check in checks.values()):
            raise RuntimeError("full-model zero-difference gate failed")

        for name in order:
            adapter = variants[name]
            context = (adapter_module.patched_quarot_hadamard(adapter=adapter)
                       if adapter is not None else nullcontext())
            report["measurements"][name] = runner._measure_variant(
                model, context, input_ids, next_token, args_ns, device, adapter)
        report["summary"] = {
            phase: {
                name: statistics.median(report["measurements"][name]["wall_ms"][phase])
                for name in order
            }
            for phase in ("prefill", "decode", "end_to_end")
        }
        report["speedup"] = {
            phase: {name: report["summary"][phase]["dao"] /
                    report["summary"][phase][name]
                    for name in ("public_cubutterfly", "fused_candidate")}
            for phase in report["summary"]
        }
        report["adapter_setup"] = {
            name: adapter.plan_statistics() for name, adapter in variants.items()
            if adapter is not None
        }
        report["status"] = "accepted"
    except Exception as error:
        report.update(status="rejected", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        for adapter in (candidate, public):
            adapter.close()
        if runner.state_hash(model) != before:
            report["state_unchanged"] = False
            report.update(status="rejected", error="model state changed during validation/timing")
        else:
            report["state_unchanged"] = True
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "status": report["status"],
                      "summary": report.get("summary")}, indent=2))
    return 0 if report["status"] == "accepted" else 1


if __name__ == "__main__":
    raise SystemExit(main())
