#!/usr/bin/env python3
"""Bounded reduced-layer numerical diagnosis, never a 7B or timing result."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess

import torch

import run_quarot_random_end_to_end as runner
from quarot_random_checks import compare_outputs, cpu_hadamard_reference, state_hash, state_summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--expected-uuid", required=True)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--prefill", type=int, default=8)
    parser.add_argument("--decode", type=int, default=1)
    parser.add_argument("--seed", type=int, default=runner.SEED)
    parser.add_argument("--attention-backend", choices=("torch_sdpa",), required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cpu-oracle", action="store_true", help="Check each real Hadamard input against a complete float64 CPU oracle")
    parser.add_argument("--validation-warmup", type=int, default=0)
    parser.add_argument("--compute-contract", choices=("native", "fp32_composed"), default="native")
    args = parser.parse_args()
    if not 1 <= args.layers <= 4 or not 1 <= args.prefill <= 32 or not 1 <= args.decode <= 2:
        parser.error("bounded diagnosis requires layers<=4, prefill<=32, decode<=2 and positive sizes")
    output = args.output or Path(os.environ["XD_REPORT_DIR"]) / "numerics.json"
    report = {"status": "running", "scope": "reduced-layer diagnosis; not a 7B result",
              "performance_measurements": False, "tolerance_assessed": False,
              "attention_backend": args.attention_backend, "seed": args.seed,
              "layers": args.layers, "prefill": args.prefill, "decode": args.decode,
              "library": str(args.library.resolve()), "library_sha256": runner._sha256(args.library),
              "expected_gpu_uuid": args.expected_uuid}
    report["cpu_oracle"] = args.cpu_oracle
    report["validation_warmup_calls"] = args.validation_warmup
    report["compute_contract"] = args.compute_contract
    adapter = None
    handles = []
    try:
        if os.environ.get("CUDA_VISIBLE_DEVICES") != args.expected_uuid:
            raise RuntimeError("Diagnosis must run under the UUID-isolated GPU guard")
        identity = subprocess.check_output([
            "nvidia-smi", "--id=" + args.expected_uuid,
            "--query-gpu=uuid,name,memory.total", "--format=csv,noheader"], text=True).strip()
        report["gpu_identity"] = identity
        from quantized_llama import modeling_llama
        import quarot.functional
        from quarot.nn import OnlineHadamard
        from quarot_adapter import QuaRotFWHTAdapter, patched_quarot_hadamard
        runner._install_sdpa_fallback()
        runner._patch_transformers_cache_compat()
        original_config = runner._config

        def reduced_config(module):
            config = original_config(module)
            config.num_hidden_layers = args.layers
            return config

        runner._config = reduced_config
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
        model = runner._build_model(modeling_llama, device, args.seed)
        report["architecture"] = {key: getattr(model.config, key) for key in (
            "hidden_size", "intermediate_size", "num_attention_heads", "num_key_value_heads")}
        report["state_sha256"] = state_hash(model)
        report["state_summary"] = state_summary(model)
        generator = torch.Generator(device=device).manual_seed(args.seed + 1)
        input_ids = torch.randint(0, 32000, (1, args.prefill), device=device, generator=generator)
        next_token = torch.randint(0, 32000, (1, 1), device=device, generator=generator)
        traces = {}
        counts = {}
        boundary = {"name": None}

        def capture(name):
            def hook(module, inputs, result):
                del module, inputs
                value = result[0] if isinstance(result, tuple) else result
                index = counts.get(name, 0)
                counts[name] = index + 1
                phase = "prefill" if index == 0 else f"decode_{index - 1}"
                traces[f"{phase}/{name}"] = value.detach().cpu().clone()
            return hook

        for index, layer in enumerate(model.model.layers):
            for suffix, module in (("input_norm", layer.input_layernorm),
                                   ("attention", layer.self_attn),
                                   ("post_attention_norm", layer.post_attention_layernorm),
                                   ("mlp", layer.mlp), ("decoder", layer)):
                handles.append(module.register_forward_hook(capture(f"layer_{index}/{suffix}")))
        for name, module in model.named_modules():
            if isinstance(module, OnlineHadamard):
                def mark(module, inputs, name=name):
                    del module, inputs
                    boundary["name"] = name
                handles.append(module.register_forward_pre_hook(mark))

        def run():
            traces.clear()
            counts.clear()
            logits = runner._validation_outputs(model, input_ids, next_token, args)
            return logits, dict(traces)

        for _ in range(args.validation_warmup):
            run()
        baseline_logits, baseline_layers = run()
        repeat_logits, repeat_layers = run()
        report["baseline_repeat"] = {
            "logits": {name: compare_outputs(value, repeat_logits[name]) for name, value in baseline_logits.items()},
            "layers": {name: compare_outputs(value, repeat_layers[name]) for name, value in baseline_layers.items()}}
        native = quarot.functional.matmul_hadU_cuda
        adapter = QuaRotFWHTAdapter(args.library, device=device, compute_contract=args.compute_contract)
        local_checks = []

        class PairedAdapter:
            @torch.inference_mode()
            def matmul_hadU_cuda(self, values, hadK, K):
                # Separate clones prevent either implementation from mutating
                # the shared input or the other's reference output.
                dao = native(values.clone(), None if hadK is None else hadK.clone(), K).clone()
                actual = adapter.matmul_hadU_cuda(values.clone(), None if hadK is None else hadK.clone(), K)
                check = {"call_index": len(local_checks), "module": boundary["name"],
                                     "input_shape": list(values.shape), "input_dtype": str(values.dtype),
                                     "K": K, "input_nonfinite_count": int((~torch.isfinite(values)).sum()),
                                     "comparison": compare_outputs(dao.cpu(), actual.detach().cpu())}
                if args.cpu_oracle:
                    oracle = cpu_hadamard_reference(values, hadK, K)
                    check["cpu_float64"] = {
                        "dao": compare_outputs(oracle, dao.cpu()),
                        "cubutterfly": compare_outputs(oracle, actual.detach().cpu()),
                        "ideal_fp16_rounding": compare_outputs(oracle, oracle.to(values.dtype)),
                    }
                local_checks.append(check)
                return actual

        with patched_quarot_hadamard(adapter=PairedAdapter()):
            treatment_logits, treatment_layers = run()
        report["treatment"] = {
            "logits": {name: compare_outputs(value, treatment_logits[name]) for name, value in baseline_logits.items()},
            "layers": {name: compare_outputs(value, treatment_layers[name]) for name, value in baseline_layers.items()}}
        report["same_input_hadamard_checks"] = local_checks
        report["adapter_setup"] = adapter.plan_statistics()
        report["final_state_sha256"] = state_hash(model)
        report["status"] = "diagnostic_complete_unassessed"
    except Exception as exc:
        report.update(status="blocked_diagnosis", error=f"{type(exc).__name__}: {exc}")
    finally:
        for handle in handles:
            handle.remove()
        if adapter is not None:
            adapter.close()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"status": report["status"], "output": str(output)}), flush=True)
    return 0 if report["status"] == "diagnostic_complete_unassessed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
