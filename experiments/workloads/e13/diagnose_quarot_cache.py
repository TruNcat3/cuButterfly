#!/usr/bin/env python3
"""Isolate pinned int4 KV-cache correctness; no model or performance claim."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import torch

import run_quarot_random_end_to_end as runner
from quarot_random_checks import compare_outputs


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-uuid", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--quantizer-warmup", type=int, default=0)
    parser.add_argument("--unoptimized-quantizer", action="store_true")
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != args.expected_uuid:
        parser.error("run under the UUID-isolated guard")
    from quarot.transformers.kv_cache import (MultiLayerPagedKVCache4Bit,
        matmul_had_cuda, asym_quantize_and_pack_i4, unpack_i4_and_asym_dequantize)
    if args.unoptimized_quantizer:
        import quarot.transformers.kv_cache as cache_module
        original_quantizer = asym_quantize_and_pack_i4
        def unoptimized(value):
            with torch.jit.optimized_execution(False):
                return original_quantizer(value)
        asym_quantize_and_pack_i4 = unoptimized
        cache_module.asym_quantize_and_pack_i4 = unoptimized

    output = args.output or Path(os.environ["XD_REPORT_DIR"]) / "cache.json"
    report = {"scope": "pinned int4 KV-cache operator diagnostic, not application timing",
              "gpu_uuid": args.expected_uuid, "gpu_name": torch.cuda.get_device_name(),
              "quantizer_warmup_calls_per_shape": args.quantizer_warmup,
              "torchscript_quantizer_optimization_disabled": args.unoptimized_quantizer,
              "cases": []}
    device = torch.device("cuda:0")
    generator = torch.Generator(device=device).manual_seed(runner.SEED + 17)
    for prefill in (8, 32, 128):
        keys = torch.randn((1, prefill + 1, 32, 128), device=device,
                           generator=generator, dtype=torch.float16)
        values = torch.randn(keys.shape, device=device, generator=generator, dtype=torch.float16)
        query = torch.randn((1, 1, 32, 128), device=device, generator=generator, dtype=torch.float16)
        rotated_keys = matmul_had_cuda(keys, torch.float16)
        # Isolate TorchScript profiling/fusion specialization from cache storage.
        # This changes only setup, never weights, dtype, or arithmetic code.
        for _ in range(args.quantizer_warmup):
            for tensor in (rotated_keys, values):
                asym_quantize_and_pack_i4(tensor)
                asym_quantize_and_pack_i4(tensor[:, :prefill].contiguous())
                asym_quantize_and_pack_i4(tensor[:, prefill:].contiguous())
        expected_k, ks, kz = asym_quantize_and_pack_i4(rotated_keys)
        expected_v, vs, vz = asym_quantize_and_pack_i4(values)
        k_cpu = unpack_i4_and_asym_dequantize(expected_k.cpu(), ks.cpu().double(), kz.cpu().double())
        v_cpu = unpack_i4_and_asym_dequantize(expected_v.cpu(), vs.cpu().double(), vz.cpu().double())
        q_cpu = matmul_had_cuda(query, torch.float16).cpu().double()
        scores = torch.einsum("bqhd,bkhd->bhqk", q_cpu, k_cpu) / math.sqrt(128)
        oracle = torch.einsum("bhqk,bkhd->bqhd", scores.softmax(-1), v_cpu)
        for clear_unused in (False, True):
            repetitions = []
            outputs = []
            for repeat in range(3):
                cache = MultiLayerPagedKVCache4Bit(1, prefill + 1, prefill + 1,
                                                   device, 1, 32, 128)
                if clear_unused:
                    cache.pages.zero_()
                    cache.scales.zero_()
                cache.update(keys[:, :prefill].contiguous(), values[:, :prefill].contiguous(),
                             0, {"attention_mask": None})
                decode = cache.update(keys[:, prefill:].contiguous(), values[:, prefill:].contiguous(),
                                      0, {"attention_mask": None})
                observed = decode(query).cpu()
                outputs.append(observed)
                # Cache layout is [page, layer, K/V, head, token, packed-component].
                stored_k = cache.pages[0, 0, 0].transpose(0, 1).unsqueeze(0).cpu()
                stored_v = cache.pages[0, 0, 1].transpose(0, 1).unsqueeze(0).cpu()
                stored_ks = cache.scales[0, 0, 0].transpose(0, 1).unsqueeze(0).cpu()
                stored_vs = cache.scales[0, 0, 1].transpose(0, 1).unsqueeze(0).cpu()
                repetitions.append({
                    "repeat": repeat,
                    "key_bytes_mismatches": int((stored_k != expected_k.cpu()).sum()),
                    "value_bytes_mismatches": int((stored_v != expected_v.cpu()).sum()),
                    "key_params_mismatches": int((stored_ks != torch.cat((ks, kz), -1).cpu()).sum()),
                    "value_params_mismatches": int((stored_vs != torch.cat((vs, vz), -1).cpu()).sum()),
                    "cpu_float64_attention": compare_outputs(oracle, observed),
                })
            report["cases"].append({"prefill": prefill, "decode": 1,
                                    "zero_initialized_storage": clear_unused,
                                    "repetitions": repetitions,
                                    "repeat_errors": [compare_outputs(outputs[0], value) for value in outputs[1:]]})
    report["status"] = "diagnostic_complete_unassessed"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "output": str(output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
