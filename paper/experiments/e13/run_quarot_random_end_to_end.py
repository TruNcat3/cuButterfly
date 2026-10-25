#!/usr/bin/env python3
"""Run the QuaRot FWHT replacement with a reproducible random model state.

The pinned Llama-2-7B architecture receives valid seeded random int4 weights.
State hashes and finite prefill/decode logits are checked before timing.
Output errors establish implementation equivalence, not model quality.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import random
import statistics
import sys
import time
import traceback
from contextlib import nullcontext
from contextvars import ContextVar
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "results/e13_assets/sources/QuaRot"
sys.path.insert(0, str(SOURCE / "e2e"))
sys.path.insert(0, str(SOURCE))
sys.path.insert(0, str(Path(__file__).resolve().parent))


SEED = 20261003
from quarot_random_checks import (random_quantized_weight, state_hash, state_summary,
                                 checked_attention_positions, compare_outputs)


class GraphCaptureUnsupported(RuntimeError):
    pass


class GraphNumericalValidationError(RuntimeError):
    """Carry per-output graph diagnostics into the durable report."""

    def __init__(self, label, checks):
        super().__init__(f"CUDA graph {label}-token numerical validation failed")
        self.label = label
        self.checks = checks


def _prepare_graph_cache(cache, prefill, decode):
    """Materialize unchanged paged-cache metadata outside CUDA capture."""
    original_specs = cache.get_cache_specs_for_flash_infer
    metadata = {}
    for length in range(prefill, prefill + decode + 1):
        cache.length = length
        metadata[length] = original_specs(None)

    def fixed_specs(attention_mask):
        if attention_mask is not None:
            raise RuntimeError("Fixed-shape CUDA graph cache does not accept an attention mask")
        if cache.length not in metadata:
            raise RuntimeError("CUDA graph cache length is outside its prepared sequence")
        return metadata[cache.length]

    cache.get_cache_specs_for_flash_infer = fixed_specs
    cache.length = 0
    cache._needs_init = [True] * len(cache._needs_init)
    return metadata


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validated_graph_extension(path=None):
    extension = (Path(path) if path is not None else
                 ROOT / "results/e13_assets/quarot_graph_stream_build/_CUDA_graph.so").resolve()
    receipt_path = extension.with_name("build_receipt.json")
    if not extension.is_file() or not receipt_path.is_file():
        raise RuntimeError("Graph requires a current-stream QuaRot extension and build_receipt.json; "
                           "run paper/experiments/e13/build_quarot_graph_extension.py")
    receipt = json.loads(receipt_path.read_text())
    if (receipt.get("protocol") != "quarot_current_stream_only_v1" or
            receipt.get("extension_sha256") != _sha256(extension)):
        raise RuntimeError("Graph QuaRot extension protocol or build receipt hash mismatch")
    return extension, {"path": str(receipt_path), "sha256": _sha256(receipt_path),
                       "protocol": receipt["protocol"], "extension_sha256": receipt["extension_sha256"]}


def _seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _config(modeling_llama):
    # These are the public Llama-2-7B architectural values.  Avoiding
    # AutoConfig.from_pretrained makes the random performance run independent
    # of a model checkpoint or network access.
    config = modeling_llama.QuarotLlamaConfig(
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=32,
        hidden_act="silu",
        max_position_embeddings=4096,
        initializer_range=0.02,
        rms_norm_eps=1.0e-5,
        use_cache=True,
        pretraining_tp=1,
        attn_implementation="flash_attention_2",
    )
    config._attn_implementation = "flash_attention_2"
    return config


def _install_sdpa_fallback() -> None:
    """Use PyTorch SDPA when FlashAttention is unavailable.

    This keeps the random-weight application smoke test runnable on the
    shared environment.  The report labels this mode explicitly; it is not a
    replacement for the upstream FlashAttention performance configuration.
    """

    from transformers import modeling_utils
    import transformers.models.llama.modeling_llama as llama

    @classmethod
    def bypass_flash_check(cls, config, torch_dtype=None, device_map=None,
                           check_device_map=True, hard_check_only=False):
        del cls, torch_dtype, device_map, check_device_map, hard_check_only
        config._attn_implementation = "flash_attention_2"
        return config

    modeling_utils.PreTrainedModel._check_and_enable_flash_attn_2 = bypass_flash_check

    def sdpa_flash(query, key, value, dropout_p=0.0, softmax_scale=None, causal=True):
        del softmax_scale
        output = torch.nn.functional.scaled_dot_product_attention(
            query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2),
            dropout_p=dropout_p, is_causal=causal)
        return output.transpose(1, 2)

    llama.flash_attn_func = sdpa_flash
    llama.is_flash_attn_greater_or_equal_2_10 = lambda: True


def _patch_transformers_cache_compat() -> None:
    """Bridge the pinned QuaRot cache to the Transformers 4.38 API."""

    from quarot.transformers.kv_cache import MultiLayerPagedKVCache4Bit
    from transformers.cache_utils import Cache, DynamicCache

    if not hasattr(MultiLayerPagedKVCache4Bit, "__len__"):
        MultiLayerPagedKVCache4Bit.__len__ = lambda self: int(self.pages.shape[1])

    original_from_legacy = DynamicCache.from_legacy_cache

    @classmethod
    def from_legacy_cache_compat(cls, past_key_values=None):
        if isinstance(past_key_values, Cache):
            return past_key_values
        return original_from_legacy(past_key_values)

    DynamicCache.from_legacy_cache = from_legacy_cache_compat

    # The pinned QuaRot attention calls rotary_emb(x, seq_len=...) while the
    # installed Transformers release additionally requires position_ids.
    from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
    from quantized_llama.modeling_llama import QuarotFP16LlamaAttention
    import inspect
    original_rotary_forward = LlamaRotaryEmbedding.forward
    if "position_ids" not in inspect.signature(original_rotary_forward).parameters:
        raise RuntimeError("Unsupported Transformers RoPE API")
    active_positions = ContextVar("quarot_position_ids", default=None)
    original_attention_forward = QuarotFP16LlamaAttention.forward
    attention_signature = inspect.signature(original_attention_forward)

    def attention_forward_compat(self, *args, **kwargs):
        bound = attention_signature.bind(self, *args, **kwargs)
        hidden_states = bound.arguments["hidden_states"]
        cache = bound.arguments.get("past_key_value")
        if not isinstance(cache, MultiLayerPagedKVCache4Bit):
            raise RuntimeError("RoPE bridge requires the pinned paged cache")
        # QuaRot updates the shared length in layer 0, before later layers.
        positions = checked_attention_positions(bound.arguments.get("position_ids"),
                                                hidden_states.shape[1], cache.length, self.layer_idx)
        token = active_positions.set(positions)
        try:
            return original_attention_forward(self, *args, **kwargs)
        finally:
            active_positions.reset(token)

    QuarotFP16LlamaAttention.forward = attention_forward_compat

    def rotary_forward_compat(self, x, position_ids=None, seq_len=None):
        if position_ids is None:
            position_ids = active_positions.get()
            if position_ids is None or position_ids.shape[-1] != x.shape[1]:
                raise RuntimeError("RoPE call is outside a validated attention position context")
        return original_rotary_forward(self, x, position_ids, seq_len=seq_len)

    LlamaRotaryEmbedding.forward = rotary_forward_compat


def _build_model(modeling_llama, device: torch.device, seed: int):
    _seed(seed)
    model = modeling_llama.QuarotLlamaForCausalLM(config=_config(modeling_llama))
    from quarot.nn import Linear4bit
    generator = torch.Generator(device="cpu").manual_seed(seed + 2)
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, Linear4bit):
                packed, scales = random_quantized_weight(module.out_features, module.in_features, generator)
                module.weight.copy_(packed)
                module.weight_scales.copy_(scales)
                if module.bias is not None:
                    module.bias.zero_()
    model.eval().half().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for module in model.modules():
        if isinstance(module, Linear4bit):
            if not bool(torch.isfinite(module.weight_scales).all() and (module.weight_scales > 0).all()):
                raise RuntimeError("Invalid quantized random scales")
    return model


@torch.inference_mode()
def _validation_outputs(model, input_ids, next_token, args):
    model._expected_max_length = args.prefill + args.decode
    output = model(input_ids)
    outputs = {"prefill": output.logits.detach().cpu()}
    for index in range(args.decode):
        output = model(next_token, past_key_values=output.past_key_values)
        outputs[f"decode_{index}"] = output.logits.detach().cpu()
    return outputs


def _sync(device: torch.device) -> None:
    torch.cuda.synchronize(device)


def _timed(fn, device: torch.device) -> tuple[float, float]:
    _sync(device)
    begin = time.perf_counter()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    fn()
    end.record()
    _sync(device)
    return (time.perf_counter() - begin) * 1000.0, start.elapsed_time(end)


def _case_functions(model, input_ids, next_token, prefill: int, decode: int):
    def prefill_call():
        model._expected_max_length = prefill + decode
        return model(input_ids)

    # Build one cache before timing decode.  Both variants use the same cache
    # construction protocol; its allocation is setup, not decode latency.
    _sync(input_ids.device)
    initial = prefill_call()
    _sync(input_ids.device)
    cache = initial.past_key_values

    def decode_call():
        cache.length = prefill
        for _ in range(decode):
            model(next_token, past_key_values=cache)

    def end_to_end_call():
        model._expected_max_length = prefill + decode
        output = model(input_ids)
        for _ in range(decode):
            model(next_token, past_key_values=output.past_key_values)

    return {"prefill": prefill_call, "decode": decode_call, "end_to_end": end_to_end_call}


class _CapturedSequence:
    """A fixed-shape CUDA graph for one complete prefill/decode sequence."""

    def __init__(self, model, input_ids, next_token, prefill, decode):
        self.model = model
        self.input_ids = input_ids.detach().clone()
        self.next_token = next_token.detach().clone()
        self.prefill = prefill
        self.decode = decode
        self.graph = torch.cuda.CUDAGraph()
        self.cache = None
        self.logits = []

    def capture(self):
        # The warmup is outside capture so allocator, quantization and FWHT
        # plans are initialized before the graph owns their addresses.
        stream = torch.cuda.Stream(device=self.input_ids.device)
        stream.wait_stream(torch.cuda.current_stream(self.input_ids.device))
        with torch.cuda.stream(stream):
            self.model._expected_max_length = self.prefill + self.decode
            warm = self.model(self.input_ids)
            self.cache = warm.past_key_values
            for _ in range(self.decode):
                self.model(self.next_token, past_key_values=self.cache)
        torch.cuda.synchronize(self.input_ids.device)
        self.cache_metadata = _prepare_graph_cache(self.cache, self.prefill, self.decode)
        self.logits = []
        torch.cuda.synchronize(self.input_ids.device)
        with torch.cuda.stream(stream):
            self.graph.capture_begin()
            try:
                output = self.model(self.input_ids, past_key_values=self.cache)
                self.cache = output.past_key_values
                self.logits.append(output.logits)
                for _ in range(self.decode):
                    output = self.model(self.next_token, past_key_values=self.cache)
                    self.logits.append(output.logits)
            except Exception as exc:
                original = traceback.format_exc()
                try:
                    self.graph.capture_end()
                except Exception:
                    pass
                raise GraphCaptureUnsupported(original) from exc
            self.graph.capture_end()
        torch.cuda.synchronize(self.input_ids.device)

    def replay(self, input_ids=None, next_token=None):
        # Python cache bookkeeping is outside the captured graph.  The graph
        # uses fixed launch arguments for prefill and decode positions, while
        # prefill overwrites every KV page needed by the sequence.
        self.cache.length = 0
        self.cache._needs_init = [True] * len(self.cache._needs_init)
        if input_ids is not None:
            self.input_ids.copy_(input_ids)
        if next_token is not None:
            self.next_token.copy_(next_token)
        self.graph.replay()
        torch.cuda.synchronize(self.input_ids.device)
        return [value.clone() for value in self.logits]

    def timed_replay(self):
        self.cache.length = 0
        self.cache._needs_init = [True] * len(self.cache._needs_init)
        self.input_ids.copy_(self.source_ids)
        self.next_token.copy_(self.source_next)
        self.graph.replay()
        return [value.clone() for value in self.logits]


def _measure_graph_sequence(model, patch_context, input_ids, next_token, args, device,
                            original_reference, changed_reference):
    with patch_context:
        torch.cuda.reset_peak_memory_stats(device)
        setup_start = time.perf_counter()
        sequence = _CapturedSequence(model, input_ids, next_token, args.prefill, args.decode)
        sequence.source_ids, sequence.source_next = input_ids, next_token
        sequence.capture()
        setup_seconds = time.perf_counter() - setup_start
        # Ensure changed token IDs are actually consumed by the captured graph
        # before timing.  Numerical comparison is done by the caller.
        changed_ids = input_ids.detach().clone()
        changed_next = next_token.detach().clone()
        changed_ids[..., 0] = (changed_ids[..., 0] + 1) % 32000
        changed_next[..., 0] = (changed_next[..., 0] + 1) % 32000
        changed = sequence.replay(changed_ids, changed_next)
        changed_snapshot = [value.clone() for value in changed]
        original = sequence.replay(input_ids, next_token)
        restored = sequence.replay(input_ids, next_token)
        checks = {}
        for label, expected, values in (("original", original_reference, original),
                                        ("changed", changed_reference, changed),
                                        ("restored_repeat", dict(zip(original_reference, original)), restored),
                                        ("output_lifetime", dict(zip(original_reference, changed_snapshot)), changed)):
            checks[label] = {key: compare_outputs(expected[key].cpu(), values[index].cpu(),
                                                  0, 0)
                             for index, key in enumerate(expected)}
            finite = all(check["finite"] for check in checks[label].values())
            tolerance_passed = all(check["pass"] for check in checks[label].values())
            # Graph replay must reproduce its own eager backend exactly,
            # independently of the cross-backend application error policy.
            if not finite or not tolerance_passed:
                raise GraphNumericalValidationError(label, checks)
        wall, cuda_ms = [], []
        for _ in range(args.warmup):
            sequence.cache.length = 0
            sequence.graph.replay()
        _sync(device)
        for _ in range(args.calls):
            elapsed, event = _timed(sequence.timed_replay, device)
            wall.append(elapsed)
            cuda_ms.append(event)
        return {
            "warmup_calls": args.warmup,
            "wall_ms": {"end_to_end": wall},
            "cuda_event_ms": {"end_to_end": cuda_ms},
            "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
            "setup_capture_seconds": setup_seconds,
            "graph_numerical_checks": checks,
            "graph_capture": {
                "full_prefill_decode": True,
                "fixed_shape": [args.batch_size, args.prefill, args.decode],
                "input_copy_ownership": "static_tensor_copy_",
                "output_ownership": "clone",
                "cache_reset": "length_zero_before_replay",
                "cache_metadata": "original_unpadded_specs_precomputed_per_step_outside_capture",
                "eager_graph_consistency_atol": 0,
                "eager_graph_consistency_rtol": 0,
                "restored_input_repeat_and_output_lifetime_checked": True,
            },
        }


@torch.inference_mode()
def _measure_variant(model, patch_context, input_ids, next_token, args, device, adapter=None):
    with patch_context:
        torch.cuda.reset_peak_memory_stats(device)
        functions = _case_functions(model, input_ids, next_token, args.prefill, args.decode)
        warmup = {}
        samples = {}
        cuda_samples = {}
        for name, function in functions.items():
            for _ in range(args.warmup):
                function()
            _sync(device)
            plans_before = adapter.plan_statistics()["plans_created"] if adapter else None
            wall, cuda_ms = [], []
            for _ in range(args.calls):
                elapsed, event = _timed(function, device)
                wall.append(elapsed)
                cuda_ms.append(event)
            if adapter and adapter.plan_statistics()["plans_created"] != plans_before:
                raise RuntimeError("FWHT plan creation occurred inside timed calls")
            warmup[name] = args.warmup
            samples[name] = wall
            cuda_samples[name] = cuda_ms
    return {
        "warmup_calls": warmup,
        "wall_ms": samples,
        "cuda_event_ms": cuda_samples,
        "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path, default=ROOT / "build-e13-shared/libcuntt.so")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--prefill", type=int, default=512)
    parser.add_argument("--decode", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--calls", type=int, default=3)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--attention-backend", choices=("flash_attn", "torch_sdpa"),
                        default="flash_attn")
    parser.add_argument("--prepare-check", action="store_true", help="CPU-only small random-state probe; no model execution")
    parser.add_argument("--validation-only", action="store_true", help="GPU numerical comparison without latency measurements")
    parser.add_argument('--require-cache', action='store_true', help='Require all FWHT plans to replay the prepared persistent cache')
    parser.add_argument("--compute-contract", choices=("native", "fp32_composed", "public_fp32_final"), default="native",
                        help="Explicit FP32 accumulation and scale-before-FP16 composition to match Dao")
    parser.add_argument("--algorithm-policy", choices=("default", "measure"), default=None,
                        help="Override FWHT plan selection; measure performs bounded local tuning before timing")
    parser.add_argument("--validation-warmup", type=int, default=4,
                        help="Complete untimed sequences to stabilize TorchScript quantization before comparison")
    parser.add_argument("--variant-order", choices=("baseline-first", "treatment-first"), default="treatment-first")
    parser.add_argument("--execution-mode", choices=("eager", "graph"), default="eager",
                        help="Run eager calls or capture a fixed-shape full prefill+decode CUDA graph")
    parser.add_argument("--quarot-extension", type=Path,
                        help="Isolated current-stream private extension, shared by both variants")
    parser.add_argument("--atol", type=float, help="Explicit output comparison absolute tolerance")
    parser.add_argument("--rtol", type=float, help="Explicit output comparison relative tolerance")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.batch_size, args.prefill, args.decode, args.calls) <= 0 or args.warmup < 1:
        parser.error("batch/prefill/decode/calls must be positive and warmup >= 1")
    if args.validation_warmup < 0:
        parser.error("validation-warmup must be nonnegative")
    if any(value is not None and (not math.isfinite(value) or value < 0) for value in (args.atol, args.rtol)):
        parser.error("tolerances must be finite and nonnegative")
    if args.algorithm_policy is not None:
        os.environ["CUBUTTERFLY_E13_ALGORITHM_POLICY"] = args.algorithm_policy
    output = args.output
    if os.environ.get('XD_REPORT_DIR') and not output.is_absolute():
        output = Path(os.environ['XD_REPORT_DIR']) / output
    report = {
        "test": "e13_quarot_random_weight_end_to_end",
        "seed": args.seed,
        "full_model_run": False,
        "application_claim": "latency_memory_trace_only; no quality claim",
        "library": str(args.library.resolve()),
        "library_sha256": _sha256(args.library) if args.library.is_file() else None,
        "payload": {"batch_size": args.batch_size, "prefill": args.prefill, "decode": args.decode},
        "attention_backend": args.attention_backend,
        "random_weight_distribution": "Gaussian(std=0.02), symmetric per-row int4 [-7,7]",
        "validation_only": args.validation_only,
        "compute_contract": args.compute_contract,
        "validation_warmup_sequences": args.validation_warmup,
        "variant_order": args.variant_order,
        "execution_mode": args.execution_mode,
        "runner_sha256": _sha256(Path(__file__)),
        "adapter_sha256": _sha256(Path(__file__).with_name('quarot_adapter.py')),
        "checks_sha256": _sha256(Path(__file__).with_name('quarot_random_checks.py')),
        "quarot_extension_protocol": "current_stream_only_v1" if args.quarot_extension else "pinned_original",
        "quarot_extension_path": str(args.quarot_extension.resolve()) if args.quarot_extension else None,
        "quarot_extension_sha256": _sha256(args.quarot_extension) if args.quarot_extension else None,
        "algorithm_policy": os.environ.get('CUBUTTERFLY_E13_ALGORITHM_POLICY', 'default'),
        "plan_cache_path": os.environ.get('CUBUTTERFLY_PLAN_CACHE'),
    }
    if args.prepare_check:
        generator = torch.Generator().manual_seed(args.seed + 2)
        packed, scales = random_quantized_weight(16, 32, generator)
        probe = torch.nn.Module()
        probe.register_buffer("weight", packed)
        probe.register_buffer("weight_scales", scales.half())
        report.update(status="pass_cpu_prepare_check", scope="16x32 CPU probe only; no 7B or CUDA validation",
                      probe_state_sha256=state_hash(probe), probe_state_summary=state_summary(probe),
                      positive_finite_scales=bool((scales > 0).all() and torch.isfinite(scales).all()))
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        return 0
    try:
        if args.execution_mode == "graph":
            args.quarot_extension, receipt = _validated_graph_extension(args.quarot_extension)
            report.update(quarot_extension_protocol="current_stream_only_v1",
                          quarot_extension_path=str(args.quarot_extension),
                          quarot_extension_sha256=_sha256(args.quarot_extension),
                          quarot_extension_receipt=receipt)
        import transformers  # noqa: F401
        from quantized_llama import modeling_llama
        from quarot_adapter import QuaRotFWHTAdapter, patched_quarot_hadamard
        if args.quarot_extension:
            specification = importlib.util.spec_from_file_location("_CUDA_graph", args.quarot_extension.resolve())
            extension = importlib.util.module_from_spec(specification)
            specification.loader.exec_module(extension)
            import quarot
            import quarot.transformers.kv_cache as kv_cache
            quarot._CUDA = extension
            kv_cache._CUDA = extension
        if args.attention_backend == "flash_attn":
            import flash_attn  # noqa: F401
        if args.attention_backend == "torch_sdpa":
            _install_sdpa_fallback()
        _patch_transformers_cache_compat()
    except Exception as exc:
        report.update({"status": "blocked_runtime_dependency", "error": f"{type(exc).__name__}: {exc}"})
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        return 2
    if not torch.cuda.is_available():
        report.update({"status": "blocked_no_cuda", "error": "CUDA is unavailable"})
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n")
        return 2
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    report.update(device=str(device), gpu_name=torch.cuda.get_device_name(device),
                  gpu_uuid='GPU-' + str(torch.cuda.get_device_properties(device).uuid).removeprefix('GPU-'),
                  torch=torch.__version__, transformers=transformers.__version__)
    _seed(args.seed)
    input_generator = torch.Generator(device=device)
    input_generator.manual_seed(args.seed + 1)
    input_ids = torch.randint(0, 32000, (args.batch_size, args.prefill),
                              generator=input_generator, device=device, dtype=torch.long)
    next_token = torch.randint(0, 32000, (args.batch_size, 1),
                                generator=input_generator, device=device, dtype=torch.long)
    try:
        baseline_model = _build_model(modeling_llama, device, args.seed)
        baseline_hash = state_hash(baseline_model)
        baseline_state_summary = state_summary(baseline_model)
        for _ in range(args.validation_warmup):
            _validation_outputs(baseline_model, input_ids, next_token, args)
        reference = _validation_outputs(baseline_model, input_ids, next_token, args)
        repeated = _validation_outputs(baseline_model, input_ids, next_token, args)
        report["baseline_repeat"] = {name: compare_outputs(reference[name], repeated[name], 0, 0)
                                     for name in reference}
        if not all(check["pass"] for check in report["baseline_repeat"].values()):
            raise RuntimeError("Native baseline is not repeatable after validation warmup")
        # The adapter changes functions, never weights. Reuse this exact model
        # with a fresh KV cache on every sequence and verify its state again.
        treatment_model = baseline_model
        treatment_hash = state_hash(treatment_model)
        report["state_sha256"] = {"baseline": baseline_hash, "cubutterfly": treatment_hash}
        report["state_summary"] = {"baseline": baseline_state_summary, "cubutterfly": state_summary(treatment_model)}
        if baseline_hash != treatment_hash:
            raise RuntimeError("Baseline and treatment random states differ")
        # The random-weight application serializes all FWHT calls on one
        # inference stream and consumes each result before the next call.
        # Reuse adapter output storage to remove allocator churn from the
        # application path; standalone adapter callers retain the default
        # independent-output contract.
        adapter = QuaRotFWHTAdapter(args.library, device=device,
                                    compute_contract=args.compute_contract,
                                    reuse_buffers=True)
        try:
            with patched_quarot_hadamard(adapter=adapter):
                actual = _validation_outputs(treatment_model, input_ids, next_token, args)
                report["numerical_checks"] = {
                    name: compare_outputs(reference[name], actual[name], args.atol, args.rtol)
                    for name in reference
                }
                report["full_model_run"] = True
                report["adapter_setup"] = adapter.plan_statistics()
                if args.require_cache and any(plan['selection_source'] != 'user_cache'
                                              for plan in report['adapter_setup']['plans']):
                    raise RuntimeError('measurement requested tuned-cache replay but a plan did not use user_cache')
                finite_outputs = all(check["finite"] for check in report["numerical_checks"].values())
                tolerance_requested = args.atol is not None and args.rtol is not None
                tolerance_passed = all(check["pass"] for check in report["numerical_checks"].values())
                # An unassessed floating-point distribution is still useful for
                # timing.  The caller explicitly chose to inspect the complete
                # error distribution before fixing an application threshold.
                # Only non-finite output or an explicitly requested failed gate
                # may stop the performance run.
                if not finite_outputs or (tolerance_requested and not tolerance_passed):
                    report["status"] = ("failed_nonfinite_outputs" if not finite_outputs
                                         else "failed_numerical_gate")
                    output.parent.mkdir(parents=True, exist_ok=True)
                    output.write_text(json.dumps(report, indent=2) + "\n")
                    print(json.dumps(report, indent=2))
                    return 2
                if args.validation_only:
                    report.update(status="pass_validation", adapter_setup=adapter.plan_statistics())
                    output.parent.mkdir(parents=True, exist_ok=True)
                    output.write_text(json.dumps(report, indent=2) + "\n")
                    print(json.dumps(report, indent=2))
                    return 0
            measurements = {}
            order = (["baseline", "cubutterfly"] if args.variant_order == "baseline-first"
                     else ["cubutterfly", "baseline"])
            if args.execution_mode == "graph":
                changed_ids = input_ids.detach().clone()
                changed_next = next_token.detach().clone()
                changed_ids[..., 0] = (changed_ids[..., 0] + 1) % 32000
                changed_next[..., 0] = (changed_next[..., 0] + 1) % 32000
                report["graph_changed_token_validation"] = {}
                for name in order:
                    report["graph_capture_attempt_variant"] = name
                    context = (patched_quarot_hadamard(adapter=adapter)
                               if name == "cubutterfly" else nullcontext())
                    with context:
                        eager_changed = _validation_outputs(baseline_model, changed_ids, changed_next, args)
                    context = (patched_quarot_hadamard(adapter=adapter)
                               if name == "cubutterfly" else nullcontext())
                    measured = _measure_graph_sequence(
                        baseline_model, context, input_ids, next_token, args, device,
                        reference if name == "baseline" else actual, eager_changed)
                    report["graph_changed_token_validation"][name] = measured["graph_numerical_checks"]
                    measurements[name] = measured
            else:
                for name in order:
                    context = patched_quarot_hadamard(adapter=adapter) if name == "cubutterfly" else nullcontext()
                    measurements[name] = _measure_variant(baseline_model, context, input_ids, next_token,
                                                          args, device, adapter if name == "cubutterfly" else None)
            baseline, treatment = measurements["baseline"], measurements["cubutterfly"]
            treatment["adapter_setup"] = adapter.plan_statistics()
        finally:
            adapter.close()
        if state_hash(baseline_model) != baseline_hash:
            raise RuntimeError("Timed baseline random state differs from validated state")
        del baseline_model
        del treatment_model
        torch.cuda.empty_cache()
    except Exception as exc:
        status = "blocked_graph_capture" if isinstance(exc, GraphCaptureUnsupported) else "blocked_runtime_execution"
        report.update({"status": status, "error": f"{type(exc).__name__}: {exc}"})
        if isinstance(exc, GraphNumericalValidationError):
            report["graph_validation_debug"] = {
                "failed_label": exc.label,
                "checks": exc.checks,
                "finite_gate": True,
                "eager_graph_consistency_atol": 0,
                "eager_graph_consistency_rtol": 0,
            }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        return 2
    measured_names = ("end_to_end",) if args.execution_mode == "graph" else ("prefill", "decode", "end_to_end")
    report.update({
        "status": ("pass" if args.atol is not None and args.rtol is not None
                    else "pass_timing_numerical_unassessed"),
        "full_model_run": True,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "baseline": baseline,
        "cubutterfly": treatment,
        "summary": {
            name: {
                "baseline_median_ms": statistics.median(baseline["wall_ms"][name]),
                "cubutterfly_median_ms": statistics.median(treatment["wall_ms"][name]),
                "baseline_over_cubutterfly": (
                    statistics.median(baseline["wall_ms"][name])
                    / statistics.median(treatment["wall_ms"][name])
                ),
            }
            for name in measured_names
        },
        "timing_contract": {
            "plan_and_first_use_excluded": True,
            "same_random_state_seed": args.seed,
            "quality_claim": False,
            "numerical_gate_before_timing": bool(args.atol is not None and args.rtol is not None),
            "finite_output_gate_before_timing": True,
            "error_distribution_reported_without_threshold": args.atol is None or args.rtol is None,
            "variant_order": order,
            "independent_process_trials": False,
            "execution_mode": args.execution_mode,
            "graph_captured_full_prefill_decode": args.execution_mode == "graph",
            "input_copy_and_output_clone_in_timing": args.execution_mode == "graph",
        },
    })
    for name, summary in report['summary'].items():
        for method in ('baseline', 'cubutterfly'):
            milliseconds = summary[f'{method}_median_ms']
            summary[f'{method}_sequences_per_second'] = 1000 * args.batch_size / milliseconds
            tokens = args.prefill if name == 'prefill' else args.decode if name == 'decode' else args.prefill + args.decode
            summary[f'{method}_tokens_per_second'] = 1000 * args.batch_size * tokens / milliseconds
            summary[f'{method}_amortized_ms_per_sequence'] = milliseconds / args.batch_size
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
