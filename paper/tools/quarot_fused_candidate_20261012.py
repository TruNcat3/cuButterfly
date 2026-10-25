#!/usr/bin/env python3
"""Explicit-provider full QuaRot probe, reusing the unchanged model runner hooks."""
import argparse
from contextlib import nullcontext
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import statistics
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "results/paper_targeted_20261012/quarot_fused_candidate"
BASE = ROOT / "paper/experiments/e13"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(name, path):
    location = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(location)
    sys.modules[name] = module
    location.loader.exec_module(module)
    return module


def prepare(batch):
    output = OUTPUT / f"b{batch}"
    if (output / "guard").exists():
        raise ValueError("refusing to change a launched application probe")
    prior_path = ROOT / f"results/paper_targeted_20261012/application_batch/quarot_b{batch}_p1/release/result.json"
    prior = json.loads(prior_path.read_text())
    candidate_path = ROOT / "results/paper_targeted_20261012/fused_short_fwht_candidate/guard/release/result.json"
    candidate = json.loads(candidate_path.read_text())
    if candidate["status"] != "accepted" or not all(row["exact_accepted"] for row in candidate["cases"]):
        raise ValueError("requires all four accepted zero-difference candidate cases")
    provider = candidate["provider"]
    library = Path(prior["library"])
    paths = [Path(__file__).resolve(), BASE / "run_quarot_random_end_to_end.py", BASE / "quarot_adapter.py",
             BASE / "quarot_random_checks.py", ROOT / "python/cubutterfly/fused_short_fwht.py",
             Path(provider["source"]), Path(provider["module"]), library,
             ROOT / "paper/tools/run_with_gpu_telemetry.py", prior_path, candidate_path]
    identities = {str(path): digest(path) for path in paths}
    if digest(provider["module"]) != provider["module_sha256"] or digest(provider["source"]) != provider["source_sha256"]:
        raise ValueError("validated provider changed")
    if digest(BASE / "run_quarot_random_end_to_end.py") != prior["runner_sha256"] or digest(BASE / "quarot_adapter.py") != prior["adapter_sha256"]:
        raise ValueError("original application runner or adapter changed; re-review required")
    frozen = json.loads((ROOT / "results/paper_targeted_20261012/medium_large_mechanism/fwht/launch.json").read_text())
    if digest(library) != frozen["library_sha256"]:
        raise ValueError("application and GPU2 frozen runtime identity disagree")
    protocol = dict(batch_size=batch, prefill=prior["payload"]["prefill"], decode=prior["payload"]["decode"],
                    seed=prior["seed"], warmup=5, calls=10, validation_warmup=prior["validation_warmup_sequences"],
                    atol=0.0, rtol=0.0, attention_backend=prior["attention_backend"])
    command = [frozen["python"], str(ROOT / "scripts/run_isolated_gpu.py"), "--gpu-uuid", frozen["gpu_uuid"],
               "--report-dir", str(output / "guard"), "--until", "2026-10-12T18:00:00+08:00", "--", "env",
               "-u", "CUBUTTERFLY_PLAN_CACHE"]
    environment = dict(frozen["environment"], CUBUTTERFLY_E13_ALGORITHM_POLICY="default")
    command += [f"{key}={value}" for key, value in sorted(environment.items())]
    command += [frozen["python"], str(Path(__file__).resolve()), "--guard-child", "--batch-size", str(batch)]
    spec = dict(schema="cubutterfly-quarot-explicit-fused-candidate-v1", status="prepared-not-executed",
                public_search_integrated=False, provider=provider, identities=identities,
                library=str(library), gpu_uuid=frozen["gpu_uuid"], python=frozen["python"],
                protocol=protocol, expected_state_sha256=prior["state_sha256"]["baseline"],
                prior_input_recipe="Same original seed+1 GPU generator, vocabulary and batch/prefill/decode; regenerate deterministically and hash actual inputs.",
                variant_order=["dao", "public_default_fp32_composed", "explicit_fused_candidate"],
                gate="Full-model finite zero-difference against repeated Dao before timing; preserve failures, no threshold relaxation",
                timing="One independent process per batch; three backends in same process/model, eager wall/CUDA-event, not independent process trials or CUDA graph",
                launch_argv=command, deadline="2026-10-12T18:00:00+08:00")
    output.mkdir(parents=True, exist_ok=True)
    (output / "launch.json").write_text(json.dumps(spec, indent=2) + "\n")
    return spec


def checked_spec(batch):
    spec = json.loads((OUTPUT / f"b{batch}/launch.json").read_text())
    for path, expected in spec["identities"].items():
        if digest(path) != expected:
            raise ValueError("frozen probe identity changed: " + path)
    return spec


def candidate_adapter_class(base, provider):
    class CandidateAdapter(base):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.candidate_calls = {}
            self.fallback_calls = 0

        def _transform_vectors(self, values, extent, full_n):
            import torch
            if values.dtype != torch.float16 or extent not in (32, 64):
                self.fallback_calls += 1
                return super()._transform_vectors(values, extent, full_n)
            if full_n not in self._scales:
                self._scales[full_n] = float(1.0 / torch.tensor(full_n).sqrt())
            scale = self._scales[full_n]
            key = f"n{extent}:b{values.numel() // extent}:scale{scale}"
            self.candidate_calls[key] = self.candidate_calls.get(key, 0) + 1
            return provider(values.contiguous(), scale)

        def plan_statistics(self):
            result = super().plan_statistics()
            result.update(provider="standalone-fused-short-fwht-v1", public_search_integrated=False,
                          explicit_candidate_calls=dict(self.candidate_calls), fallback_calls=self.fallback_calls)
            return result
    return CandidateAdapter


def application(batch):
    spec = checked_spec(batch)
    output = Path(os.environ["XD_REPORT_DIR"])
    sys.path.insert(0, str(ROOT / "python"))
    sys.path.insert(0, str(BASE))
    import torch
    from cubutterfly.fused_short_fwht import FusedShortFWHT
    runner = load("fused_candidate_original_runner", BASE / "run_quarot_random_end_to_end.py")
    import quarot_adapter
    from quantized_llama import modeling_llama
    if spec["protocol"]["attention_backend"] != "torch_sdpa":
        raise ValueError("this frozen probe requires the original torch_sdpa mode")
    runner._install_sdpa_fallback()
    runner._patch_transformers_cache_compat()
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    args = SimpleNamespace(**spec["protocol"])
    report = dict(schema=spec["schema"], public_search_integrated=False, spec=spec, status="running",
                  numerical_checks={}, measurements={}, application_claim="Random-weight complete application consistency/performance only, not model quality")
    def save():
        (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    save()
    adapters = []
    try:
        with torch.inference_mode():
            runner._seed(args.seed)
            generator = torch.Generator(device=device).manual_seed(args.seed + 1)
            input_ids = torch.randint(0, 32000, (batch, args.prefill), generator=generator, device=device, dtype=torch.long)
            next_token = torch.randint(0, 32000, (batch, 1), generator=generator, device=device, dtype=torch.long)
            report["input_sha256"] = {name: hashlib.sha256(value.cpu().numpy().tobytes()).hexdigest() for name, value in
                                      (("input_ids", input_ids), ("next_token", next_token))}
            model = runner._build_model(modeling_llama, device, args.seed)
            before = runner.state_hash(model)
            report["state_sha256"] = before
            if before != spec["expected_state_sha256"]:
                raise RuntimeError("model state differs from original B1/B4 source cohort")
            for _ in range(args.validation_warmup):
                runner._validation_outputs(model, input_ids, next_token, args)
            reference = runner._validation_outputs(model, input_ids, next_token, args)
            repeated = runner._validation_outputs(model, input_ids, next_token, args)
            report["baseline_repeat"] = {name: runner.compare_outputs(reference[name], repeated[name], 0, 0) for name in reference}
            if not all(check["pass"] for check in report["baseline_repeat"].values()):
                raise RuntimeError("Dao full application is not zero-difference repeatable")
            old = quarot_adapter.QuaRotFWHTAdapter(spec["library"], device=device, compute_contract="fp32_composed")
            adapters.append(old)
            candidate_class = candidate_adapter_class(quarot_adapter.QuaRotFWHTAdapter, FusedShortFWHT(spec["provider"]["module"]))
            candidate = candidate_class(spec["library"], device=device, compute_contract="fp32_composed")
            adapters.append(candidate)
            variants = dict(dao=None, public_default_fp32_composed=old, explicit_fused_candidate=candidate)
            for name, adapter in variants.items():
                if adapter is None:
                    continue
                with quarot_adapter.patched_quarot_hadamard(adapter=adapter):
                    actual = runner._validation_outputs(model, input_ids, next_token, args)
                checks = {key: runner.compare_outputs(reference[key], actual[key], 0, 0) for key in reference}
                report["numerical_checks"][name] = checks
                save()
                if not all(check["pass"] for check in checks.values()):
                    raise RuntimeError(name + " failed full-model finite zero-difference gate; not timing")
            for name in spec["variant_order"]:
                adapter = variants[name]
                context = quarot_adapter.patched_quarot_hadamard(adapter=adapter) if adapter is not None else nullcontext()
                report["measurements"][name] = runner._measure_variant(model, context, input_ids, next_token, args, device, adapter)
                save()
            if runner.state_hash(model) != before:
                raise RuntimeError("application weights changed during timing")
            report["adapter_setup"] = {name: adapter.plan_statistics() for name, adapter in variants.items() if adapter is not None}
            report["summary"] = {phase: {name: statistics.median(row["wall_ms"][phase]) for name, row in report["measurements"].items()}
                                 for phase in ("prefill", "decode", "end_to_end")}
            report.update(status="accepted", full_model_run=True, quality_claim=False)
    except Exception as error:
        report.update(status="rejected", error=str(error))
    finally:
        report["adapter_setup"] = {name: adapter.plan_statistics() for name, adapter in
                                    zip(("public_default_fp32_composed", "explicit_fused_candidate"), adapters)}
        for adapter in reversed(adapters):
            adapter.close()
        save()
    return 0 if report["status"] == "accepted" else 1


def guard_child(batch):
    spec = checked_spec(batch)
    sys.path.insert(0, str(ROOT / "paper/tools"))
    import prepare_principle_gap_campaign as base
    attempt = base.guard_context(OUTPUT / f"b{batch}")
    (attempt / "owned_processes.json").write_text(json.dumps(dict(
        pid=os.getpid(), parent_pid=os.getppid(), process_group=os.getpgrp(), session=os.getsid(0),
        attempt=str(attempt), policy="guard-owned wrapper; telemetry owns and cleans application child session"), indent=2) + "\n")
    os.environ["CUBUTTERFLY_PREPARE_EVENTS"] = str(attempt / "preparation_events.jsonl")
    telemetry = load("fused_candidate_application_telemetry", ROOT / "paper/tools/run_with_gpu_telemetry.py")
    previous = sys.argv
    sys.argv = ["run_with_gpu_telemetry.py", "--gpu-uuid", spec["gpu_uuid"], "--stop-on-thermal-throttle",
                "--output-prefix", str(attempt / "telemetry"), "--", spec["python"], str(Path(__file__).resolve()),
                "--application-child", "--batch-size", str(batch)]
    handlers = {number: signal.signal(number, base.interrupted) for number in (signal.SIGTERM, signal.SIGINT)}
    try:
        return telemetry.main()
    finally:
        sys.argv = previous
        for number, handler in handlers.items():
            signal.signal(number, handler)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, choices=(1, 4), required=True)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--guard-child", action="store_true")
    parser.add_argument("--application-child", action="store_true")
    args = parser.parse_args(argv)
    if args.prepare:
        print(json.dumps(prepare(args.batch_size), indent=2))
        return 0
    if args.guard_child:
        return guard_child(args.batch_size)
    if args.application_child:
        return application(args.batch_size)
    parser.error("select CPU --prepare or guard-owned child mode")


if __name__ == "__main__":
    raise SystemExit(main())
