"""Prepare bounded FFT-L sources, full CPU oracles and exact baseline gaps.

No plans, Torch imports, GPU initialization, kernel timings or accuracy verdicts.
The raw semantic manifest is consumable by the existing semantic runner. Scaled
artifacts are separate CPU controls; ordinary runners regenerate raw inputs.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import semantic_campaign_manifest as campaign
import semantic_cpu_pipeline as cpu
import semantic_input_profiles as profiles


def prepare(config_path, library, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"output must be fresh: {output}")
    config = campaign.load_config(config_path)
    if sum(1 for _ in itertools.islice(campaign._requests(config), 33)) > 32:
        raise ValueError("quick preparation is bounded to 32 requested variants")
    manifest = campaign.generate_manifest(config, library, config_path=config_path)
    for case in manifest["cases"]:
        s = case["descriptor"]["semantics"]
        if s["operator"] != 0 or s["fft_type"] != 0 or s["storage"] not in (4, 5) or s["compute"] not in (0, 1):
            raise ValueError("only FP16/BF16 C2C native/FP32 arithmetic is in this preparation scope")
    output.mkdir(parents=True)
    (output / "raw_manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    rows = []
    for case in manifest["cases"]:
        s = case["descriptor"]["semantics"]
        try:
            prepared = cpu.prepare_case(case, reference_max_elements=65536)
        except cpu.runner.Gap as exc:
            rows.append({"semantic_id": case["id"], "status": "preparation-gap",
                         "support_status": exc.status, "reason": str(exc)})
            continue
        cohorts = [("raw-stress", prepared["source"], {"profile": "raw-stress", "reference_input": "original quantized input"})]
        if s["storage"] == 4:
            scaled, metadata = profiles.apply_input_profile(
                s, prepared["source"], profiles.SCALED_PROFILE,
                encode_storage=cpu.runner.encode_storage, decode_storage=cpu.runner.decode_storage)
            cohorts.append((profiles.SCALED_PROFILE, scaled, metadata))
        for profile, source, metadata in cohorts:
            identifier = cpu.runner.digest({"semantic_id": case["id"], "input_profile": metadata})
            artifact = output / f"{identifier}.npz"
            expected = cpu.runner.reference(s, source, max_elements=65536)
            np.savez_compressed(artifact, source=source, reference=expected,
                                storage=cpu.runner.encode_storage(source, s["storage"]))
            rows.append({
                "id": identifier, "semantic_id": case["id"], "status": "cpu-prepared",
                "compute_type": "fft-l", "semantics": s, "input_profile": metadata,
                "storage_precision": "fp16" if s["storage"] == 4 else "bf16",
                "compute_precision": "fp32" if s["compute"] == 1 else ("fp16" if s["storage"] == 4 else "bf16"),
                "accumulation": "fp32" if s["compute"] == 1 else "native",
                "artifact": {"path": artifact.name, "sha256": cpu.runner.sha256_file(artifact)},
                "elements": int(source.size), "oracle_dtype": str(expected.dtype),
                "cubutterfly": case["capabilities"],
                "baselines": {name: cpu.runner.baseline_contract(name, s) for name in ("cufft", "vkfft")},
                "execution_input_binding": "existing semantic runner regenerates raw input" if profile == profiles.RAW_PROFILE
                else "prepared artifact only; use source-bound scaled replay, never ordinary raw runner",
            })
    report = {
        "schema": "cubutterfly-fft-l-quick-preparation-v1", "status": "cpu-prepared",
        "manifest_sha256": manifest["manifest_sha256"], "rows": rows,
        "fingerprints": {"sources": [campaign.fingerprint(path) for path in (
            Path(__file__), Path(cpu.__file__), Path(cpu.runner.__file__), Path(profiles.__file__))],
                         "numpy": np.__version__, "python": sys.version},
        "protocol": {"timing": False, "float_verdict": "error-report", "reference_max_elements": 65536,
                     "reference": "full NumPy complex128 higher-precision diagnostic reference from final quantized input; not a same-precision oracle, low-precision CPU performance baseline, or numerical acceptance",
                     "errors": "retain full actual/reference/error arrays and distributions at GPU execution; no accepted tolerance or accuracy claim during preparation"},
        "gaps": ["cuFFT/VkFFT exact FP16/BF16 semantic adapters are absent; never promote requests to FP32",
                 "BF16 scaled-input profile is not implemented; raw BF16 requests remain present",
                 "FP16/BF16 FP32-arithmetic paths narrow intermediate values back to requested low storage; they do not provide FP32 intermediate storage",
                 "scaled artifacts require explicit source-bound execution; run_fft_range_control additionally requires a compatible measured row and exact mapping",
                 "this bounded C2C rank1/2 population does not cover real FFT, arbitrary ranks, strides or placement"]}
    (output / "preparation.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/fft_l_quick_semantic.json")
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = prepare(args.config, args.library, args.output)
    print(json.dumps({"status": report["status"], "cohort_rows": len(report["rows"]),
                      "report": str(args.output / "preparation.json")}))


if __name__ == "__main__":
    main()
