#!/usr/bin/env python3
"""Short numerical gates for the bounded E13 QuaRot FWHT adapter.

This is deliberately a synthetic operator check.  E13 has no authorized
Llama-2 checkpoint in this workspace, so this script does not run a model or
claim end-to-end quality/performance.  The default reference is a pure
PyTorch radix-2 implementation; ``--dao-reference`` instead calls the
pinned QuaRot function with the installed Dao CUDA extension
``fast_hadamard_transform`` and writes a separate report.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import torch

from quarot_adapter import (
    DEFAULT_LIBRARY,
    PINNED_QUAROT_COMMIT,
    QuaRotFWHTAdapter,
    patched_quarot_hadamard,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
PINNED_HADAMARD = REPO_ROOT / "results/e13_assets/sources/QuaRot/quarot/functional/hadamard.py"
DEFAULT_OUTPUT = REPO_ROOT / "results/paper_profile_rc2/end_to_end/quarot_adapter_correctness.json"
DEFAULT_DAO_OUTPUT = REPO_ROOT / "results/paper_profile_rc2/end_to_end/quarot_adapter_dao_correctness.json"
EXPECTED_GPU2_UUID = "GPU-0eae5f07-33de-321b-70a9-ffeae49309e6"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_hadamard_source() -> ModuleType:
    """Load the pinned function file without requiring QuaRot's CUDA extension."""

    if not PINNED_HADAMARD.is_file():
        raise FileNotFoundError(f"pinned QuaRot source is missing: {PINNED_HADAMARD}")
    spec = importlib.util.spec_from_file_location("e13_pinned_quarot_hadamard", PINNED_HADAMARD)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load pinned source: {PINNED_HADAMARD}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _reference_fwht(values: torch.Tensor, scale: float) -> torch.Tensor:
    """Reference radix-2 FWHT using only PyTorch tensor arithmetic."""

    extent = int(values.shape[-1])
    if extent <= 0 or (extent & (extent - 1)):
        raise ValueError(f"reference requires power-of-two extent, got {extent}")
    transformed = values.contiguous()
    width = 1
    while width < extent:
        grouped = transformed.reshape(*transformed.shape[:-1], extent // (2 * width), 2, width)
        left = grouped[..., 0, :]
        right = grouped[..., 1, :]
        transformed = torch.stack((left + right, left - right), dim=-2).reshape_as(transformed)
        width *= 2
    return transformed * scale


def _gpu_uuid() -> str | None:
    """Resolve the visible device to a UUID when nvidia-smi is available."""

    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    rows: dict[str, str] = {}
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",", 1)]
        if len(fields) == 2:
            rows[fields[0]] = fields[1]
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible and "," not in visible:
        token = visible
        if token.startswith("GPU-"):
            return token
        return rows.get(token)
    # With no remapping, PyTorch cuda:0 is the physical index 0.
    return rows.get("0")


def _case(
    adapter: QuaRotFWHTAdapter,
    *,
    name: str,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    K: int,
    hadK: torch.Tensor | None,
    device: torch.device,
    stream: torch.cuda.Stream | None = None,
    reference_mode: str = "torch",
    hadamard_module: ModuleType | None = None,
) -> dict[str, Any]:
    generator = torch.Generator(device=device)
    generator.manual_seed(20261003 + len(name))
    values = torch.randn(shape, generator=generator, device=device, dtype=dtype) * 0.125
    n = int(shape[-1])
    target_stream = stream or torch.cuda.current_stream(device)
    with torch.cuda.stream(target_stream):
        if reference_mode == "dao":
            if hadamard_module is None:
                raise ValueError("Dao reference mode requires the pinned QuaRot hadamard module")
            # This is the real pinned QuaRot call, whose power-of-two branch
            # invokes fast_hadamard_transform CUDA rather than the torch ref.
            reference = hadamard_module.matmul_hadU_cuda(values, hadK, K)
        elif reference_mode == "torch" and K == 1:
            reference = _reference_fwht(values, 1.0 / (n**0.5))
        elif reference_mode == "torch":
            sub = values.view(*values.shape[:-1], K, n // K)
            reference = _reference_fwht(sub, 1.0 / (n**0.5))
            reference = torch.matmul(hadK.to(device=device, dtype=dtype), reference).reshape(values.shape)
        else:
            raise ValueError(f"unknown reference mode: {reference_mode}")
        result = adapter.matmul_hadU_cuda(values, hadK, K)
        done = torch.cuda.Event()
        done.record(target_stream)
    done.synchronize()

    difference = (result - reference).float()
    max_abs = float(difference.abs().max().item())
    reference_norm = float(reference.float().abs().max().item())
    max_rel = max_abs / (reference_norm + 1.0e-12)
    l2_rel = float(torch.linalg.vector_norm(difference).item()) / (
        float(torch.linalg.vector_norm(reference.float()).item()) + 1.0e-12
    )
    if dtype == torch.float16:
        abs_tolerance, rel_tolerance = 7.5e-2, 1.0e-2
    else:
        abs_tolerance, rel_tolerance = 5.0e-4, 2.0e-4
    return {
        "name": name,
        "shape": list(shape),
        "dtype": str(dtype).replace("torch.", ""),
        "K": K,
        "subtransform": n // K,
        "stream": int(target_stream.cuda_stream),
        "max_abs": max_abs,
        "max_relative": max_rel,
        "relative_l2": l2_rel,
        "reference": "pinned_quarot_dao_cuda" if reference_mode == "dao" else "torch_radix2",
        "tolerance": {"max_abs": abs_tolerance, "max_relative": rel_tolerance},
        "pass": max_abs <= abs_tolerance and max_rel <= rel_tolerance,
    }


def _check_patch_preserves_non_fwht(hadamard_module: ModuleType, adapter: QuaRotFWHTAdapter) -> dict[str, Any]:
    original_get_hadK = hadamard_module.get_hadK
    original_random = hadamard_module.random_hadamard_matrix
    original_fht_module = hadamard_module.fast_hadamard_transform
    with patched_quarot_hadamard(functional_module=hadamard_module, adapter=adapter):
        patched_identity = hadamard_module.matmul_hadU_cuda == adapter.matmul_hadU_cuda
        unchanged_inside = (
            hadamard_module.get_hadK is original_get_hadK
            and hadamard_module.random_hadamard_matrix is original_random
            and hadamard_module.fast_hadamard_transform is original_fht_module
        )
    restored = (
        hadamard_module.matmul_hadU_cuda is not adapter.matmul_hadU_cuda
        and hadamard_module.get_hadK is original_get_hadK
        and hadamard_module.random_hadamard_matrix is original_random
    )
    return {
        "patched_only_matmul_hadU_cuda": bool(patched_identity),
        "non_fwht_symbols_unchanged_while_patched": bool(unchanged_inside),
        "original_call_site_restored": bool(restored),
        "pass": bool(patched_identity and unchanged_inside and restored),
    }


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, default=DEFAULT_LIBRARY)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--expected-uuid", default=None)
    parser.add_argument(
        "--dao-reference",
        action="store_true",
        help="compare against pinned QuaRot matmul_hadU_cuda using Dao's CUDA extension",
    )
    args = parser.parse_args()
    output = args.output or (DEFAULT_DAO_OUTPUT if args.dao_reference else DEFAULT_OUTPUT)
    reference_mode = "dao" if args.dao_reference else "torch"

    report: dict[str, Any] = {
        "test": (
            "e13_quarot_fwht_adapter_dao_correctness"
            if args.dao_reference
            else "e13_quarot_fwht_adapter_correctness"
        ),
        "source": {
            "repository": "spcl/QuaRot",
            "commit": PINNED_QUAROT_COMMIT,
            "path": str(PINNED_HADAMARD),
            "checkout_head": None,
        },
        "library": str(args.library.resolve()),
        "library_sha256": None,
        "full_model_run": False,
        "application_claim": (
            "none; synthetic FWHT equivalence against pinned Dao CUDA reference"
            if args.dao_reference
            else "none; synthetic FWHT gate only"
        ),
        "reference_mode": reference_mode,
        "cases": [],
        "coverage": {
            "attention_K1_n32": False,
            "llama2_mlp_K172_n11008": False,
            "fp16": False,
            "fp32": False,
            "caller_stream": False,
            "no_zero_padding": False,
            "hadK_multiply_and_full_n_scale": False,
        },
        "dependencies_outstanding": [
            "No authorized Llama-2-7B checkpoint or tokenizer is available for E13.",
            "Full QuaRot integration additionally requires its CUDA extension and pinned Python dependencies.",
            "No full-model quality, application latency, or speedup claim is made by this report.",
        ],
    }

    if not args.library.is_file():
        report["status"] = "blocked_missing_library"
        report["blocker"] = f"shared library not found: {args.library}"
        _write_report(output, report)
        return 2
    report["library_sha256"] = _sha256(args.library)
    if not torch.cuda.is_available():
        report["status"] = "blocked_no_cuda"
        report["blocker"] = "torch.cuda.is_available() is false"
        _write_report(output, report)
        return 2

    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("--device must name a CUDA device")
    torch.cuda.set_device(device)
    uuid = _gpu_uuid()
    report["device"] = {
        "torch_device": str(device),
        "gpu_name": torch.cuda.get_device_name(device),
        "gpu_uuid": uuid,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }
    expected_uuid = args.expected_uuid
    if expected_uuid is not None and uuid != expected_uuid:
        report["status"] = "blocked_wrong_gpu"
        report["blocker"] = f"expected GPU UUID {expected_uuid}, observed {uuid}"
        _write_report(output, report)
        return 2

    try:
        hadamard_module = _load_hadamard_source()
    except (ImportError, ModuleNotFoundError) as error:
        report["status"] = "blocked_missing_dao_reference" if args.dao_reference else "blocked_missing_source_dependency"
        report["blocker"] = str(error)
        _write_report(output, report)
        return 2
    if args.dao_reference:
        dao_package = hadamard_module.fast_hadamard_transform
        dao_cuda = importlib.import_module("fast_hadamard_transform_cuda")
        if not callable(getattr(dao_package, "hadamard_transform", None)):
            raise RuntimeError("installed fast_hadamard_transform package has no hadamard_transform entry point")
        report["reference"] = {
            "implementation": "pinned QuaRot matmul_hadU_cuda",
            "backend": "fast_hadamard_transform.hadamard_transform",
            "package_version": getattr(dao_package, "__version__", None),
            "package_path": str(Path(dao_package.__file__).resolve()),
            "cuda_extension_path": str(Path(dao_cuda.__file__).resolve()),
        }
    hadK, K = hadamard_module.get_hadK(11008)
    if K != 172 or tuple(hadK.shape) != (172, 172):
        raise RuntimeError(f"pinned get_hadK(11008) returned K={K}, shape={tuple(hadK.shape)}")
    report["source"]["checkout_head"] = PINNED_QUAROT_COMMIT
    report["source"]["file_sha256"] = _sha256(PINNED_HADAMARD)

    adapter = QuaRotFWHTAdapter(args.library, device=device)
    custom_stream = torch.cuda.Stream(device=device)
    try:
        cases = [
            _case(
                adapter,
                name="attention_k1_n32_fp32",
                shape=(4, 32),
                dtype=torch.float32,
                K=1,
                hadK=None,
                device=device,
                reference_mode=reference_mode,
                hadamard_module=hadamard_module,
            ),
            _case(
                adapter,
                name="attention_k1_n32_fp16_caller_stream",
                shape=(4, 32),
                dtype=torch.float16,
                K=1,
                hadK=None,
                device=device,
                stream=custom_stream,
                reference_mode=reference_mode,
                hadamard_module=hadamard_module,
            ),
            _case(
                adapter,
                name="mlp_k172_n11008_fp32",
                shape=(2, 11008),
                dtype=torch.float32,
                K=172,
                hadK=hadK,
                device=device,
                reference_mode=reference_mode,
                hadamard_module=hadamard_module,
            ),
            _case(
                adapter,
                name="mlp_k172_n11008_fp16",
                shape=(1, 11008),
                dtype=torch.float16,
                K=172,
                hadK=hadK,
                device=device,
                reference_mode=reference_mode,
                hadamard_module=hadamard_module,
            ),
        ]
        report["cases"] = cases
        report["coverage"].update(
            {
                "attention_K1_n32": all(case["pass"] for case in cases[:2]),
                "llama2_mlp_K172_n11008": all(case["pass"] for case in cases[2:]),
                "fp16": all(case["pass"] for case in (cases[1], cases[3])),
                "fp32": all(case["pass"] for case in (cases[0], cases[2])),
                "caller_stream": cases[1]["stream"] == int(custom_stream.cuda_stream),
                "hadK_multiply_and_full_n_scale": all(case["pass"] for case in cases[2:]),
            }
        )

        try:
            adapter.matmul_hadU_cuda(torch.zeros((1, 48), device=device), None, 1)
        except ValueError as error:
            report["coverage"]["no_zero_padding"] = "no padding" in str(error)
        else:
            report["coverage"]["no_zero_padding"] = False
        report["integration_hook"] = _check_patch_preserves_non_fwht(hadamard_module, adapter)
    finally:
        torch.cuda.synchronize(device)
        adapter.close()

    report["status"] = "pass" if all(case["pass"] for case in report["cases"]) and report["integration_hook"]["pass"] else "fail"
    report["pass"] = report["status"] == "pass"
    _write_report(output, report)
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
