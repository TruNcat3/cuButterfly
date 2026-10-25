"""Bounded diagnostic for the pinned private extension's CUDA stream launches."""
import ctypes
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/e13_assets/sources/QuaRot"
sys.path.insert(0, str(SOURCE))
from quarot import _CUDA


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quarot-extension", type=Path)
    args = parser.parse_args()
    extension = _CUDA
    if args.quarot_extension:
        specification = importlib.util.spec_from_file_location("_CUDA_graph", args.quarot_extension)
        extension = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(extension)
    runtime = ctypes.CDLL("libcudart.so.12")
    runtime.cudaGetLastError.restype = ctypes.c_int
    runtime.cudaGetErrorString.restype = ctypes.c_char_p
    runtime.cudaGetErrorString.argtypes = [ctypes.c_int]
    x = torch.ones((4, 4096), device="cuda", dtype=torch.float16)
    scale = torch.ones((4, 1), device="cuda", dtype=torch.float16)
    reference = _CUDA.sym_quant(x, scale)
    torch.cuda.synchronize()
    report = {
        "operation": "quarot._CUDA.sym_quant",
        "extension_path": extension.__file__,
        "extension_sha256": hashlib.sha256(Path(extension.__file__).read_bytes()).hexdigest(),
        "torch": torch.__version__,
        "gpu_uuid": str(torch.cuda.get_device_properties(0).uuid),
        "original_source_launch": "quarot/kernels/quant.cu:60 sym_quantize_f16_i4_kernel<<<grid, block>>>",
        "active_launch_stream": "pytorch_current_stream" if args.quarot_extension else "legacy_default_stream",
        "extension_protocol": "current_stream_only_v1" if args.quarot_extension else "pinned_original",
    }
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        graph.capture_begin()
        q = extension.sym_quant(x, scale)
        status = runtime.cudaGetLastError()
        report["cuda_last_error_after_private_launch"] = status
        report["cuda_last_error_string"] = runtime.cudaGetErrorString(status).decode()
        try:
            graph.capture_end()
            report["capture_end"] = "pass"
        except Exception as exc:
            report["capture_end"] = str(exc)
    if report["capture_end"] == "pass":
        try:
            q.zero_()
            graph.replay()
            torch.cuda.synchronize()
            report["nonzero_count_after_replay"] = int(torch.count_nonzero(q))
            report["expected_nonzero_count"] = q.numel()
            report["bitwise_equal_to_original_eager"] = torch.equal(q, reference)
        except Exception as exc:
            report["replay_error"] = str(exc)
    report["status"] = "diagnostic_complete"
    output = Path(os.environ["XD_REPORT_DIR"]) / "private_graph_stream.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if args.quarot_extension and not report.get("bitwise_equal_to_original_eager", False):
        raise RuntimeError("Current-stream private graph replay differs from original eager output")


if __name__ == "__main__":
    main()
