#!/usr/bin/env python3
import pathlib
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_fp64_fft_robustness import analyze  # noqa: E402


def row(log_n, batch, backend, time):
    return {
        "logN": str(log_n), "batch": str(batch), "backend": backend,
        "kernel_ms": str(time), "local_stages": "7", "prefix_threads": "128",
        "prefix_ept": "4", "suffix_threads": "128", "suffix_ept": "8",
    }


def main():
    rows = [
        row(15, 4, "online-reorder", 2.0), row(15, 4, "cufft", 3.0),
        row(15, 4, "online-reorder", 2.2), row(15, 4, "cufft", 3.2),
    ]
    result = analyze(rows)[0]
    assert result["total_points"] == 131072
    assert result["cuntt_median_ms"] == 2.1
    assert result["cufft_median_ms"] == 3.1
    assert result["timing_quality"] == "stable"
    assert result["range_separation"] == "faster"


if __name__ == "__main__":
    main()
