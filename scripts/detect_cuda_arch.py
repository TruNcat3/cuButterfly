#!/usr/bin/env python3
"""Choose the first CUDA-visible GPU for a local build; no historical target default."""
import os
import subprocess


def detect():
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible in ("-1", "NoDevFiles"):
        raise RuntimeError("no CUDA device is visible; provide CMAKE_CUDA_ARCHITECTURES for a cross build")
    command = ["nvidia-smi"]
    if visible:
        command.extend(["-i", visible.split(",")[0]])
    command.extend(["--query-gpu=compute_cap", "--format=csv,noheader,nounits"])
    result = subprocess.run(command, text=True, capture_output=True, check=True)
    capability = result.stdout.strip().splitlines()[0].replace(".", "")
    if not capability.isdigit():
        raise RuntimeError("could not identify CUDA architecture")
    return capability


if __name__ == "__main__":
    print(detect())
