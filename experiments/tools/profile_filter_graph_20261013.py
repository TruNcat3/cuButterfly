#!/usr/bin/env python3
"""Run the existing filter comparison and append one graph trace per backend."""
import importlib.util
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments/tools'))
from profile_graph_application import profile_call


def main():
    path = ROOT / 'experiments/workloads/e13/run_fft_filter_end_to_end.py'
    spec = importlib.util.spec_from_file_location('filter_graph_profile', path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    measure = runner.measure
    index = 0
    def measured(call, torch, *args, **kwargs):
        nonlocal index
        timing, output = measure(call, torch, *args, **kwargs)
        directory = Path(os.environ['XD_REPORT_DIR'])
        timing['gpu_trace'] = profile_call(call, directory / f'backend{index}.trace.json',
                                         synchronize=torch.cuda.synchronize)
        index += 1
        return timing, output
    runner.measure = measured
    runner.main()


if __name__ == '__main__':
    main()
