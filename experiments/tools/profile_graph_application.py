"""GPU kernel timeline diagnostics, separate from uninstrumented application timing."""
from collections import defaultdict
import json
from pathlib import Path


def interval_union(intervals):
    total = 0.0
    end = None
    for start, stop in sorted(intervals):
        if stop < start:
            raise ValueError('negative interval')
        total += max(0.0, stop - max(start, end if end is not None else start))
        end = max(stop, end if end is not None else stop)
    return total


def profile_call(function, output, *, synchronize):
    import torch
    output = Path(output)
    synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                           torch.profiler.ProfilerActivity.CUDA]) as prof:
        result = function()
        synchronize()
    prof.export_chrome_trace(str(output))
    trace = json.loads(output.read_text())
    events = [event for event in trace['traceEvents']
              if event.get('cat') == 'kernel' and event.get('ph') == 'X']
    rows = defaultdict(lambda: {'calls': 0, 'sum_us': 0.0})
    for event in events:
        row = rows[event['name']]
        row['calls'] += 1
        row['sum_us'] += event['dur']
    intervals = [(event['ts'], event['ts'] + event['dur']) for event in events]
    return dict(trace_path=output.name, kernel_count=len(events),
                kernels=[dict(name=name, **row) for name, row in sorted(
                    rows.items(), key=lambda item: -item[1]['sum_us'])],
                kernel_busy_union_us=interval_union(intervals),
                kernel_sum_us=sum(event['dur'] for event in events),
                profiled=True, replaces_headline_timing=False,
                interpretation='Kernel durations only; do not include host submission or memcpy. '
                               'Trace is instrumented; use ordinary graph timings for speedup.')
