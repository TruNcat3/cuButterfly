# v0.7 Current Comprehensive Performance Comparison

Date: 2026-08-18. Device: Tesla V100-SXM2-16GB (`sm_70`).

## Executive Result

The project has two different performance conclusions that must not be mixed:

1. The release/search paths remain competitive with the installed or archived
   high-performance libraries on the representative matrix: 1.019x cuFFT,
   1.052x Dao FHT, and 1.314x archived GPU-NTT geomean throughput. Those NTT
   comparisons cover uint64 `logN=16`, not the new streaming kernel.
2. The long-transform v0.7 architecture is not yet a universal replacement for
   v0.6. Generated resident `10+10` reaches v0.6 at sufficiently large batch;
   resident `7+7+6` proves a single-transform wave but currently loses badly
   because its physical scheduler is not work-conserving.

The current release decision is therefore correct: retain mature v0.6/static
selection as the fallback, enable `10+10` only at measured crossover points,
and keep `7+7+6` as an explicit research candidate.

## Matched Batch Scaling

CUDA-event values are medians of three independent processes, each using 10
warmups and 50 timed iterations. Exact forward correctness of both generated
word-width kernels is checked before the timing loop.

### uint32 speedup versus v0.6 barrier

| batch | resident 10+10 | generic 7+7+6 | resident 7+7+6 |
|---:|---:|---:|---:|
| 1 | 0.620x | 0.216x | 0.187x |
| 4 | 0.782x | 0.181x | 0.103x |
| 8 | 0.918x | 0.135x | 0.084x |
| 16 | 1.037x | 0.103x | 0.078x |
| 32 | 1.190x | 0.112x | 0.096x |
| geomean | 0.887x | 0.143x | 0.104x |

### uint64 speedup versus v0.6 barrier

| batch | resident 10+10 | generic 7+7+6 | resident 7+7+6 |
|---:|---:|---:|---:|
| 1 | 0.699x | 0.283x | 0.221x |
| 4 | 0.907x | 0.341x | 0.150x |
| 8 | 0.991x | 0.336x | 0.126x |
| 16 | 1.030x | 0.283x | 0.114x |
| 32 | 1.024x | 0.153x | 0.104x |
| geomean | 0.921x | 0.269x | 0.138x |

`10+10` exhibits the expected pipeline fill/drain behavior: its relative
throughput rises with batch and crosses v0.6 around batch 16. The `7+7+6`
implementation does the opposite. This does not invalidate the architectural
expectation that more homogeneous subgraphs should improve average pipeline
efficiency. It demonstrates that the current kernel does not realize that
expectation: fixed CTA roles cannot consume all runnable work, and two global
boundaries amplify synchronization traffic as more transforms enter flight.

Full event times and trial spread are in
[`comprehensive/analysis.md`](comprehensive/analysis.md).

## NCU Attribution

The matched uint64 `logN=20,batch=4` capture isolates the difference:

| implementation | time us | speedup vs barrier | read ratio | write ratio | warp-inst ratio | integer-inst ratio | barrier stall |
|:--|---:|---:|---:|---:|---:|---:|---:|
| v0.6 barrier | 536.67 | 1.000x | 1.000x | 1.000x | 1.000x | 1.000x | 6.07% |
| resident 10+10 | 609.15 | 0.881x | 0.995x | 1.001x | 1.096x | 1.017x | 3.53% |
| generic 7+7+6 | 1606.24 | 0.334x | 3.181x | 1.736x | 2.182x | 1.534x | 0.04% |
| resident 7+7+6 | 3490.11 | 0.154x | 6.282x | 4.150x | 1.218x | 1.131x | 37.02% |

The resident three-layer physical core removes much of the generic core's
instruction overhead: warp instructions fall from 2.182x to 1.218x of v0.6,
and integer instructions from 1.534x to 1.131x. Performance nevertheless gets
worse because barrier stall reaches 37.02% and actual DRAM read reaches 6.282x.
Thus the next optimization target is orchestration and boundary ownership, not
another modular butterfly core.

For resident uint64 `7+7+6`, batch 4 takes 5.077x the batch-1 time, so
per-transform time is 1.269x worse. Reads/writes grow by 5.541x/6.491x while
warp/integer instructions grow by only 3.829x/3.852x. The excess traffic is
consistent with repeated readiness observation and synchronization-stalled
roles, rather than useful arithmetic.

The generated NCU table is
[`v07_resident_776_ncu_analysis.md`](v07_resident_776_ncu_analysis.md); raw
reports remain in [`../ncu_hierarchical_streaming_776/`](../ncu_hierarchical_streaming_776/).

## Position Relative To External Libraries

| operator/baseline | representative shapes | searched/library geomean | status |
|:--|---:|---:|:--|
| FP32 FFT / cuFFT | 4 | 1.019x | parity overall |
| FP32 FWHT / Dao FHT | 3 | 1.052x | parity to faster |
| uint64 NTT / archived GPU-NTT | 3 | 1.314x | faster on tested `logN=16` rows |

These rows describe the best selected library paths, not resident `7+7+6`.
There is no matched external `logN=20` row for the new kernel, so the present
data cannot support an external-library superiority claim for v0.7 streaming.
The detailed external protocol is
[`../v100_three_way_v07_r4_final.md`](../v100_three_way_v07_r4_final.md).

## Next Success Criteria

The next `7+7+6` scheduler should be accepted only if all of the following are
observed under the same protocol:

1. batch-4 traffic is close to four times batch-1 traffic, rather than
   5.5--6.5x;
2. barrier stall is below 10% without moving the cost to a contended atomic
   queue;
3. per-transform time improves as batch grows;
4. the resident core first beats generic `7+7+6`, then crosses v0.6;
5. exact forward/inverse correctness and the segment-0/1 overlap trace remain.

The implementation direction is distributed wave ownership: a CTA that owns a
dependency region must always have useful local work or be able to claim from a
small region-local queue. The second boundary also needs a more local closure
so segment 2 can start before segment 1 drains. A single global task head has
already been rejected because its atomic contention serializes the graph.
