# FFT lag-point trace analysis on A100

This report separates steady-state kernel loss from first-call setup and from
the framework's scheduling work. The measurements use the existing Nsight
Systems traces and the paired CUDA-event sweep in
[`results/fft_trace_20261022`](../results/fft_trace_20261022).

## Rank-one `N=65536, B=16`

The no-overlap online mapping launches one FP32 cuFFTDx prefix and one suffix
kernel. The trace reports 18.432 us and 16.832 us, respectively, for a total
of 35.264 us. The paired sweep reports 0.036727 ms for this mapping and
0.023689 ms for cuFFT. The custom path therefore reaches about 64.5% of cuFFT
throughput at this point, even after removing plan construction from the CUDA
event interval.

This residual is not caused by the stage-pipeline scheduler. Both kernels use
the imported cuFFTDx block core, but the framework still materializes the
prefix result through a global intermediate and launches a separate suffix.
The two-group mapping, boundary addressing, and EPT/thread choices are the
remaining steady-state candidates. They belong to the physical lowering and
search space, not to the butterfly dependency model itself.

The overlap trace launches four prefix and four suffix kernels. Its kernel
instances sum to 71.232 us, while the no-overlap trace has only two launches.
The CUDA runtime counts are:

| mapping | kernel launches | event records | stream waits |
|:--|--:|--:|--:|
| bulk | 2 | 6 | 0 |
| stage overlap, tile 4 | 8 | 16 | 10 |

The overlap implementation is real. Splitting the batch reduces each launch's
parallel work and adds dependency operations; bulk wins this comparison.
These timings alone do not establish SM or bandwidth saturation. In particular,
the newer register-prefix/grouped-suffix mapping measured below has substantial
unused capacity. The full-batch tile is now available to the search as a control
point and remains subject to measurement.

## Rank-two `512x512, B=16`

The forced axis-tiled trace contains two `xd_axis_kernel` instances on the
same stream with durations 41.888 us and 71.777 us. The axis executor in
`src/xd_real_fft.cu` submits the logical axes sequentially, so the long-rank
path does not materialize cross-axis stage overlap. The prior paired sweep was
0.11827 ms for this axis path versus 0.09769 ms for the complete cuFFT plan;
the composite fallback was about 0.32614 ms.

This is a framework/lowering limitation in the current rank-two path. It is
independent of the rank-one butterfly theory: the axis plans expose separate
global boundaries and `execute_c2c_axes` waits for one axis before submitting
the next. A future xD candidate should fuse compatible axes or expose their
independent tiles to a dependency-preserving pipeline and measure the complete
boundary cost. Selecting the already available complete-cuFFT path is a
fallback, not evidence that cuButterfly's own organization has improved.

## First-call versus steady state

The overlap trace also contains approximately 175 ms in
`cudaStreamCreateWithFlags` and approximately 57 ms in the first
`cudaLaunchKernel`. These are plan/module/JIT setup effects. They are outside
the benchmark's CUDA-event kernel interval and must be removed by installation
precompilation or a warm-up before reporting steady-state operator latency.
They are nevertheless real end-to-end costs when a user invokes an unseen
mapping for the first time, so the installation workflow must keep the
cuFFTDx provider enabled and precompile confirmed mappings.

## Diagnosis

The trace supports three separate conclusions:

1. The mixed-dataflow schedule is implemented and executable. Its current
   regression at this long FFT is a scheduling/occupancy tradeoff that the
   calibrated selector should reject.
2. Rank-one bulk loss remains a physical core/boundary issue: two global-stage
   launches and their EPT/thread mappings do not match cuFFT's complete-plan
   implementation. This is an engineering/search gap, not evidence against
   the hierarchical hybrid-dataflow formulation.
3. Long rank-two loss is a lowering issue because axes are serialized. It
   cannot be repaired by changing the rank-one tile size alone; the xD design
   space needs a fused-axis candidate or a measured complete-plan fallback.

These observations identify engineering work; they do not prove global
optimality or exclude all deficiencies in the model. The work items are to
precompile confirmed providers, measure rank-one boundary/core alternatives,
and extend xD lowering where sequential axis traffic dominates.

## Scheduler repair and controlled remeasurement

The follow-up uses A100 80GB PCIe UUID
`GPU-0eae5f07-33de-321b-70a9-ffeae49309e6`, FP32 forward C2C, `N=65536`,
`B=16`, contiguous out-of-place buffers and research compilation. Unlike the
earlier cuFFTDx-prefix experiment above, this mapping has a native register
prefix and grouped cuFFTDx suffix, partition `6+10`, EPT `8/8`, and CTA
threads `32/512`. Do not attribute differences between these two experiments
solely to the scheduler.

The common scheduler now executes a single tile directly on the caller stream,
removes redundant initial waits on later group streams, and omits consumed-slot
events when no subsequent tile can reuse the slot. Ordinary calls still fence
cross-stream plan reuse and join all work to the caller. Destructor completion
also covers work submitted directly on the caller.

Graph capture previously failed by waiting on the ordinary invocation's event
recorded outside capture. The repair captures the same producer/consumer DAG,
using a separate graph-local join event. It does not serialize multi-tile work
to bypass the failure. Graph users must serialize shared plan/workspace use and
keep resources alive through replay; see the
[API contract](c_api.md#cuda-graph-and-shared-workspace).

The ordinary comparison uses three randomized rounds, 100 repeats per mode,
and CUDA-event timing after construction and warmup. Every before/after
mapping is identical. The Graph column uses three randomized rounds of 100
replays, with 20 complete executions in each graph. Compilation, allocation,
CPU reference checks and graph construction are outside timing.

| Schedule | Ordinary before, us | Ordinary after, us | Graph after, us |
| --- | ---: | ---: | ---: |
| cuFFT reference | 23.572 | 23.593 | 21.283 |
| Bulk | 30.607 | 30.628 | 28.286 |
| Overlap, tile 1 | 204.127 | 205.465 | 114.534 |
| Overlap, tile 4 | 66.283 | 67.271 | 38.085 |
| Overlap, tile 8 | 47.802 | 47.714 | 34.829 |
| Single tile, tile 16 | 35.637 | 30.474 | 28.305 |

The single-tile ordinary latency falls by 14.5%, reaching the bulk path.
Changes in the other ordinary modes are small and are not claimed as gains.
Graph and ordinary timings are separate protocols, not a before/after speedup.
The best custom Graph schedule here remains about 0.752x cuFFT throughput.

Artifacts are the `verification.json`, `raw.csv`, `metadata.json` and
`summary.json` files in
[`fft_schedule_before_20261022`](../results/fft_schedule_before_20261022) and
[`fft_schedule_after_20261022`](../results/fft_schedule_after_20261022).
[`graph_final.json`](../results/fft_schedule_after_20261022/graph_final.json)
records the final build's Graph confirmation. Earlier `graph.json` and the
trace retain their original binary identity; adding the CTest registration
triggered a fingerprint rebuild, so final confirmation is recorded separately.

## Actual Graph overlap and remaining resource limits

The exported DOT retains all same-tile producer/consumer edges and both ring
reuse edges. Consecutive plan invocations are ordered, while prefix tile 1 and
suffix tile 0 have no dependency between them. The trace analysis filters
`graphNodeId > 0`, excluding 42 ordinary warmup kernels, and associates 1280
captured kernels with eight graph launches, or 160 complete plans.

Every plan has overlapping prefix/suffix intervals whose nodes are independent
in the DOT. Median overlap is 23.168 us, median per-plan kernel span is
37.216 us, and median kernel-duration sum is 60.224 us. This is actual pipeline
execution, not merely two stream labels. Fifteen dependent-node pairs have
127--256 ns apparent interval intersections; these are excluded from the
independent-overlap evidence rather than treated as useful concurrency.
Profiler times do not replace ordinary benchmark results. See
[`graph_trace_analysis.json`](../results/fft_schedule_after_20261022/graph_trace_analysis.json).

The bulk mapping's NCU records revise the earlier saturation hypothesis:

| Metric | Register prefix | Grouped suffix |
| --- | ---: | ---: |
| CTA count | 4096 | 256 |
| Threads per CTA | 32 | 512 |
| Registers per thread | 60 | 32 |
| Shared bytes per CTA | 2048 | 32768 |
| NCU waves per SM | 1.19 | 0.59 |
| Theoretical occupancy | 50% | 100% |
| Achieved occupancy | 30.36% | 51.27% |
| SM throughput | 54.41% | 32.29% |
| DRAM throughput | 26.79% | 18.23% |

The one-warp prefix CTA reaches the block-count limit before filling the warp
capacity. The suffix has less than one full resident wave. Tile 4 reduces its
grid further to 64 CTAs on a 108-SM GPU. Thus neither whole-batch bandwidth
saturation nor lack of implemented overlap explains the entire loss.
These are isolated-kernel NCU measurements with replay; they do not quantify
resource contention during simultaneous stages. Raw reports and CSV exports
are `prefix_ncu.*` and `bulk_ncu.*` in the after-results directory; the latter
contains the suffix capture only.

## Bounded CTA organization experiment

A `3 x 3` experiment keeps the core implementations, EPT, `6+10` decomposition,
precision, input and boundary policy fixed. It varies prefix CTA threads
`32/64/128` and suffix CTA threads `128/256/512`. The corresponding number of
local transforms per CTA changes with these thread counts; this is a linked
organization change, not an arithmetic-core replacement. All nine points pass
full-batch verification and produce Graph outputs identical to ordinary calls.

| Prefix CTA threads | Suffix 128, us | Suffix 256, us | Suffix 512, us |
| --- | ---: | ---: | ---: |
| 32 | 42.694 | 30.603 | 28.125 |
| 64 | 42.271 | 30.168 | 27.822 |
| 128 | 42.210 | 29.866 | 27.267 |

An independent five-round, 200-replay finalist confirmation measures
28.280 us for `32/512`, 27.091 us for `128/512`, and 21.284 us for cuFFT.
The candidate reduces latency by 4.2% relative to the original mapping,
reaching 0.786x cuFFT throughput on this cell. All custom candidates report
relative L2 error about `3.01e-7` against CPU FP32, similar to cuFFT; no new
floating acceptance threshold is introduced.

Shrinking the suffix CTA to expose more blocks makes it slower. Increasing
occupancy alone is therefore not an optimization rule: local transform
grouping, global access and boundary exchange must be evaluated together.
The measured candidate is retained in
[`cta_preflight.json`](../results/fft_schedule_after_20261022/cta_preflight.json),
[`cta_graph.json`](../results/fft_schedule_after_20261022/cta_graph.json) and
[`cta_finalists.json`](../results/fft_schedule_after_20261022/cta_finalists.json).
This is a controlled local search, not an installed selector result or a
full-matrix speedup claim. It does not resolve the remaining cuFFT gap or xD
cross-axis lowering.

## Regression coverage and reproduction

CPU dependency and candidate-projection checks pass 76 tests. GPU checks pass
80 ordinary shared-mechanism cases plus eight negative contracts, the FFT
stage-overlap async suite, and 88 Graph cases covering FFT, NTT, FWHT,
subset/superset zeta and structured butterflies. Graph cases include both
directions, single/multiple tiles, tails, FP32/FP64 and 32/64-bit integer
paths, changed-input replay, two calls per capture, and return to ordinary
execution. `shared_graph_mechanism_correctness` is registered with CTest.

```sh
CUDA_VISIBLE_DEVICES=GPU-0eae5f07-33de-321b-70a9-ffeae49309e6 \
  ctest --test-dir build-e13-cufftdx --output-on-failure \
  -R '^shared_graph_mechanism_correctness$'

CUDA_VISIBLE_DEVICES=GPU-0eae5f07-33de-321b-70a9-ffeae49309e6 \
CUBUTTERFLY_COMPILE_MODE=research \
  /home/wt/yes/envs/cubutterfly/bin/python scripts/benchmark_pipeline_graph.py \
  --library build-e13-cufftdx/libcuntt.so \
  --verification results/fft_schedule_after_20261022/cta_preflight.json \
  --output /tmp/cubutterfly-cta-confirmation.json \
  --modes cufft cta-32-512 cta-128-512 --rounds 5 --repeat 200
```
