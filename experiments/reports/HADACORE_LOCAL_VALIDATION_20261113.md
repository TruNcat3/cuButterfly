# HadaCore local-core validation and FWHT diagnosis

The optional FP16 MMA tile provider completed 204 geometry validations on the
A100 80GB PCIe (SM80, UUID `GPU-0eae5f07-33de-321b-70a9-ffeae49309e6`).
Its output matched the pinned complete HadaCore implementation bitwise in every
validated trial. With one group per block, the local provider is 1.508× faster
in geometric mean across 68 supported requests, with 68/68 nominal wins and a
range of 1.117–1.996×. This is a local-provider comparison, not an automatic
cuButterfly-library result.

## Population and validation

The frozen population contains N=2¹…2¹⁵ and B∈{1,2,4,16,64,256}: 90 requests.
68 admit complete packed groups; the other 22 are explicitly unsupported by this
provider because their complete input cannot fill its minimum vectorized group.
They remain in the summary, without padding, precision substitution or batch
reduction. The 68 legal requests each cover groups-per-block 1, 2 and 4.
The full run reused the 27 validated quick-population geometry records; their
original evidence remains traceable rather than being relabelled as fresh runs.

All 204 records retain before/after exclusivity guards, repeated-output and graph
replay checks, immutable-input and buffer-guard checks, same-contract upstream
comparisons, resource metadata and hashes. All outputs are finite and exactly
match upstream. CPU per-stage FP16 radix-2 references are retained as numerical
diagnostics with full error distributions and lossless artifacts; no new floating
error threshold was imposed.

The arithmetic contract is `fp16-mma-orthonormal-final-scale`: FP16 MMA with
normalized subtransform coefficients, followed by FP32 output scaling and FP16
writeback. It differs from native per-stage FP16 rounding. Exact upstream matches
establish extraction/fusion correctness for the measured inputs, not universal
bitwise equivalence to every FWHT implementation.

## Paired complete-call performance

Each trial uses 10 warmups, 20 complete calls per CUDA Graph and the median of
seven replays. Compilation, allocation, plan setup and references are excluded
from both backends. The pinned upstream comparator includes its required separate
scaling kernel. The tile provider fuses that same scaling into final stores.
The resulting reduction in calls and intermediate memory traffic is an intended
organization/fusion benefit, not evidence that its MMA arithmetic is intrinsically
faster. Further parent-stage composition is outside this measured comparison.

| N | B | Local group=1, µs | Upstream, µs | Upstream/local |
| ---: | ---: | ---: | ---: | ---: |
| 256 | 1 | 2.355 | 3.502 | 1.487× |
| 256 | 16 | 2.560 | 4.162 | 1.626× |
| 256 | 256 | 2.918 | 4.595 | 1.575× |
| 4096 | 1 | 3.277 | 4.686 | 1.430× |
| 4096 | 16 | 3.482 | 5.059 | 1.453× |
| 4096 | 256 | 4.915 | 9.627 | 1.959× |
| 32768 | 1 | 12.698 | 14.222 | 1.120× |
| 32768 | 16 | 12.902 | 15.864 | 1.230× |
| 32768 | 256 | 42.035 | 69.890 | 1.663× |

The fixed group=1 statistic uses one predeclared geometry per request. The
post-hoc best local latency over groups 1/2/4 also selected group=1 for all 68
requests, so its paired summary is identical. It remains an oracle statistic,
not evidence of independent AUTO selection or a speedup from temporal group reuse.
Increasing sequential groups can reduce exposed blocks and is not universally
beneficial.

The measured design is one stage group covering log₂N butterfly levels, tile
length N, packed natural input/output, FP16 MMA arithmetic, one shared scratch
group and no double-buffer prefetch. Threads follow the compiled launch geometry:
32 for N256, 256 for N4096, and 512 for N32768; dynamic scratch is respectively
512, 8192 and 65536 bytes. Groups-per-block changes sequential scratch reuse,
not the scratch allocation. Across all 204 entries, measured registers are
26–128/thread and local-memory allocation is 0–8 bytes/thread. Local bytes are
resource allocation data and must not be called measured spill traffic.

## Existing scalar-path profile evidence

Four NCU reports compare the old native scalar/shared radix-4 replay with the
external complete HadaCore path at B256. The table reports arithmetic kernels;
the separate HadaCore scaling kernels remain in the exported profile CSV.

| N | Core | Profiled duration, µs | Registers/thread | Active warps, % peak | Long-scoreboard stall, % |
| ---: | --- | ---: | ---: | ---: | ---: |
| 4096 | Native scalar radix-4 | 30.304 | 16 | 14.64 | 55.78 |
| 4096 | External HadaCore MMA | 5.664 | 39 | 26.67 | 36.45 |
| 32768 | Native scalar radix-4 | 393.664 | 16 | 12.14 | 55.85 |
| 32768 | External HadaCore MMA | 40.416 | 127 | 23.94 | 16.20 |

These profiles support a concrete weak-path diagnosis: the old selection used
128-thread scalar/shared tiles, with low achieved warp occupancy and substantial
scoreboard waits. The MMA comparison changes arithmetic as well as execution,
so it is not a fixed-core causal ablation. Neither these counters nor profiler
kernel durations prove that the full hierarchical method is flawed or optimal.
Profiler durations are not mixed with ordinary timing medians or used in the
68-point speedup aggregate. Shared-bank conflicts remain in the CSV for follow-up;
raw conflict counts alone do not establish their share of execution time.

Both NSys reports are now available. The old 24-job diagnostic queue has 17
complete jobs; its seven unfinished items comprise the rejected 512-thread
configuration and six unmeasured thread variants. These are retained separately
from the completed local-provider matrix. A valid 512-thread MMA provider does
not make a 512-thread scalar template legal.

## Reproduction and evidence boundary

Run the CPU-only summarizer from the repository root:

```bash
python experiments/tools/summarize_hadacore_local_validation.py \
  --input /home/wt/git/cuButterfly/results/hadacore_tile_validation_full_20261113 \
  --trace /home/wt/git/cuButterfly/results/fwht_trace_20261111/selected \
  --output /home/wt/git/cuButterfly/results/hadacore_tile_validation_full_20261113/summary
```

It verifies evidence hashes, exact-match/guard status, GPU and binary identity,
complete geometry populations and matching timing protocols before aggregating.
It imports existing NCU reports without using the GPU. Outputs are `paired.csv`,
`resources.csv`, `ncu.csv` and `summary.json`; raw evidence is unchanged.
The summary records full source identities, including provider binary
`c4d4587f716788e03f85c15c9927c9046436b1235e654dd65c4ed9d2eaaed48a`.

Public selection integration is a separate acceptance step. These local results
do not certify C++ AUTO, native FP16 equivalence, inverse/strided/multiaxis
coverage, arbitrary hardware, or every compiled chunk variant. Public-path
verification and any final selection results should be appended after their own
matching measurements.

## Public-path checkpoint

The C, C++ and Python interfaces now expose an explicit
`ALLOW_FP16_MMA` arithmetic policy. STRICT retains native stage rounding. AUTO
compares matched measured native endpoints with validated local MMA candidates;
missing paired calibration retains native selection with a recorded reason.
Explicit replay checks target, binary, profile, geometry, numerical evidence
and actual compiled resources. Online `MEASURE` with this policy is explicitly
unsupported until its verifier can consume the MMA arithmetic contract.

The frozen main library for this step is
`build/hadacore_public_20261113/libcuntt.so` in the runtime checkout, SHA256
`15772cf943a69228eff977093860bdde4bbadf1e678421e86ddf498d5d0365bd`.
Public AUTO smoke tests for N2/B256 and N256/B1 both selected the measured MMA
geometry and matched its output bitwise. Graph, input preservation and buffer
guards passed. These two tests are not a substitute for the full public matrix.

Native controls for all 68 legal requests are complete, and the paired profile
contains no pending rows. The public AUTO validation also completed all 68
requests with `validated-public-selection`. Every request selected the measured
MMA local core under `ALLOW_FP16_MMA`; no request used the native fallback in
this paired cohort. Public outputs matched the selected calibration evidence
bitwise, with guards, input preservation and Graph replay passing.

The native checkpoint is
`results/hadacore_native_validation_20261113/native-checkpoint.json`; the paired
profile is `results/hadacore_tile_validation_full_20261113/profile-paired.json`;
the public checkpoint is
`results/hadacore_public_validation_20261113/completion.json` in the runtime
checkout. `scripts/complete_hadacore_calibration.py` provides the resumable
ordered workflow. The public timing medians were within 0.98–1.05× of the
selected calibration medians, so the selection protocol was reproduced rather
than merely checked for output equality.

Preparation exposed a separate CPU bottleneck: one N512/B64 native projected
ranking request contained about 285,383 candidates and serialized to roughly
1.61 GB. Full candidate serialization, group decoding and scoring dominate that
plan's preparation; this time is excluded from kernel timing. The run retains
the full space. Exact scoring-input reuse and shared request metadata are
follow-up engineering work, not a reason to silently lower the search budget.

The related Python checks passed 94 tests with one skip; the main-library build,
CPU semantic checks and evidence-tampering checks also passed.
