# Organization argument and evidence boundaries

Revision date: 2026-10-28. This record supports the local manuscript revision;
it is not a new measurement campaign or a declaration of human verification.
No raw results, numerical acceptance rules, or GPU queues were changed.

## Research question

The motivation concerns the organization of parallel execution relative to a
resource roof. It must not reduce all execution gaps to insufficient independent
work. Communication, synchronization, ownership, residency, and core/resource
interaction are possible contributors. Resource utilization percentages alone
do not isolate their causal effects.

The three limiting cases in the motivation are analytical: a throughput plateau
with growing batch, a larger applicable hardware roof without a corresponding
throughput increase, and a local capacity crossing. The cross-target example
conditions on a serialized service remaining unchanged; it is not a measured
A100-to-Blackwell result. Doubling only arithmetic capacity in a bandwidth-bound
case does not imply that the applicable Roofline bound doubles.

## Reused measured contrasts

All following scaling rows are the separate A100 PCIe 40GB M1 cohort. The core
is scalar radix-2, physical groups are 10+10, threads are 256, and overlap is
disabled. Each cell has three processes, each with 10 warmups and 20 repeats.
Central times are medians of process means; ranges are all three means, not CIs.

| Contrast | Median time (ms) | Observed range (ms) | Interpretation |
| --- | ---: | --- | --- |
| FWHT FP64, N=2^20, local0, B=1 | 0.107213 | 0.101939–0.107264 | 9.780306 Gvalues/s |
| Same, B=4 | 0.413389 | 0.413030–0.413491 | 10.146143 Gvalues/s |
| Same, B=16 | 1.479475 | 1.479373–1.549005 | 11.339979 Gvalues/s |
| NTT word64/Shoup, N=2^20, B=16, local2 | 2.007143 | 1.991014–2.629888 | Two-stage local groups |
| Same, local4 | 2.747955 | 2.747392–3.309209 | Up to four-stage local groups |

Throughput is B*N / median_ms / 10^6 in billions of values/s. The FWHT endpoint
ratio is about 1.1595 despite 16 times the work; this does not prove hardware
saturation. The NTT latency ratio is about 1.3691. Its modulus is exactly
576460756061519873. Ownership, synchronization and generated instructions can
change with grouping, so the loss is not uniquely attributed to registers.
The existing full scaling figure retains all workloads and trial ranges.

Sources: [M1 results](NARRATIVE_SCALING_RESULTS_20261027.md),
[M1 profile](NARRATIVE_SCALING_PROFILE_20261027.md), and
[scaling CSV](../generated/narrative_scaling_20261027.csv).
CSV SHA-256: `d3024dac4ccfca549b8e47d57e488a25392dc00a7b33999b0bc2770d1d93eae7`.

## Existing core attribution is sufficient for its stated claim

The FP32 native/cuFFTDx Thread × stagewise-HBM/resident-prefix-2 experiment
contains six requests, four cells per request and three trials per cell.
Organization gains within the two fixed leaves are 1.21702× and 1.22072×;
core replacement within the two organizations is 1.04233× and 1.04550×.
Thus gains in these controls are not solely due to core replacement. Different
threads, storage and launch structures belong to the organization intervention.

A stronger organization comparator asks how much improvement remains after
other organization choices are tuned. It is not needed to re-establish the
already-supported fixed-core attribution. The separate FP64 control already
illustrates this distinction: one-step prefetch gains of 1.730×/1.880× over
four-tile synchronous execution become 1.034×/1.093× against the faster one-tile
synchronous alternative. Neither experiment proves every mechanism wins for
all operators. Existing negative FWHT/NTT crossings remain in the manuscript.

Sources: [FP32 point data](../generated/core_organization_compact_fp32_points.csv),
[source report](../../results/paper_completion_20260917/core_dataflow_ablation/REPORT.md),
[FP64 metadata](../generated/core_organization_compact_metadata.json), and
[FWHT/NTT crossing results](CORE_CROSSING_RESULTS_20261027.md).
FP32 point CSV SHA-256:
`ae2f8bd834689b06628499dcb489a68aceb8aa650e569f7a9a6d388d170f3fce`.

## Analytical construction

For a stage band with s distinct pairing bits, component size is 2^s and
component count is B*N/2^s. Retaining a whole component costs e*2^s value
bytes plus auxiliary storage. G materialized read/write bands transfer
2*e*B*N*G ideal value bytes; actual hardware transactions can differ.

For N=16, partitions [1,1,1,1], [2,2], and [4] require 2, 4, and 16 resident
values per component and 128, 64, and 32 total value transfers. A four-value
budget after auxiliary reservation limits resident bands to two stages.
N=32 then requires at least three such bands, for example [2,2,1]. This is a
capacity/traffic example, not a timing optimum or an additional hardware level.

The draw.io Roofline uses seven measured A100 40GB M1 points at their derived
ideal intensity. F0/F2/F4 are matched FFT FP64 organizations, H1/H16 are FWHT
batch points, and N2/N4 are matched word64 NTT organizations. All use two
materialized stage groups, so equal x coordinates within an operator family
are a controlled organization comparison. The complete medians, trial ranges,
derived intensity, pair rate, and source are frozen in
`paper/generated/roofline_cases_20261029.csv` (SHA-256
`67d103bf7d266f895157f95d9de900fec138a8682a08c0fadfb2166e26c8631d`). The
draw.io source and exports are tracked in the figure manifest. The blue roof
remains schematic and is neither fitted to the observations nor interpreted as
a measured hardware peak.

## Scope retained outside the argument

The preceding manuscript snapshot reported 138 completed default-path finite-field
compositions, a separate tuned comparison and a 90-request cyclic extension.
Those were outside the 66-request headline operator population and remain so.
Likewise, the twelve captures from the six-case organization control require
operator-region attribution before they can support new counter claims.
These collection details stay in campaign/evidence records rather than appearing
as development milestones in the paper.

No new claims are made about calibration time, developer effort, unseen-parameter
generalization, or universal model consumption across lowering paths. The
rank-one registry and multidimensional complete-plan measurement remain distinct
selection paths. The main paper describes their common selection role without
claiming one universal evolutionary algorithm.

A100 40GB and 80GB are both SM80 and remain separate evidence populations.
The PRO 6000 Server 96GB comparison follows A100 completion. Its interpretation
must distinguish absolute throughput, efficiency against a matched applicable
resource roof, a transported organization, and a target-selected organization.
No additional GPU matrix or waiting task was started by this revision.
