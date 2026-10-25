# NTT historical comparison audit (2026-10-17)

The current five valid GPU-NTT pairs have a geometric mean of **1.084784x**.
The historical RC2 **1.472341x** is a different 214-pair population, not the
previous measurement of these five requests. The closest historical cohort has
four matching semantic requests: their historical GM is 1.195170x and their
current GM is 1.165639x. Different physical GPUs and builds prevent attributing
that difference to a code regression. This audit is read-only apart from this
report and [its numerical ledger](ntt_comparison_history_20261017.json); no GPU
work was performed.

## Historical number provenance

| Evidence | Population | GPU-NTT ratio |
| --- | --- | ---: |
| [RC2 frozen summary](../../cuButterfly-v0.9.1-rc2/results/integrated_repairs_20261001_replay_fix1/a100-80gb/ntt/summary.json), `groups/all-complete` | 214 completed pairs, 209 wins | GM 1.472341x |
| Same summary, `groups/word64` | 144 pairs | GM 1.333543x |
| Same summary, `groups/word32` | 70 pairs | GM 1.804965x |
| Same summary, `groups/inverse` | 107 pairs | GM 1.571609x |
| Same summary, `groups/coset` | 30 pairs | GM 1.656135x |
| [2026-10-12 ledger](../paper/evidence/EXPERIMENT_REUSE_20261012.json), last `operator_groups` entry | 12 word64 pairs, three processes | GM 1.285470x; maximum 1.614771x |
| [V100 three-way raw summary](../results/v100_three_way_comparison.csv), three NTT rows | N=65536, B=4/64/256, forward, natural | 1.117641/1.409846/1.437809x |

[RC2 status](../paper/RC2_STATUS_20261002.md) explicitly reports 1.472x.
The 1.614771x October 12 maximum is N=4096/B256 normalized inverse, not an
aggregate. RC2 also contains groups near 1.6x, but the user's intended source
of “1.6x” cannot be established from that phrase alone.

The similarly named local [September 29 summary](../results/crypto_external_20260929/summary.json)
has GM 1.354585x over 214 pairs. It is not the RC2 1.472341x artifact.

## Contract and timing differences

RC2's [identity](../../cuButterfly-v0.9.1-rc2/results/integrated_repairs_20261001_replay_fix1/a100-80gb/ntt/identity.json)
records A100 80GB PCIe UUID `GPU-2971e500-d3f6-d7e6-e5f2-cb67a06fff02`,
cuButterfly binary SHA256 beginning `e0dfa6a3`, research compile policy and fused
boundary mode. GPU-NTT Merge is pinned to
`d03c5eaeadaa780d153496afcb3b6a9b79a13a63`. Its completed
[journal cells](../../cuButterfly-v0.9.1-rc2/results/integrated_repairs_20261001_replay_fix1/a100-80gb/ntt/journal.json)
cover N={256,4096,65536,262144,1048576}, B={1,3,7,16,256}, both directions,
cyclic/coset/negacyclic modes and one/two modulus channels. Completed moduli are
8380417 (23 bits), 1073479681 (30), 550829555713 (40), 568447511560193 (50),
and 582090251837636609 (60). None equals the current modulus
576460756061519873 (60 bits). Therefore no RC2 cell fully matches a current
request. Even the RC2 cyclic word64 subset (GM 1.363718x) is only context.

RC2 uses natural input/output and exact modular validation, three trials with
100 warmups and 100 CUDA-event repetitions per input pattern. Setup, compilation
and CPU checking are excluded; permutations, twists and normalization are
included. Per-cell speedup is median random-pattern GPU-NTT time divided by
median random-pattern cuButterfly time; boundary-pattern trials validate edge
cases and are not duplicate observations. The GM is `exp(mean(log(ratio)))`
over completed pairs. The 78 unavailable and 54 baseline-correctness-blocked
contracts remain excluded, not counted as wins. RC2 supplies frozen mapping
proposals, so it also differs from the current public selection experiment.

Current [measurement identity](../results/install_compare_repair_20261014/measurement/results.json)
uses the same physical GPU as RC2, build SHA256 beginning `cc78cfad`, runner
`0.9.3.4-async-experiments.4`, and CUDA 12.4. All five pairs are packed,
out-of-place, rank-one uint64, modulus 576460756061519873, natural input/output;
inverse rows include inverse normalization. GPU-NTT uses the same pinned revision
through semantic bridge SHA256 beginning `a61e3fe7` (`sm80 + compute80 PTX`).
The complete call includes required output bit reversal for forward and required
order/normalization operations for inverse. It excludes setup and planning.
Each process uses 10 warmups, 20 complete calls per CUDA Graph, seven event-timed
replays divided by 20, then the sample median. Thus RC2's ordinary event loop
and current Graph timing are different protocols.

The [V100 report](v100_external_baselines.md) instead uses Tesla V100-SXM2-16GB,
CUDA 11.8/sm70, 1000 warmups, 100 repetitions and five process trials. Its
60-bit modulus matches the current one and natural output includes permutation,
but hardware, build, batch population and timing differ. Its archived library
measurements were not interleaved with the three-way local refresh. A 1.438x
V100 point is not a current A100 aggregate.

## Current five pairs and closest overlap

Current times below are milliseconds from the [unified CSV](../results/install_compare_20261013/cpu_external_v2/unified_comparison.csv).
Index is the zero-based `rows` index in the current measurement `results.json`;
each entry links to its detailed `rows/<id>.json` artifact.

| Index | N / B / direction | cuButterfly ms | GPU-NTT ms | Ratio |
| ---: | --- | ---: | ---: | ---: |
| 27 | 4096 / 1 / forward | 0.007731200 | 0.009420800 | 1.218543x |
| 26 | 4096 / 1 / inverse | 0.007833600 | 0.010547200 | 1.346405x |
| 28 | 65536 / 1 / forward | 0.014438400 | 0.014284800 | 0.989362x |
| 29 | 65536 / 1 / inverse | 0.014540800 | 0.016537599 | 1.137324x |
| 61 | 65536 / 4 / forward | 0.032153600 | 0.026163200 | 0.813694x |

The October 12 [unified report](../results/paper_targeted_20261012/unified_report/summary.json)
contains the same first four case IDs, with the same semantics and Graph
protocol. Their three-process median times give:

| N / direction, B=1 | Historical cuB ms | Historical GPU-NTT ms | Historical ratio | Current cuB / historical cuB |
| --- | ---: | ---: | ---: | ---: |
| 4096 / forward | 0.007782400 | 0.009369600 | 1.203947x | 0.993421x |
| 4096 / inverse | 0.007833600 | 0.010700800 | 1.366013x | 1.000000x |
| 65536 / forward | 0.013926400 | 0.014336000 | 1.029412x | 1.036765x |
| 65536 / inverse | 0.013721600 | 0.016537599 | 1.205224x | 1.059702x |

The historical [trial identity](../results/paper_targeted_20261012/ntt_extension/trial-1/results.json)
is A100 80GB PCIe UUID `GPU-0eae5f07-33de-321b-70a9-ffeae49309e6` and cuB
SHA256 beginning `24486975`; the GPU-NTT bridge hash matches current. The four
common requests have the same coarse selected backend/core: hybrid2d/radix4
for N4096, shared-iterative/radix2 for N65536. This does not prove complete
mapping or binary equality. The current B4 point selects hybrid2d/radix4,
`n1_log=6`, 256 threads; it has no October 12 same-request observation.

The B1 intersection GM changes from 1.195170x to 1.165639x. Adding the measured
B4 loss produces the current five-pair GM 1.084784x. This arithmetic explains
the population effect without deleting the loss. The historical 12-pair GM
also includes B16/B256; it must remain a separate cohort. The small observed
B1 latency shifts cannot isolate build, mapping, device or run-state causes.

## Finite remeasurement needs

1. Repeat the existing five current contracts on one exclusively available GPU,
   same build, inputs and Graph protocol, with three independent processes.
   Preserve all five, including N65536/B4 forward.
2. For causal historical/current comparison, replay only the four common B1
   contracts from the October 12 build and current build on that same GPU under
   the same Graph protocol. Compare complete selected mappings before attributing
   a difference to selection or kernels. Availability of the old binary must be
   checked; historical timings cannot substitute for this replay.
3. If reproducing the historical 1.6x point is needed, separately repeat the one
   N4096/B256 normalized-inverse contract with modulus 576460756061519873.
   It remains a historical-witness cohort and does not replace the five-point GM.

Reproducing the 214-point RC2 headline requires its declared matrix and ordinary
event protocol; it is not needed to diagnose these five requests. No evidence
here establishes that today's build regressed from 1.472x on matched requests.

## Same-shape A100 replay (2026-10-19)

The historical 1.4x claim was independently rechecked with a narrower, directly
matched control. On A100 80GB GPU1, the legacy single-modulus GPU-NTT Merge
executable and the current `hybrid2d/radix4` path used modulus
`576460756061519873`, forward natural-order output, and the same N/batch shapes.
The current path passed exact modular validation. The resulting ratios were
1.127x, 1.223x, 1.363x, and 1.392x for N=2^16 with batch 4, 16, 64, and 256;
1.285x and 1.309x for N=2^18 with batch 16 and 64; 0.988x and 1.085x for
N=2^20 with batch 4 and 16. The geometric mean of these eight forward points is
1.214x. The full ledger is
[`gpuntt_legacy_same_shape_a100_20261019.csv`](../results/e2e_rns_workload_20261018/gpuntt_legacy_same_shape_a100_20261019.csv).

This replay confirms the user's remembered 1.4x in its actual population: the
1.36--1.39x points are the medium/large-batch N=2^16 cases. It does not imply a
uniform speedup, and it is separate from the new RNS `GpuNttMerge` compatibility
provider, whose two-modulus A100 controls remain a different contract.
