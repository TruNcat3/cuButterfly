# Research convergence, 2026-09-19

> Note: Paths marked `local artifact` are local experiment records and are not included in this source release.

The research objective is a broadly effective butterfly mapping framework and
credible matched-library evidence. A finite, declared comparison matrix should
drive promotion and the paper summary. A few explained losing cases do not
require indefinite local tuning, while selected wins cannot establish a broad
performance claim. Retain the fastest verified implementation for each supported
case; private experiments are not promoted merely because they improve their
own weaker predecessor.

## What the current comprehensive evidence actually covers

The latest completed A100 80GB comprehensive cohort is
`comprehensive_20260917/a100-80gb-gpu1` (local artifact: `../results/comprehensive_20260917/a100-80gb-gpu1/summary.json`; not included in this source release):
71 complete cells and one unavailable cell out of 72. Coverage is FFT 29/29,
FWHT 12/12, NTT 13/14, structured-2x2 6/6, subset-zeta 5/5, superset-zeta 5/5,
and xor-zeta 1/1. The unavailable case requests NTT word64 N=2^16, batch16,
natural input and bit-reversed output. This records an implementation/contract
gap, not a failure of the mapping theory.

The corresponding generated summary (local artifact: `../paper/generated/current_results_summary.json`; not included in this source release)
contains these matching external-library comparisons:

| Operator / baseline | Paired cells | Throughput geomean | Faster / equal / slower |
|---|---:|---:|---:|
| FFT / cuFFT | 29 | 0.9789x | 11 / 1 / 17 |
| FFT / VkFFT | 18 | 0.9928x | 8 / 0 / 10 |
| FWHT / Dao-FWHT | 8 | 2.2779x | 6 / 0 / 2 |
| NTT / GPU-NTT-natural | 6 | 1.1292x | 4 / 0 / 2 |

Ratios are baseline time divided by cuButterfly time. These are results for
that cohort's build, UUID and protocol, not a fresh comparison of the current
source or the private suffix candidates. The complete acceptance includes
three matching inverse external comparisons: FFT FP32/cuFFT, FFT FP64/cuFFT
and FWHT FP32/Dao. Inverse NTT itself is correct/measured, but GPU-NTT's
current adapter does not provide that matching inverse baseline. Summary rows
are flat records; direction is a top-level field, not nested in `workload`.

The older `paper/evidence/manifest.json` snapshot (5/72 complete in September
16 data) must not be used as the current execution completion count. The
current figure renderer uses the September 17 cohort above. Likewise, the
separate 40GB large-FFT cohort and later selected suffix pools must retain
their own build/protocol/UUID identities; averaging them together would change
the workload population and comparison contract.

## What the latest local FFT work establishes

The public block/chunk8 path remains the winner in the two new bounded
radix-8 studies. A dedicated DIF lane FFT improves the previous generic
eight-lane core, and explicit constexpr phases remove further device work.
Both preserve correctness, global stage boundaries and natural output order.
Neither replacement beats the public fast path in these workloads, so no
production implementation, installed package or selector is replaced.

See the radix-8 study (local artifact: `../results/paper_completion_20260918/fft_radix8_suffix_20260919/README.md`; not included in this source release)
and constant-phase follow-up (local artifact: `../results/paper_completion_20260918/fft_radix8_constants_20260919/README.md`; not included in this source release).
These are useful evidence that processing-unit realization matters even when
shared exchange is aligned. They do not require another global dataflow
boundary, establish a theoretical obstruction, or prove a global optimum.

## Minimum next comparison and reporting step

Use the existing public research-compilation and acceptance workflow from
[research_acceptance.md](research_acceptance.md). Freeze the candidate portfolio,
build, target UUID and declared workload population, then precompile before
the exclusive GPU window. Refresh the affected FFT comparison against matched
cuFFT/VkFFT contracts using current public implementations; label inverse,
precision, layout and placement coverage explicitly. Keep correctness-only
extensions, mechanism ablations and external-library speedups in their own
roles. Do not relabel unavailable baselines as measured losses or wins.

Report each operator/library's geometric mean, win/tie/loss count, lower-tail
performance and remaining exceptions, alongside the mechanism ablations.
No new numerical acceptance threshold is silently imposed here: this is a
reporting discipline for the user's "broadly competitive or faster" research
target, not a requirement that every cell beat every library. Existing exact
cohorts remain valid evidence for their recorded builds, and are not rerun
merely to produce more trials.

When that declared matrix supports the broad claim, promote the measured
portfolio through the normal public selector/install path and freeze the
paper's updated evidence and figures. Hardware migration is a separate
milestone; this local-core study does not claim to complete it. Further
optimization of the losing private suffix branch can remain future work.
