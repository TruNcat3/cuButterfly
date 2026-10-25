# Weak-case diagnosis and common-path repair

This note records the current-build diagnosis for the two negative end-to-end
controls. It keeps execution-boundary effects separate from lowering and
candidate-selection effects, so a weak row does not get attributed to the
butterfly formulation without a trace.

## Observed cases

| Case | Eager speedup | Graph speedup | Selected path | Diagnosis |
| --- | ---: | ---: | --- | --- |
| 2048², B=4, FP32 | 0.536x | 1.159x | VkFFT axis / cuFFTDx axis | Host submission and multi-kernel call boundary dominate eager mode |
| 2048², B=16, FP32, current build | 1.024x | 1.177x | VkFFT axis / cuFFTDx axis | Current search finds a fused complete plan; the remaining gap is boundary and application work |
| 2048², B=16, historical build | 0.842x | 0.840x | packed fallback | Static xD fallback caused four transpose launches and a separate scale kernel |

The historical B16 trace measured 12 serialized kernels. Four transposes
accounted for about 38.9% of kernel time and the separate scale kernel for about
10.3%. The runtime-measured path reduced the plan to seven kernels and removed
those two avoidable boundaries. The current B16 reports are
[`report.json`](../results/e2e_current_20261017/poisson_2048_b16_torch_v2/report.json)
and the current B4 report is
[`report.json`](../results/e2e_current_20261017/poisson_2048_b4_torch_v2/report.json).

The B4 eager result is a different class of weakness. Its complete-call CUDA
work is faster after Graph capture, but direct calls submit enough small
operations that host launch latency dominates. This is why the public Graph
adapter is part of the repeated-call execution path and why eager and Graph
numbers are reported separately.

## Common framework repair

The xD selector now emits structured diagnostics in `selection_reason` for
both complex and compact-real transforms. The diagnostics include the number
of generated choices, executable candidates, measured candidates, rejected
candidates, live input/output bytes, workspace bytes, the tuning budget, free
memory, and the exact reason when the complete-plan search is skipped. The
mapping JSON remains semantic and replay-stable; diagnostics are deliberately
kept outside the replay contract.

The tuning memory gate is shared by both xD paths. It reports one of
`within-budget`, `io-exceeds-half-budget`, `workspace-exceeds-budget`,
`temporary-allocation-exceeds-half-free-memory`, or a CUDA memory-query error.
When the gate is closed, the plan remains executable, but the resulting static
selection is explicitly marked as unmeasured rather than being presented as a
measured optimum.

The compact-real cache generation is now `xd-real-v22-adaptive-tuning`. A
change to the complete-plan tuning protocol therefore invalidates old static
packed entries. This prevents an installation upgrade from silently reusing a
mapping produced before the adaptive budget repair.

## Verification

- Current build: `build-contract-20261017/libcuntt.so`.
- B4 and B16 current-build plans both report `selection_source=runtime_measured`
  and `tuning.reason=within-budget` on the A100 80GB test GPU.
- B16 forward reports 29 generated choices, 25 executable candidates and 52
  measurements after axis refinement. The inverse reports 47 executable
  candidates and 72 measurements.
- Relative L2 error against same-precision CPU FP32 is `4.14e-7` to
  `5.82e-7`; every output is finite and independently allocated.
- A controlled `CUBUTTERFLY_XD_TUNE_MAX_BYTES=1` run is retained under
  [`diagnostic_budget_gate`](../results/e2e_current_20261017/diagnostic_budget_gate/report.json).
  It selects the executable packed fallback but reports
  `selected_without_measurement=true` and `reason=io-exceeds-half-budget`;
  this is a gate-behavior check, not a performance point.
- The C++ semantic, mapping, NTT-template, and C API regression tests pass.

The remaining B4 eager deficit is therefore an execution-mode property, not a
missing design point. The remaining application work is to add the same
complete-call Graph controls for the other butterfly-heavy applications and
to retain per-operator profiles under the same timing and trace contract.
