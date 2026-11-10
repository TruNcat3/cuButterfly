# Evaluation protocol notes — 2026-11-08

These notes preserve protocol details shortened in `paper/sections/evaluation.tex`.
They describe existing measurements, not new experiments or an expanded claim.
The four manuscript subsections and all measured results retain their original
populations. These protocol notes require no new GPU measurements.

- **CKKS holdout timing:** after loop/stage-batch selection, each request uses
  twenty holdout event-timed rounds of ten Graph replays. These replays are not
  independent process trials. The native-loop, matched-batch, and faster-native
  comparisons retain the same 36 requests.
- **Fixed-leaf FP32 FFT organization control:** six workloads × four cells ×
  three trials give 72 checked rows, with 10,000 warmups and 100 repetitions.
  Ownership, storage, launch geometry, and wrappers change with organization;
  this does not isolate residency alone.
- **Non-FFT endpoint selection:** candidates and resource expectations are frozen
  before screening. Three new processes confirm the selected mappings; the
  second reverses order. Conventional/extended pools are 2/6 for FWHT and 1/3
  for NTT under a six-candidate ceiling. All candidates execute successfully;
  the NTT domains retain the original mapping rather than forcing a change.
- **FP64 factor profiles:** each profile captures one complete call after
  preparation and warmup. NCU replay has uncontrolled caches and clocks;
  ordinary unprofiled timings supply speedups. NSYS establishes launch scope,
  not overlap within a kernel.

Bootstrap retains `sec_level=none` and
`source_binary_correspondence: not_verified`; its approximately 1.03× result
is a recorded small-parameter comparison, without verified correspondence
between the source and measured binary. This is separate from CKKS's exact
ciphertext checks.

Numerical coverage and reference arithmetic are documented in
[NUMERICAL_SCOPE_20261108.md](NUMERICAL_SCOPE_20261108.md). The main FP32 reference uses FP64 arithmetic. That diagnostic and the
same-precision FWHT and explicitly rounded FP16 controls have different
reference contracts and remain separately reported. No floating acceptance threshold is introduced.

Application inclusion follows
[APPLICATION_COHORTS.md](../generated/current_refresh/APPLICATION_COHORTS.md):
two checked historical FNO observations remain in the upper panel; two early
QuaRot observations are archived and excluded for invalid random-state
preparation and absent output checks. Independent application follow-ups retain
separate build and timing identities.

## Preparation observations retained outside the main narrative

A ten-request public DEFAULT FWHT check took 43.2 minutes to prepare plans.
CPU preparation changes preserved all scores and rankings across 30,146 and
371,848 candidates. Three public requests retained their mappings and outputs
while preparation fell from 17.7 to 14.8 minutes. These shared-host measurements
are neither interleaved trials nor an installation benchmark. They do not
establish a new search contribution or a hardware-independent preparation cost.
