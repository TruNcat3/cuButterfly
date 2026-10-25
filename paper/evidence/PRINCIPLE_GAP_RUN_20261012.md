# Principle-gap run audit (2026-10-12)

This records actual v1 and v2 execution, not user floating numerical acceptance. Machine-readable
provenance, exact source hashes and resolved mappings are in
[PRINCIPLE_GAP_RUN_20261012.json](PRINCIPLE_GAP_RUN_20261012.json).

## Separate Gates

- Execution: isolation guard released the attempt; telemetry completed with return
  code 0. Telemetry samples cover child lifetime, not exact CUDA-event intervals.
  Its 205 thermal samples range from 37 to 38 C; observed NVML reason masks are
  0 and 1, with no thermal-slowdown bits (`0x20 | 0x40`).
  v2 also released cleanly: 226 samples, no thermal-slowdown bits, return code 0.
- Preparation: three `compile_pending` events remain in the attempt log. Two
  unique module requests subsequently became ready; their actual module hashes
  match broker records. This does not retroactively pass the v1 preparation gate.
  v2 has no preparation event file, no pending/failed events, and all preflights
  and three trials per variant completed using the frozen ready modules.
- Mapping: v2 completes FWHT B1/B16 and NTT B1; NTT B16 remains the v1 subset.
  Resolved differences are exactly `shared_layout` for FWHT and `dataflow_layout`
  for NTT; physical group,
  core, partition, threads, batching and overlap controls are unchanged.
- Numerical: NTT B1/B16 passed exact full-batch internal reference checking for both
  layouts (1/16 batches). FWHT variants passed existing benchmark
  internal sanity only. User floating numerical acceptance remains unassessed;
  error distribution is missing. No new or relaxed threshold is introduced.

## Observed Timings

All values below are microseconds for the launch sequence, not full C API latency.
Three independent benchmark-process trials use the frozen 100-warmup/100-repeat
protocol. Results are descriptive; three trials do not establish significance.

| Case (source) | Linear trials | Writer-aligned trials | Median linear / writer |
| --- | --- | --- | --- |
| FWHT N4096 B1 (v2) | 10.537, 10.199, 10.537 | 10.557, 7.854, 10.445 | 1.008808 |
| FWHT N4096 B16 (v2) | 10.578, 10.578, 10.619 | 11.315, 11.254, 11.274 | 0.938265 |
| NTT N4096 B1 (v2) | 10.322, 10.547, 10.721 | 10.824, 10.824, 10.844 | 0.974409 |
| NTT N4096 B16 (v1) | 15.340, 15.360, 15.370 | 16.138, 16.108, 16.097 | 0.953563 |

Paired medians (linear / writer) are 10.537 / 10.445, 10.578 / 11.274,
10.547 / 10.824, and 15.360 / 16.108 us, respectively. v2 FWHT B1 has a low
7.854-us treatment trial; retain this variability rather than claim significance.

NTT B16 medians are 15.360 and 16.108 us: the writer-aligned layout is slower
at this point. Archive this complete conditional subset without claiming whole
v1 acceptance. The original NTT B1 v1 treatment had no baseline; it is not
substituted for the complete independently rerun v2 pair. Original v1 rows
remain nested in JSON, including successful treatment timings and pending errors.

## Direct Failure And Minimal Repair

FWHT B1/B16 linear preflights encountered pending request
`8e4cef433105ee43585397cc0d767512797a4f8ba14c64d6612dad93faebea93`.
NTT B1 linear encountered pending request
`f7e9a834026e1bd24dc93fadc9d07e0a379e37aa68c98df61a8e3ee5737ef6e4`.
The controlled runner records a failed command as a preflight error and does not
retry that variant after compilation completes. These failures occurred before
verification; they are not observed numerical failures.

[v2 config](../experiments/principle_gap_campaign_20261012_v2.json) prepares only
the three missing pairs (six variants, 18 timing processes). It uses a fresh
output and attempt artifacts, keeps NTT B16 from v1, and preserves the frozen v1
wrapper, config and failed records. No low-level runtime change is required.
The current benchmark mapping path exposes no error-distribution report;
adding such a report is a separate `blocked_implementation` dependency, not
silently satisfied by sanity or CPU tests. Completion requires full-batch error
distribution for the same resolved mapping, input and reference without inventing
a user threshold. This is mechanism ablation I layout evidence, not fixed-core
organization transfer (mechanism ablation II).

The repair audit CLI now returns nonzero on rejected mapping/internal-sanity
audits. This post-run CPU-only fix does not change the executed wrapper or config.
