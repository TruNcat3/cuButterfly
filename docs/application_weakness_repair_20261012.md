# Application weak-path repair (2026-10-12)

The operator diagnostics and application measurements are separate scopes.
This record keeps the application boundary visible and records the repairs
that can be applied without changing the calibrated CUDA library.

## Evidence before the repair

The FNO Darcy forward has eight FFT boundary calls. In the existing trace, the
cuButterfly FFT device time is about 220 microseconds across those calls,
compared with about 326 microseconds for the PyTorch FFT calls. The complete
forward is nevertheless 3.51 ms for cuButterfly versus 3.43 ms for the native
model. The saved result is
`results/paper_targeted_20261012/application_diagnostics/fno_v1/release/fno_end_to_end.json`.
The operator wins inside the application, but the framework boundary and the
remaining convolutions, tensor construction, and launch gaps dominate the
3.4 ms forward. A temporary wrapper simplification was tested in a paired
run and was slightly slower, so it was not retained. The separate whole-model
CUDA-graph control already measures 1.019x at resolution 64 and 1.035x--1.037x
at resolution 128 for the same fixed-shape protocol, showing that this weak
case is a dispatch boundary rather than an FFT arithmetic failure. Those
controls are under `results/paper_targeted_20261012/fno_dispatch_v2/`. They
use the older frozen library `24486975...`, not the current calibration build
`08db08db...`, and fixed device inputs. They establish a useful execution
mode to integrate, not current-build application acceptance.

The FFT filter runner recomputed the fixed filter spectrum on every call. It
also compared cuButterfly at a power-of-two padding with CuPy's native fast
padding. The former makes an application result useful for a complete-call
workflow but obscures the reusable-filter case.

The QuaRot adapter's FP32-composed path performed a fresh output allocation,
an independent scale kernel, and an FP32-to-FP16 allocation/copy on every
FWHT call. This is visible in the application traces even though the FWHT
kernel itself is competitive.

## Changes

`experiments/workloads/e13/run_fft_filter_end_to_end.py` now has an explicit
`--cache-filter-spectrum` mode. It precomputes the fixed filter spectrum
outside the timed region for cuButterfly, PyTorch, and a CuPy native-padding
control. The original complete-call mode remains unchanged. The preparation
path also keeps the SciPy convolution oracle available on hosts with an old
`scipy.fft` namespace.

`experiments/workloads/e13/quarot_adapter.py` adds an opt-in `reuse_buffers=True`
mode. It reuses the FWHT output and the narrowed FP16 output for a serialized
inference stream. The standalone default still returns independent storage.
The QuaRot runner enables this mode and avoids a redundant `contiguous()` when
the input is already contiguous. No shared-library or CUDA source was changed
by this repair, so the running A100 calibration identity remains valid.

## Rechecks

The filter and buffer-reuse checks use the current library `08db08db...`.
The standalone fused-provider full-model probes below use the frozen library
`24486975...` for their public-adapter control, with the current Python adapter.
Compare backends within each probe; these are not a version speedup study.

The standalone FP16-storage/FP32-compute FWHT dispatch test used batch 128 and
extent 64. Fresh output averaged 73.62 microseconds and the reuse path 66.92
microseconds, a 1.10x dispatch improvement. Both paths were finite and kept
the input unchanged. This is an adapter result, not a full model result:
`results/real_xd_dispatch_20261012/quarot_dispatch_ab.json`.

The latest full random-weight QuaRot run passed the zero-error gate after the
FP16 narrowing buffer repair. For batch 1, prefill was 78.94 ms versus 75.16
ms native, decode 2475.80 ms versus 2443.06 ms, and complete prefill plus
decode 2558.02 ms versus 2511.82 ms. The remaining end-to-end gap is 1.84%
for this run, so output reuse and narrowing-buffer reuse alone do not eliminate
it. The result is
`results/real_xd_dispatch_20261012/quarot_reuse_b1_gate_v2.json`.

For the fixed 1024-by-1024, 31-by-31 filter, batch 1, the same current
library and numerical gate produced:

| mode | cuButterfly | PyTorch same padding | CuPy native padding |
|---|---:|---:|---:|
| recompute filter spectrum | 332.08 us | 284.47 us | 366.08 us |
| cache filter spectrum | 225.48 us | 199.68 us | 325.84 us |

Caching removes about 32% of cuButterfly's complete-call time and about 30%
of the PyTorch control time. The explicit output-reuse buffers change this
case by only about 0.5%, so allocator reuse is not the remaining bottleneck.
cuButterfly remains 12.9% slower than the matched PyTorch same-padding control
in this case, while it is faster than the CuPy native-padding control. All
three outputs passed the explicit `atol=1e-5`, `rtol=1e-6` gate. Reports are under
`results/real_xd_dispatch_20261012/filter_cached_cohort/`.
These thresholds were supplied by this diagnostic command, not approved as
the paper's floating-point acceptance rule. The user's error-distribution-only
policy remains in force; the reports retain all reference errors.

The same filter at batch 16 exposes a separate scaling boundary. With the
static selector, cuButterfly took 4250.62 microseconds, or 265.66
microseconds per image, while the matched-padding PyTorch control took
3374.69 microseconds, or 210.92 microseconds per image. Raising the xD tuning
budget and enabling online measurement selected `runtime_measured` with the
`strided-axes` lowering and reduced cuButterfly to 3802.83 microseconds, or
237.68 microseconds per image. Reusing the image and filter adapter buffers
does not change this result within run variance. The candidate list contained
only two legal plans for this large request. This separates a missing
static-table entry from the remaining xD lowering/core scaling gap; it does
not justify treating the application as solved. The two reports are
`image1024_filter31_batch16.cached.json` and
`image1024_filter31_batch16.cached-measure8g-reuse.json` in
`results/real_xd_dispatch_20261012/filter_cached_batch16/`.

The final B=16 complete-call comparison is 3802.83 us cuButterfly, 3373.36 us
PyTorch with the same padding, and 1434.62 us CuPy with its native padding.
cuButterfly is 12.7% slower than the matched-padding control and 2.65x as slow
as native CuPy. The padded domain is 2048x2048 versus CuPy's 1080x1080.
The current fused xD axis inventory stops at length 1024, so this request has
only packed/strided vendor-axis candidates. This is a lowering/core coverage
gap that additional model fitting alone cannot remove.

The short FWHT candidate isolates the second application boundary. It computes
the N=32/64 transform with FP32 warp arithmetic, applies QuaRot's scale before
one FP16 store, and is compared against both Dao and the public adapter. On
the same random-weight B=1, prefill=512, decode=32 workload, all three full
model outputs passed the zero-difference gate:

| phase | Dao | public cuButterfly adapter | fused short candidate |
|---|---:|---:|---:|
| prefill | 75.35 ms | 78.41 ms | 72.29 ms |
| decode | 2422.32 ms | 2459.35 ms | 2395.50 ms |
| end-to-end | 2491.46 ms | 2544.53 ms | 2474.74 ms |

The values above are the candidate probe's medians for the individual
prefill/decode/end-to-end phases. The candidate reaches 1.042x, 1.011x and
1.007x Dao respectively, whereas the public adapter reaches 0.961x, 0.985x
and 0.979x. It remains a standalone provider with
`public_search_integrated=false`; the result demonstrates the missing fused
kernel contract and does not yet change the frozen shared library. The raw
report is
`results/real_xd_dispatch_20261012/quarot_fused_candidate_latest_b1/release/result.json`.
The independent B=4 run preserves the same zero-difference gate and reaches
1.068x prefill, 1.016x decode and 1.014x end-to-end Dao, while the public
adapter reaches 0.917x, 0.994x and 0.979x. Its raw report is
`results/real_xd_dispatch_20261012/quarot_fused_candidate_latest_b4/release/result.json`.
Each batch was measured in one process with three timed calls per phase and
the fixed order Dao, public adapter, fused candidate. In particular, the
0.7%--1.4% full-model advantage is preliminary and does not establish
statistical superiority. The updated probe supports rotation across process
trials, records source/GPU identities, and fails if model state changes.
Those reporting additions postdate these two saved measurements.

## Interpretation and next boundary

These results do not support a claim that every application is already faster.
They show three distinct limits. FNO needs framework-level fusion or graph
capture to amortize the boundary because FFT is a small fraction of the
forward. Fixed-filter convolution should use the cache mode when the filter is
reused. QuaRot still pays for FP32-composed conversion and a separate scale
operation; output reuse removes allocator work but cannot remove those two
semantic operations. Fusing normalization into the FWHT execution path would
require a new public FWHT output-scale contract and a rebuilt library, so it is
kept as the next isolated library change rather than silently folding it into
the current application result.

The remaining implementation sequence is explicit:

1. Carry a scale-before-narrowing epilogue through the public FWHT semantic,
   candidate, cache and execution contracts. Preserve native low-precision
   rounding as a different contract. Integrate the already tested short
   provider and verify public selection plus complete QuaRot B=1/B=4 with
   rotated process trials. Do not hardcode an application-specific winner.
2. Extend the xD axis core inventory beyond 1024 and preserve compatible
   inter-axis layouts. Measure complete 2048x2048 B=1/B=16 plans and filter
   calls; keep native-padding CuPy as an application baseline. Follow-up source
   inspection on October 13 corrected the padding diagnosis: the public compact
   real path already accepts arbitrary lengths such as 1080; the application
   runner still forced power-of-two padding. The explicit fast-padding cohort
   and its results are tracked in `e2e_unified_coverage_20261013.md`.
3. Promote the validated whole-model FNO graph mode into a reusable application
   path with changing-input verification, and retest against the same mode
   for PyTorch on the current build. Keep eager results separately.

The active GPU1 calibration library has not been rebuilt or replaced by
these application diagnostics. None of the three remaining integration items
is marked complete by these candidate or historical measurements.

The broader application evidence is also incomplete. The consolidated
`results/paper_targeted_20261012/unified_report/applications.csv` still records
CKKS as missing a complete application result with no source or trial. These
FFT/FWHT diagnostics provide no NTT/CKKS end-to-end acceptance. The QuaRot
candidate cohort must be listed separately from the public-library rows;
neither these B=1/B=4 diagnostics nor earlier operator results fill its B=8
application gap. Random model states establish implementation consistency,
not pretrained quality or realistic generated-token accuracy.
