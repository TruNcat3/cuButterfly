# Performance bottleneck round: 2026-10-12

This note records one bounded optimization round on the A100 80 GB GPU with
UUID `GPU-0eae5f07-33de-321b-70a9-ffeae49309e6`. It uses new result roots and
does not replace earlier cohorts. Compilation, plan creation, and warmup are
outside the reported steady-state timings.

## What was measured

The tests used the existing frozen FP32/compact xD FFT contracts and the
existing QuaRot random-weight contract. Every full application candidate was
checked before timing. NTT and FFT fixed mappings were profiled with NCU only;
NCU replay values are diagnostic and are not substituted for normal timings.

| weak path | controlled change | result |
| --- | --- | --- |
| QuaRot short FWHT | FP16 load, FP32 warp accumulation, one explicit scale, FP16 store in one kernel | four N32/N64 requests exact against Dao and the old composed path; large standalone cases fell from 51.91 to 24.18 us and 69.02 to 24.46 us |
| QuaRot full application | explicit short-FWHT provider, same model/input and zero-error gate | B1 end-to-end 2482.40 to 2461.87 ms; B4 2590.96 to 2560.40 ms. This provider is not yet in public search |
| xD C2R | remove the initial compact-spectrum device copy when complex axes exist | with old mapping replayed into the new library, all 8 points had exact output bytes and preserved input; eager time fell 10.96%–23.21% across 2D/3D and B1/B4 |
| FNO boundary | remove duplicate fixed-shape Python flatten/validation; compare eager and equal whole-model CUDA Graph controls | eager cuButterfly remains 2.7%–6.2% slower at 64/128 resolution, while graph replay is 1.9%/3.5% faster. Graph is a separate control, not an eager replacement |
| FFT/FWHT/NTT fixed points | NCU replay of the two stage groups | high active warp percentage (90.0%–95.9%) rules out a simple occupancy shortage; long-scoreboard stalls remain 19.1%–41.5%, with more pressure in linear FFT/FWHT paths |

## C2R copy-elision evidence

The candidate is `results/paper_targeted_20261012/c2r_copy_elision_build_v6/`
and changes only the inverse branch in `src/xd_real_fft.cu`. It uses the
existing out-of-place first complex-axis executor when `axis_plans_` is
nonempty, while retaining the copy for rank-1/no-axis cases and retaining the
workspace required by the vendor C2R boundary. The old and candidate shared
the old library's selected mappings in the final run. CUDA Graph capture was
unavailable for both libraries because the worker captured on the default
stream; no graph result is claimed.

Final report: `results/paper_targeted_20261012/c2r_copy_elision_gpu_v5/`.
The standalone candidate is experimental and has not been promoted to the
public selector.

## NCU interpretation

The successful capture is in
`results/performance_bottleneck_profile_20261012/guarded_v3/guard/release/`.
For the first and second stage groups, respectively, the measured kernel time
in microseconds was:

| mapping | group 0 | group 1 | registers | long-scoreboard stall |
| --- | ---: | ---: | --- | --- |
| FWHT linear | 37.79 | 143.23 | 16 / 16 | 31.30% / 19.05% |
| FWHT writer-aligned | 86.34 | 146.53 | 27 / 30 | 12.03% / 18.26% |
| FFT linear | 131.33 | 318.72 | 29 / 30 | 38.43% / 36.33% |
| FFT writer-aligned | 148.61 | 316.42 | 32 / 30 | 25.11% / 33.03% |
| NTT linear | 163.33 | 453.47 | 30 / 29 | 26.08% / 25.40% |
| NTT writer-aligned | 202.50 | 452.90 | 30 / 32 | 16.84% / 24.02% |

The counters support a memory-dependency/scoreboard investigation and a
stage-specific layout tradeoff. They do not prove that one mapping is globally
optimal. The FWHT resident probe lowered its first-group time to 31.81 us but
raised registers to 21; it is a separate fixed-point result.

## Remaining limits

The FFT filter cohort still trails same-padding Torch after filter-cache reuse
by roughly 3.6%–11.5% across the recorded size, batch, and normalization
contracts. CuPy's actual real `next_fast_len` is 1080/2100; the old 2160 label
is not used for this cohort. The remaining gap is therefore an unresolved xD
core/launch issue, rather than an unmeasured padding convention.
The fused FWHT provider and the C2R copy-elision library require integration
into the public candidate registry, recalibration, and a fresh comprehensive
matrix before they can support a general library claim.

## Validation

The affected CPU and harness tests pass: `23 passed` across the NCU parser,
C2R runner, filter normalization, fused short FWHT, and QuaRot candidate
tests. Full outputs, hashes, mapping provenance, and failed attempts remain in
their respective result roots.
