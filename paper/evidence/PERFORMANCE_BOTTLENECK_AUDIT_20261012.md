# Performance bottleneck audit (2026-10-12)

CPU-only evidence audit. No GPU profiling was launched. Raw identities, source fields, per-kernel trace categories, static PTX/resource dumps and eight unexecuted NCU commands are retained in the JSON ledger. Recompute with `CUDA_VISIBLE_DEVICES="" python3 paper/evidence/audit_performance_bottleneck_20261012.py`.

## Corrected conclusions

| Claim | Evidence level | Supported conclusion |
| --- | --- | --- |
| direction_selection | source-confirmed interface + measured plans | Forward/inverse plans already created independently with separate semantics and direction keys. Same winning organization is not evidence of tied selection. |
| short_fft_core | unquantified | Slow short xD transforms cannot be causally assigned to weak arithmetic cores from complete-call timings alone. Launch, scratch/layout, native real kernels, Python dispatch and normalization remain possible contributors. |
| normalization | source-confirmed mismatch in diagnostic protocol | Original v2 Torch same-padding diagnostic uses default backward normalization; cuButterfly uses norm=forward with product *= volume. Same convolution outputs but different placement/cost of scaling. New breakdown adds matched-normalization control. |
| padding_label | source-confirmed metadata risk | breakdown cupy_padding uses scipy.fft.next_fast_len(real=True), although runtime competitor is cupyx.scipy.signal.fftconvolve; original filter runner correctly records cp_fft.next_fast_len. Do not assume metadata equals executed CuPy padding without vendor trace. |
| fno_kernel_sum | measured profiler diagnostic | Use cat=kernel trace events only; annotation/device totals and instrumented intervals are not additive kernel totals. GPU kernel work is near parity in this trace and does not establish complete application speedup. |
| layout_loss | normal-timing measured; cause unquantified | Writer-aligned/hermes-xor loses at this fixed N2^20/B4 for all three tested arithmetic families. Need same-point counters to separate bank conflicts, integer address instructions, barriers and occupancy effects. This says nothing about automatic-library winner selection. |
| resident_gain | normal-timing measured; source mechanism inference | FWHT resident2 improves this fixed point about 3.5%; two-stage register leaves reduce stage-publication frequency in source. Actual bank/occupancy/address/stall contribution is not measured at this point. |

## Matched fixed-point timing

| Study | Baseline us | Treatment us | Baseline / treatment | Changed axes |
| --- | ---: | ---: | ---: | --- |
| fwht | 185.088 | 236.001 | 0.784268 | ['shared_layout'] |
| fft | 453.448 | 467.845 | 0.969227 | ['shared_layout'] |
| ntt | 623.391 | 659.384 | 0.945414 | ['dataflow_layout'] |
| fwht-residency | 184.873 | 178.616 | 1.035030 | ['local_stage_partitions'] |

All four comparisons use N=2^20, B=4, [10,10] physical groups, 256 threads and three ordinary timing trials. Labels such as `median_treatment_over_baseline` in existing reports actually contain baseline/treatment; the audit recomputes the direction from absolute latencies. These are fixed mappings, not automatic-library performance. FWHT residency v2 uses linear shared layout in both cells; historical helper labels must not be mistaken for a layout change.

## FNO kernel-only trace

| Backend | Kernel count | Kernel duration sum us | FFT kernel sum us |
| --- | ---: | ---: | ---: |
| cubutterfly | 118 | 594.044 | 73.758 |
| torch | 122 | 604.610 | 74.015 |

These are sums of profiler kernel durations, not critical-path or normal end-to-end times. `gpu_user_annotation`, `user_annotation`, copies and runtime events are excluded. Kernel category grouping is name-based and all exact names remain in JSON. Annotation `application/fno_*` totals and CUDA interval timings cannot be summed with child kernels.

## Normalization and padding controls

| Filter | Matched-normalization Torch extra over original diagnostic us | cuButterfly extra over matched Torch us |
| --- | ---: | ---: |
| image1024_filter31 | 41.370 | 15.360 |
| image1024_filter31_batch4 | 114.893 | 52.838 |
| image2048_filter31 | 191.898 | 101.171 |
| image2048_filter31_batch4 | 455.270 | 111.206 |

The additional matched-normalization control shows that some earlier same-padding gap belongs to normalization execution differences. These are paired single-process diagnostics, not an isolated arithmetic-core factorial. An existing v2 N512 filter31 B4 record has scipy reference padding [576,576] but actual CuPy metadata [560,560], demonstrating that the two next_fast_len implementations must not be interchanged. The breakdown metadata uses SciPy for the field labelled cupy_padding; actual execution padding remains to be traced for its four cases.

## Finite NCU Preparation

Nsight Compute 2024.1 is installed; the read-only driver parameter is `RmProfilingAdminOnly: 0`. This suggests non-sudo profiling is permitted but does not runtime-test counters. Eight commands profile two group kernels for each of the four already timed pairs, using the frozen launch environment and UUID. No command was executed.

Each prepared command retains 100 warmup calls, reduces repeat to one, filters shared_iterative/shared_resident kernel names, skips 200 matching warmup group launches and captures two group kernels. Source timing loops support the skip assumption; the eventual capture must confirm the two expected group names and mapping identity. Counters are kernel-only and memcpy is excluded. Profiler latency does not replace the ordinary timing row.

Keep bank conflicts, address/integer instructions, barriers/scoreboard, register/shared resources and occupancy per kernel. Sum byte/event counts across the two kernels where meaningful; do not average ratios or infer complete-call traffic. Compare same semantic point and same kernel scope only. Launches must wait for the specified UUID to be idle/exclusive and must not touch the GPU1 queue.

Static PTX and resource dumps are from hash-bound ready modules. Static address/bitwise counts may motivate hypotheses but cannot establish executed instruction costs or a bottleneck. cuobjdump reports static SHARED:0 while this template allocates dynamic shared buffers; do not call the kernel shared-memory-free.

## xD real-executor code audit

This appendix audits current source, not an identity assertion about the already measured frozen binary. No runtime edit, build or GPU run was performed. The JSON binds source hashes and each selected plan to its original result field. There is no `src/xd_executor*` file in this tree: execution is implemented in `src/xd_real_fft.cu`.

| Shape | B | Forward axis order / cores | Inverse axis order / cores | Inverse post-scale? |
| --- | ---: | --- | --- | --- |
| [1024, 1024] | 1 | [0] / ['cufft-rank1-strided'] | [0] / ['cufft-rank1-strided'] | False |
| [1024, 1024] | 4 | [0] / ['cufftdx-axis-tiled'] | [0] / ['cufftdx-axis-tiled'] | False |
| [2048, 2048] | 1 | [0] / ['cufft-rank1-strided'] | [0] / ['cufft-rank1-strided'] | False |
| [2048, 2048] | 4 | [0] / ['cufft-rank1-strided'] | [0] / ['cufft-rank1-strided'] | False |
| [64, 64, 64] | 1 | [1, 0] / ['thread-shared-axis', 'thread-shared-axis'] | [1, 0] / ['thread-shared-axis', 'thread-shared-axis'] | False |
| [64, 64, 64] | 4 | [1, 0] / ['thread-shared-axis', 'thread-shared-axis'] | [1, 0] / ['thread-shared-axis', 'thread-shared-axis'] | False |
| [128, 128, 128] | 1 | [1, 0] / ['thread-shared-axis', 'thread-shared-axis'] | [1, 0] / ['thread-shared-axis', 'thread-shared-axis'] | False |
| [128, 128, 128] | 4 | [0, 1] / ['thread-shared-axis', 'thread-shared-axis'] | [1, 0] / ['thread-shared-axis', 'thread-shared-axis'] | False |

Forward executes vendor last-axis R2C/D2Z then complex axes; inverse executes compact D2D copy, complex axes, then vendor C2R/Z2D (`src/xd_real_fft.cu:745-775`). Both preserve compact n/2+1 storage, not a full-spectrum expansion. All eight selected plans have actual per-axis `packed=false`, empty axis permutations and empty output permutation; `packed_axes=true` only enables a policy and does not imply a pack/restore pass. Forward scale is already fused when its final complex core is fused (754-758); vendor-last cases use a separate complex scale. All eight inverse plans have unit scale and no inverse normalization, so no extra real scale kernel is launched. The 128^3 B4 directions actually select different axis orders. A direction-selection gap or an inverse-normalization explanation for these eight points is unsupported.

Minimal generic optimization: keep the C2R scratch but replace its initial copy by the existing first-axis out-of-place execution when complex axes are nonempty. Unpacked fused cores have const input and separate output; strided vendor plans have equal input/output strides and distances (690-695), so changing base pointers requires no alternate stride plan. Prefix loops offset both pointers equally (851-854). Packed first axes already permute into owned storage (798-805); final FFT/restore chooses the required scratch destination (810-811,869-871). This applies to all eight current diagnostics and real rank>=2 plans; retain the copy for rank1/no-axis because vendor C2R may overwrite the input. Real batch tiling is disallowed (145-146), so there is no real tiled-tail exception. Source feasibility does not quantify copy cost or guarantee speedup.

Suggested replacement at 762-765 (not applied):

```cpp
if (axis_plans_.empty()) {
    check_cuda(cudaMemcpyAsync(scratch, input_complex, compact_bytes_,
                               cudaMemcpyDeviceToDevice, stream),
               "copy xD inverse compact input");
} else {
    execute_c2c_axes(const_cast<Complex*>(input_complex), scratch,
                     CUFFT_INVERSE, true);
}
```

Validate an isolated new binary with full-output error and exact caller-input preservation across vendor strided/prefix loops, fused local/register/axis/pair, packed/mixed axes and restore parity, fp32/fp64, rank1 fallback, eager/graph. Vendor in-place versus out-of-place algorithms need not be bit-identical. No new floating threshold is defined here.

A separate optional optimization can move nonunit inverse scale to the last fused complex-axis write and skip the final real scale. This is mathematically linear-equivalent but changes floating rounding and possible underflow/overflow; report errors and placement, not bit-exactness. Preserve post-scale for vendor-last/no-axis paths. It benefits none of these eight inverse plans and must not be bundled into the copy-elision attribution.
