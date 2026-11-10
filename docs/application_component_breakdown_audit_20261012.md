# Application component breakdown audit

`experiments/tools/profile_application_boundaries_20261012.py` now emits the common
`cubutterfly-application-breakdown-v1` schema while preserving the previous
profiler fields and timing files.

The schema fixes the comparison boundary:

| Component | Meaning in the current evidence |
| --- | --- |
| `compile_plan` | Preparation cost. Excluded from timed application calls and currently not independently sampled by this profiler. |
| `adapter_conversion` | Contiguous conversion, dtype conversion, reshape, scale and wrapper work. Included in eager timing; isolated samples are diagnostic only. |
| `operator_gpu` | FFT/FWHT GPU execution and its device-side kernels. Profiler entries may be nested under an application interval. |
| `application_rest` | The surrounding model/application work, dispatch, and other GPU/CPU activity. |

The headline timing is an eager CUDA interval, or a separately labelled CUDA
Graph replay control. It includes adapter, operator, and application work, and
excludes compile and plan setup. Profiler event totals are marked nested and
non-additive. Isolated QuaRot stages are retained as diagnostics and are not
summed to reconstruct the composed call.

Current evidence gaps are explicit. FNO has no independent compile/plan,
adapter-only, or application-only GPU samples. QuaRot has isolated casts, scale,
and core samples, but they are not additive and omit the common model matmul and
`hadK` path. The FFT filter profiler is produced by a separate xD tool and does
not currently expose the same component schema; its stage records must remain
independent measurements until that tool is migrated. No timing is fabricated
for these missing components.
