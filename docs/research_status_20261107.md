# A100 status and FFT-L scope update

The bounded A100 acceptance work is complete for the currently measured
operator and mechanism tracks. This is a scope statement, not a claim that
every declared benchmark combination has been measured.

| Track | A100 status |
| --- | --- |
| Main operator comparison | Complete for the frozen 66-request population |
| NTT repair comparison | Complete for the frozen five-request diagnostic |
| Controlled mechanism study | Complete for the six-case controlled population |
| xD/grouped FFT supplement | Complete for the recorded rank-4/rank-8 population |
| Incremental final acceptance | Complete on A100 80GB; three affected cells, public selection and installed-default continuity checked |
| FFT-L FP16 | Complete for the recorded nine public cells, 76 factor-study rows and 40 local-core geometry rows; results remain error-report-only |
| FFT-L FP16 extension | Independent A100 40GB diagnostic complete: 11 initial pairs, 33 explicit control rows and nine reverse-order confirmation rows; not a calibrated 80GB result |
| FFT-L FP16 automatic selection | A100 40GB eleven-request tuning and two public DEFAULT replay rounds complete; installed Conda profile/template hooks and default module cache verified |
| FFT-L BF16 | The frozen eleven-request cohort is complete; further BF16 experiments are deferred by the user's FP16-only scope decision |
| End-to-end butterfly applications | Nine frozen A100 80GB application processes are complete, with 14 comparison rows across FFT filtering, QuaRot and small HEonGPU bootstrap |
| Internal NTT compositions | 138 requests and 552 exact measurements are complete; these are internal composition controls, not 138 end-to-end applications |

The active benchmark manifests therefore keep BF16 for operators that have a
valid framework contract, while the `fft-l` cases in
[`config/benchmark_quick.json`](../config/benchmark_quick.json) and
[`config/benchmark_full.json`](../config/benchmark_full.json) use FP16 only.
The BF16 artifacts retain their original provenance. cuFFT **Xt does support
BF16 storage**; its mixed internal arithmetic is distinct from the current
native-BF16 scalar core. Deferring BF16 is a scope decision, not evidence that
the vendor has no BF16 implementation. See the
[Xt audit](cufft_xt_strategy_audit.md).

The final acceptance summary is
[`results/a100_closure_20261103/final_acceptance/summary.json`](../results/a100_closure_20261103/final_acceptance/summary.json).
The FP16 FFT-L records are in
[`results/a100_fp16_acceptance_20261105/`](../results/a100_fp16_acceptance_20261105/)
and the BF16 local cohort is in
[`results/a100_bf16_columns_20261106/full_cohort/`](../results/a100_bf16_columns_20261106/full_cohort/).
The independent A100 40GB FP16 extension, including paired cuFFT Xt timings,
is in
[`results/a100_fp16_fft_l_extension_20261107/`](../results/a100_fp16_fft_l_extension_20261107/).
Its initial short-budget run reports speedups from 0.206x to 0.908x, with
median 0.444x. That runner pointed at a legacy 40GB profile directory without
the current static resource artifact, and the 80GB measured registry entries
did not match the device. This is uncalibrated selection evidence, not an
installed/tuned-library comparison or proof of a core regression. The records
are retained without merging them into the A100 80GB aggregate.
The guarded control precompiled all modules and replayed those selected points
against existing mapping proposals and cuFFT. N32768/B256 improved 4.54x over
the initial selected mapping in the first control round. However, its cuFFT
ratio changed from 1.034x to 0.978x on reverse-order confirmation; N262144/B32
changed from 1.460x to 1.052x. These observations separate configuration loss
from the remaining provider gap, but do not establish stable peak speedups or
automatic tuning on 40GB. Both rounds and complete numerical reports are saved.

The subsequent [A100 40GB automatic-selection closure](../results/a10040_fp16_autoselect_20261107/README.md)
remeasured configuration proposals on that card, then confirmed all eleven
requests through public `DEFAULT` selection and the shared Conda installation.
The installed round has geometric-mean cuFFT/cuButterfly speedup **0.8584x**,
range **0.739x–1.088x**, with one nominal win. The staged-profile replay is
consistent at **0.8579x**. This closes the artifact-routing and automatic-use
gap for those measured keys; it does not remove the remaining FP16 performance
deficit. Independent stage-service calibration was skipped in this bounded
40GB supplement, so the model is explicitly unvalidated for unseen requests.
The existing 80GB results retain their own device and build provenance.

## Application evidence correction

The earlier statement that only historical Poisson evidence existed was
incorrect. The [application audit](../results/a100_e2e_closure_20261107/README.md)
checks the completed nine-process queue and retains its recorded build identity.
The 138-request count refers to internal NTT compositions, not missing
application runs. Completed measurements are reusable with their original
provenance; a newer library hash does not relabel them as new measurements.

| Scenario | Butterfly type | Baseline | Recorded speedup |
| --- | --- | --- | --- |
| Image filtering, 1024²/2048², B16 | FFT FP32 | CuPy cached-filter Graph | 1.048–1.069x |
| Random-weight Llama-2-7B, B1/B4 | FWHT FP32-composed | QuaRot / Dao FWHT | 0.957–0.968x |
| Small regular bootstrap, N4096, B1/B8 | NTT | HEonGPU native | 1.032–1.033x |

All rows compare complete Graph executions. Filter errors remain report-only,
as requested; an unset floating threshold is not an unfinished measurement.
Bootstrap uses `sec_level=none`, and its source/binary correspondence is marked
`not_verified`; it cannot establish production-security FHE performance.

The frozen A100 operator, mechanism, composition and application cohorts are
measured. This does not mean all Cartesian combinations are supported or faster,
nor does it establish every application's performance for the newly installed
FP16 build. Retain the negative regions and the exact scope in paper claims.
