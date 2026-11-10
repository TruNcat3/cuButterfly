# cuButterfly research evidence gap review

This review fixes the evidence boundary for the A100 80GB research cohort
`a100-80gb-research-0.9.3.11`. It separates measurements that can support the
paper from historical diagnostics that use an older binary or a different
timing contract.

## Evidence that is usable now

| Evidence | Current state | What it supports |
| --- | --- | --- |
| 66-case operator comparison | Complete, Graph protocol, 29 exact integer rows and 37 finite floating-point error distributions | The unified semantic contract and the measured operator comparison population |
| Five-point NTT search repair | Complete on one A100 80GB UUID and one current library build | The bounded candidate inventory repair; public-search geometric mean 1.1765× against GPU-NTT for the five selected word64 points |
| Grouped rank-4/rank-8 FFT supplement | Complete, separate adapted cohort | That the xD adapter path can be competitive for the two measured FP32 shapes |
| Current mechanism cohort v2 | Complete on the same A100 80GB UUID and current build | A controlled organization witness across FFT, FWHT and NTT, with full-batch correctness, launch geometry, shared memory, registers and a paired NCU/NSYS profile |
| Current constrained selector/core controls | FFT, FWHT and NTT selector ablations plus six legal FWHT/NTT core controls complete on the current build | The selector restriction is executable and auditable; core controls are reported only inside each lowering's support domain |
| Current-build Graph control | PyTorch/cuFFT versus cuButterfly 2D Poisson controls complete at N=1024/2048, B=4/16 | Current-build complete-call eager and Graph results with explicit setup exclusion and FP32 error distributions; image/FWHT/NTT application rows remain open |
| CPU external baselines | Complete but shared-host provisional | A fallback comparison where no exact GPU baseline exists; not a substitute for a missing GPU implementation |

The mechanism cohort is deliberately interpreted as a control. For its six
requests, changing only `stage_partition` changes the median stagewise-over-
resident8 ratio from 10.87× to 22.44×. The large ratios are expected because
the stagewise control launches one global-scratch kernel per stage. The profile
for FWHT `N=2^18, B=1` records 18 versus 3 kernels, 131072 versus 1024 CTAs,
and about 2091 versus 88 microseconds of profiled kernel duration. These data
explain launch and boundary costs; they are not an external-library speedup
claim and do not establish that resident8 is optimal for every request.

## Must be added before a strong paper claim

1. **More current-build Graph end-to-end measurements.** One canonical
   PyTorch/cuFFT control is complete. Retest a small, broader set with the
   current build and the same Graph contract: image filtering for
   FFT, one FWHT-heavy QuaRot path, and one NTT-heavy CKKS/FHE path if the
   complete application implementation is available. Use matched `B=1,4,16`
   where the application supports it. Include adapter, permutation, padding,
   normalization, and output conversion in the complete call. Report the
   butterfly fraction so a small operator fraction is not presented as a
   library failure.
2. **Current-build profile comparison.** One representative profile per
   operator is enough for the paper if it reports the same counters and the
   same semantic call. At minimum retain kernel count, grid geometry, shared
   bytes, registers, active warps, barrier/scoreboard stalls, and DRAM/L2
   traffic. NCU/NSYS runs remain diagnostic and must not replace normal timing.
3. **Accuracy and gap table.** Keep the complete floating-point error
   distributions beside the timing table. Keep missing GPU implementations,
   unsupported xD bridges, modulus limits, and unresolved application paths as
   explicit cells instead of silently substituting CPU, another precision, or a
   flattened rank.

## Historical data policy

The old FFT core/dataflow factorial, old FWHT/NTT layout controls, and the
2026-10-13 FNO/Poisson/registration/QuaRot runs remain valuable diagnostic and
motivation evidence. They are excluded from the current main table because
their library SHA or timing protocol differs. They may be cited as historical
development evidence only after the figure caption names the old build and
the separate cohort.

## 2026-10-18 current-build supplement

The current source was rebuilt as both a shared library and a static consumer
library. The new application artifacts are kept in
`results/e2e_current_20261018/` and are not mixed with the older 2026-10-17
controls.

The real-input FFT filter path now has a matched-padding Graph cohort. For
`image1024_filter31_batch16`, using the same `1080 x 1080` fast padding for
all methods, cuButterfly measured 1406.3 us, CuPy 1500.2 us, and Torch/cuFFT
1532.6 us. For `image2048_filter31_batch16`, using the same `2100 x 2100`
padding, the corresponding medians were 5730.8 us, 5903.9 us, and 6007.0 us.
The earlier power-of-two-padding run is retained separately because its CuPy
control used a smaller native padding and is therefore not a matched-padding
comparison. Both matched rows produced finite outputs with relative L2 errors
of at most `5.18e-7` against the same-precision CPU reference.

The current static consumer build also ran a three-channel negacyclic NTT
composition at `N=2^16`, word32, with moduli `1073479681`, `2013265921`, and
`2130706433`. Full-batch exact verification passed for `B=1,4,16`; the complete
call medians were 0.0585, 0.1135, and 0.3285 ms, respectively. This is a
current-build composition witness and does not replace a matched external
CKKS/GPU-NTT comparison.

The composition driver now exposes the same public `--auto-select` request as
`cuntt_bench`; it does not silently substitute a stored mapping. With the
current research build, automatic selection resolved all three channels to the
legal `hybrid2d` radix-4 template (`n1_log=6`, four rows, 256 threads), passed
exact full-batch verification, and measured 0.0680, 0.0800, and 0.1579 ms for
`B=1,4,16`. Relative to the shared-iterative default these are 0.86x, 1.42x,
and 2.08x. A separately supplied, previously measured `n1_log=8` mapping was
faster on this build at 0.0462, 0.0630, and 0.1342 ms, respectively. The
comparison is useful because it identifies a real calibration gap: the public
portable selector currently chooses the first feasible candidate when no
matching measured registry record exists, so `--auto-select` is not equivalent
to an online timing search. The explicit mapping is retained as a control and
is not promoted as a universal default. The three automatic-selection reports
are `ntt_l16_b1_auto.json`, `ntt_l16_b4_auto.json`, and
`ntt_l16_b16_auto.json` under `results/e2e_current_20261018/crypto_ntt/`.
The same generic path also passed exact full-batch verification for inverse
negacyclic word32 at `B=16` (0.1690 ms) and forward negacyclic word64 at `B=4`
(0.0390 ms), with the same explicit runtime-auto-select provenance. These
rows check that the selector extension is independent of direction and word
width; they are coverage controls rather than external-library speed claims.

To verify the intended install-time path, the three measured `n1_log=8`
single-channel records were placed in an isolated current-build registry and
replayed through the same composition command. All channels reported
`selection_calibrated=true` and `mapping_provenance=calibrated-registry`; the
`B=16` three-channel composition fell from 0.1579 ms with the uncalibrated
portable choice to 0.1342 ms with the registry replay (1.18x). This is a
calibration replay control, not a new universal table: the registry is kept
under `results/e2e_current_20261018/crypto_ntt/registry_current_ntt_control.json`
and must be regenerated when the build fingerprint or hardware changes.

The paired current-build NTT profile for `N=2^16`, `B=16` recorded three
repeated channel groups. The shared-iterative core used a `4096 x 128` grid,
26 registers per thread, and 3072 bytes of shared memory per block. Its two
stage groups took about 52.7 us and 85.9 us in the profiler, with 83--85%
active-warps utilization. The first group was mostly compute/synchronization;
the second had about 38% long-scoreboard stalls and 11% barrier stalls. This
points to the next bounded optimization target: retain the selected
hierarchical organization while reducing stage handoff and global/twiddle
readiness stalls. The profiler output is diagnostic and is not used as the
steady-state timing value.

The same profile was repeated for the public automatic mapping. Its hybrid2d
second pass used a `4096 x 256` grid, 29 registers per thread, 2064 bytes of
shared memory, and about 87.7% active-warps utilization. Its first pass used
only 256 CTAs, 30 registers, 17424 bytes of shared memory, and about 27.4%
active-warps utilization, with roughly 37% long-scoreboard stalls. This
explains why the automatically selected `n1_log=6` mapping is correct and
substantially better than the old shared-iterative default at larger batches,
but still loses to the measured `n1_log=8` control: the selector has no
current-build timing record to distinguish the first-pass occupancy tradeoff.
The raw profile is `results/e2e_current_20261018/profile_ntt/ntt_b16_auto.ncu.csv`.

The QuaRot driver now separates an unassessed floating-point distribution from
a failed numerical gate. When no `--atol/--rtol` is supplied, it records all
errors and proceeds to timing after the finite-output check. Explicit
threshold failures and non-finite outputs still stop the run. The first
current-build QuaRot probe exposed large full-model error under the native
FP16 contract. The current FP32-composed eager run for the same seeded random
Llama-2-7B configuration produced finite, exactly matching logits for the
measured prefill/decode outputs and measured 68.87/608.68/677.32 ms for the
Torch baseline versus 69.92/619.61/689.49 ms with cuButterfly FWHT. Thus this
row is a correctness and overhead result (`0.985x` end-to-end), not an
acceleration claim. CUDA Graph replay remains a separate fixed-KV metadata
case and is not included until its changed-input validation completes.

The graph investigation then ran a reduced `prefill=16, decode=1` control in
both variant orders. The baseline-first control produced the same failure as
the cuButterfly-first control: graph replay returned 512000 non-finite prefill
elements and 32000 non-finite decode elements before any timing. This isolates
the current block to the application graph/cache capture path rather than the
FWHT kernel or its numerical contract. The durable diagnostics are
`quarot_debug_graph_p16_d1.json` and
`quarot_debug_graph_p16_d1_baseline_first.json`; no graph speedup is reported
until the native baseline graph is finite.

## Current run artifacts

- Scope: [`a100_80gb_research_v1.json`](scopes/a100_80gb_research_v1.json)
- Current controlled study: `results/mechanism_current_20261017_v2/study.json`
- Timing journal: `results/mechanism_current_20261017_v2/timing.json`
- Timing/profile summary: `results/mechanism_current_20261017_v2/report.json`
- NCU/NSYS manifest: `results/mechanism_current_20261017_v2/profile/manifest.json`
- Selector ablation traces: `results/selector_ablation_20261017/fft_n16384_b16/`
- FWHT selector ablation traces: `results/selector_ablation_20261017/fwht_n16384_b4/`
- NTT selector ablation traces: `results/selector_ablation_20261017/ntt_n16384_b4/`
- Current Graph Poisson control: `results/e2e_current_20261017/poisson_1024_b16_torch_v2/report.json`
- Current Graph Poisson controls: `results/e2e_current_20261017/poisson_2048_b4_torch_v2/report.json` and `results/e2e_current_20261017/poisson_2048_b16_torch_v2/report.json`
- Current Graph Torch-only runner: `experiments/tools/run_periodic_poisson_torch_graph_20261017.py`
- Current matched-padding FFT filters: `results/e2e_current_20261018/fft_n1024_b16/` and `results/e2e_current_20261018/fft_n2048_b16/`
- Current QuaRot FP32-composed eager row: `results/e2e_current_20261018/quarot_b1_p128_d8_eager_fp32.json`
- QuaRot graph finite-output diagnostics: `results/e2e_current_20261018/quarot_debug_graph_p16_d1.json` and `results/e2e_current_20261018/quarot_debug_graph_p16_d1_baseline_first.json`
- Current composed NTT rows: `results/e2e_current_20261018/crypto_ntt/`
- Current NTT profile: `results/e2e_current_20261018/profile_ntt/ntt_b16_all.ncu.csv`
- Selector ablation summary tool: `scripts/summarize_selector_ablation.py`
- Legal core/organization control study: `results/selector_ablation_20261017/core_organization_v2/study.json`
- Core/organization measurements: `results/selector_ablation_20261017/core_organization_v2/run_gpu2/results.json`
- Core/organization summary: `results/selector_ablation_20261017/core_organization_v2/summary.csv`
- Reproducible control runner: `scripts/run_core_organization_controls.py`

All six legal control rows report `correct=1`. The shared FWHT stagewise and
8+6 rows measured 0.379699 ms and 0.011469 ms at N=16384, B=4. The temporal
FWHT radix2 and radix4 rows measured 0.007066 ms and 0.005632 ms at N=1024,
B=16. The specialized NTT radix2 and vector-radix4 rows measured 0.356352 ms
and 0.356864 ms at N=2^20, B=1. These are controlled witnesses, not a claim
that one core or organization wins across all semantic requests.

The exploratory NTT N=2^14 files under
`results/selector_ablation_20261017/core_organization/run_gpu2/` are retained
with the run log but are not adopted: their serialized request changed the
subgraph mapping while the reported execution-group lowering remained the
generic dataflow-radix4 path. This is an implementation boundary, not a
performance result.

The selector ablations use the same semantic request and an eight-candidate
measurement budget for each unrestricted/restricted pair. The unrestricted
best measured kernel times were 0.019149 ms for FFT N=16384, B=16, 0.011674
ms for FWHT N=16384, B=4, and 0.014336 ms for NTT N=16384, B=4. Disabling
stage-overlap, resident-fusion, or successor-layout removed 112292/635/106645
FFT candidates, 13730/379/11056 FWHT candidates, and 13412/381/10813 NTT
candidates, respectively. The restricted best-time deltas were within the
small eight-sample measurement noise for FFT and FWHT. NTT showed 0.7% regret
for resident-fusion and 2.9% for successor-layout, while stage-overlap was
within noise. The NTT successor-layout restricted trace accepted eight
candidates instead of six because two unrestricted candidates were rejected
by their `n1_log` legality check; this is recorded as a trace outcome rather
than silently normalized. These runs establish an executable, auditable
restriction protocol and candidate-population effects. They do not establish
that any individual mechanism is universally necessary or sufficient.

On 2026-10-17 the current-build Poisson Graph controls completed on GPU 2 of
the A100 80GB cohort. At N=1024, B=16, cuButterfly was 1.195x faster than
PyTorch/cuFFT in eager mode and 1.197x faster in Graph mode by complete-call
wall median. The cuButterfly Graph median was 0.7521 ms versus 0.9004 ms for
PyTorch/cuFFT. At N=2048, B=16, eager speedup was 1.026x and Graph speedup
was 1.177x, with cuButterfly at 3.0396 ms versus 3.5789 ms for PyTorch/cuFFT.
At N=2048, B=4, eager speedup was 0.536x while Graph speedup was 1.159x.
This is an execution-boundary effect: the eager complete call submits many
small kernels, while the Graph result measures the intended repeated-call
steady state. All outputs were finite and independently allocated; relative
L2 errors against CPU FP32 were 4.14e-7--5.82e-7. Plan construction, search
and Graph capture were excluded from timing and retained in `setup`.

The current plan reason now records the generic selection diagnostics for both
complex and compact-real xD paths. For the fresh 2048² controls, the forward
compact-real plan exposed 29 generated choices, 25 executable candidates, 52
measured candidates after axis refinement, and a successful 2 GiB/VRAM gate.
The inverse exposed 47 executable candidates and 72 measurements. This makes
the old 0.84x B16 result auditable: its report has no runtime measurements and
selected the packed fallback after the historical fixed budget rejected the
candidate set. A skipped tuning gate is now reported with its byte budget,
free memory, workspace, and reason rather than being presented as a kernel
performance result.
The original CuPy-inclusive runner was also attempted but stalled while
importing the CUDA/NumPy extension set under the host's high I/O load; it did
not consume GPU 2 and is not a measurement. The Torch-only control uses the
same semantic call and remains the accepted current-build result.

The next GPU queue should therefore be current-build Graph application
retests followed by the remaining per-operator profiles. No aggregate
“全面提升” claim is justified until those rows are either measured or
explicitly marked as unavailable.

## End-to-end attribution (2026-10-18)

The current evidence separates three effects that are easy to conflate. For
the N=2048, B=16 Poisson Graph control, the complete-call CUDA-event median is
3.568 ms for PyTorch/cuFFT and 3.027 ms for cuButterfly, or 1.177x. The
instrumented traces report 2.855 ms and 2.393 ms of kernel time, respectively.
Thus the non-kernel remainder is about 0.713 ms and 0.634 ms, roughly 20% of
each complete call. Graph capture has already made this remainder comparable
between implementations; the visible gain is primarily the transform path,
not an artifact of omitting synchronization. The trace files are
`results/e2e_current_20261017/poisson_2048_b16_adaptive_trace/`.

The N=2048, B=4 Poisson control exposes the execution-boundary case. Eager
cuButterfly is 1.842 ms versus 0.987 ms for PyTorch/cuFFT (0.536x), while the
same complete call under Graph replay is 0.852 ms versus 0.988 ms (1.159x).
The transform is therefore not intrinsically slower at this request. Eager
submission and synchronization dominate when the batch is too small to amortize
the sequence of kernels. This row must be reported as an eager integration
limitation, not as a contradiction of the operator result.

The matched-padding FFT filter controls show the application choice when the
butterfly work is a large part of the call. At N=1024, B=16, cuButterfly is
1.067x faster than CuPy and 1.090x faster than the same-padding Torch/cuFFT
diagnostic. At N=2048, B=16, the corresponding ratios are 1.030x and 1.048x.
The 1.53 s and 4.34 s plan/search setup costs are recorded but excluded from
steady-state timing, so these rows do not claim faster first-call startup.

The QuaRot full-model row has a different composition. The FP32-composed eager
run is 677.32 ms for the baseline and 689.49 ms with cuButterfly (0.982x),
with finite, exactly matching reported logits. The FWHT adapter creates four
plans in 545 ms outside the timed interval, and the plans are selected by the
static model rather than a local timing policy. The full run also includes
attention, quantized matrix multiplications, cache movement, and the adapter's
scale/cast/matrix path. Consequently, an FWHT improvement is diluted by the
rest of the model. This is an application-composition and integration result,
not evidence that the butterfly kernel has no benefit. The native CUDA Graph
row is currently blocked earlier: baseline-first and treatment-first reduced
controls both produce non-finite replay outputs, so no Graph speedup is
reported for QuaRot.

These measurements support a layered conclusion. Selected steady-state
operators and butterfly-dense complete calls meet the intended improvement
target. Universal end-to-end superiority has not been established. The
remaining work is to instrument each application with butterfly, adapter,
launch, and non-butterfly time, enable measured registry selection for the
QuaRot plans, and repair the application Graph capture path before comparing
Graph replay. The next application set should include butterfly-dense NTT/FHE
or CKKS pipelines; a general LLM inference row should remain a composition
case rather than the sole performance verdict.

## Follow-up repair and FHE workload contracts (2026-10-18)

The QuaRot runner and serialized adapter benchmark now accept an explicit
`--algorithm-policy default|measure` switch. `measure` performs bounded local
FWHT candidate selection during setup; its cost is recorded outside timed
calls. The report retains the policy and every plan's selection source, so a
`static_model` miss cannot be mistaken for an installation-calibrated result.
The benchmark also accepts `--compute-contract public_fp32_final`, which is the
fused scale-and-final-storage path. It has not yet been promoted to a QuaRot
application result: the small full-model validation was blocked by a host
filesystem page wait before producing a report.

The supplemental CKKS/TFHE contract is
`experiments/workloads/ckks_tfhe_workloads.json`. CKKS uses the pinned HEonGPU v2
regular bootstrap (`N=65536`, 25 Q primes, 5 P primes, scale `2^40`, secret
weight 192) and keeps the v1 `N=4096` path as an explicitly insecure smoke
case. TFHE uses the pinned context (`N=1024`, `n=512`, `k=1`, `bk_l=2`,
`bk_bg_bit=10`) with gate batches 1, 8, 64, and 512, MUX, and depth 2/4/8
NAND/XOR circuits. The native complete-call reference is now measured in
`results/e2e_current_20261018/tfhe_native_full.csv`; all 44 rows passed exact
bit decryption. The full-call runner is
`experiments/workloads/run_heongpu_tfhe_e2e.py` and reports setup/encryption
separately while timing every public gate or circuit call through CUDA
completion.

This is the fair comparison boundary. A cuButterfly treatment does not get to
exclude its adapter, layout conversion, additional launches, allocation, or
synchronization cost. If it wins with those costs included, that is stronger
evidence; if it loses, the loss is a real integration result. HEonGPU fuses
`SmallForwardNTT` and `SmallInverseNTT` inside private bootstrapping kernels,
so the remaining treatment work is a device-level fused callback or templated
bootstrap variant. A standalone cuButterfly NTT latency is retained only as
diagnostic attribution and cannot replace the complete-call row. The pinned
TFHE source exposes no standalone mod-up/mod-down API; those operations remain
CKKS/RNS component cases rather than TFHE claims.

## Measured CKKS operation integration (2026-10-18)

The new `experiments/workloads/run_heongpu_ckks_e2e.py` replaces the NTT calls
inside complete CKKS key switching, square/relinearize/rescale, square followed
by key switching, and slot rotation. The original keys, ciphertexts and
non-butterfly computation are shared. The adapter performs per-prime root
conversion, negacyclic twist, packing and output permutation on the GPU.
An independent CPU modular reference and GPU-NTT agree in all 12 adapter tests.

The A100 80 GB runs completed 52 Graph pairs and 16 eager diagnostics. All 68
pairs produce exactly equal integer ciphertexts. Graphs retain temporary
buffers and pass changed-input, restored-input and output-lifetime checks.
Q=[50,40,40], P=[50] or [50,50] and sec_level=none are explicit integration
parameters; these runs do not establish production security or a complete
bootstrap result. The full table and provenance links are in
`results/e2e_current_20261018/ckks_comparison.md`.

The adapter loses every measured pair. Method-I Graph key switching ranges
from 0.213x to 0.323x; method II ranges from 0.178x to 0.268x. For a single
ciphertext, the observed Graph kernel count increases from 14 to 60 in method I
and from 15 to 81 in method II. The adapter serializes individual moduli and
does not merge independent ciphertexts into one NTT batch. Its selected plans
also report unmeasured-feasible fallback rather than calibrated registry hits.
The kernel-count evidence establishes a concrete integration cost; attributing
the entire loss to calibration, local arithmetic or the theory would require
additional controlled measurements.

The next implementation priority is an RNS/ciphertext batch interface with
native root/order contracts and selection costs that include those boundaries.
TFHE standalone key switching contains torus decomposition and accumulation,
not a butterfly, so it remains an unchanged component rather than an NTT
speedup candidate. CKKS full bootstrap and TFHE fused-bootstrap treatment
remain separate unfinished application milestones.

## Native RNS boundary follow-up (2026-10-18)

`cuntt::RnsPlan` now reuses the existing Hybrid2D arithmetic with explicit
per-domain roots and per-polynomial addresses. Each pass launches across all
RNS domains, with negacyclic twist, inverse scaling and boundary order fused
into the original reads/writes. The HEonGPU adapter exposes `native-rns` and
keeps `grouped` as an explicit control. This is an additive C++ interface, not
yet a global RNS descriptor in the C API or installation cost-model registry.
Its request-local tuner measures splits, rows, threads, radix and cross-twiddle
placement outside timing and capture; `measured-rns-request` is distinct from
an installation calibration hit.

The same A100 80 GB UUID completed 52 native-RNS pairs and 52 grouped controls.
Method I covers N=4096/16384/65536 and B=1/8/64; method II covers N=4096/16384
and B=1/8. All ciphertext words agree exactly, including changed/restored Graph
input checks. The new path's kernel count equals the original in all 52 rows.
Its latency speedup over HEonGPU is 0.684–0.924x (geometric mean 0.777x) for
method I and 0.699–0.918x (0.810x) for method II. No new row wins. Relative to
the grouped control, cuButterfly itself improves 2.741x and 3.359x respectively.
These improvements combine RNS organization, boundary fusion and request-level
mapping tuning; they are not an isolated single-component ablation.

Raw CSVs, logs, artifact hashes and the full table live under
`results/e2e_rns_20261018/`. The four performance files use the same saved library
and benchmark binary. The node-level Nsight Systems trace is a separate
diagnostic run, excluded from speedup aggregates. Core verification covers
36 exact CPU-reference cases through logN=20, 12 GPU-NTT/CPU adapter cases,
and six legacy Hybrid2D checks after sharing the arithmetic template.

The removed per-modulus launches and standalone conversions no longer explain
the remaining deficit. The separate N=16384, B=8 trace attributes 97.4–97.7%
of the added summed GPU kernel time to the two replacement NTT passes. Their
combined duration is 1.46–1.57x the original NTT duration, with equal NTT launch
counts and matching non-NTT kernel signatures. Both passes contribute; the
trace does not distinguish arithmetic, boundary addressing, intermediate
traffic or mapping selection as the underlying cause. See
`results/e2e_rns_20261018/trace_analysis.md` and its read-only reproduction
script. Traced timings are excluded from the performance aggregates.
The application still
submits independent ciphertexts through scalar public calls; cross-ciphertext
stage batching, broader RNS candidate integration and full bootstrapping remain
unfinished. The negative application result neither establishes a theoretical
limit nor supports a claim of universal end-to-end acceleration.

## Fused-root core and boundary mapping follow-up (2026-10-18)

The RNS path already directly consumed and produced bit-reversed data. The
new work improves lane-to-address mapping and adds a parameterized fused-root
CT/GS core to request-local tuning; it does not add a missing global sorting
step. Wider row groups, compile-time direction and power-of-two indexing are
also available. The global RNS planner and installation model remain separate
unfinished work.

The final library completed 52 Graph pairs on A100 80 GB UUID
`GPU-0eae5f07-33de-321b-70a9-ffeae49309e6`. Method-I/II geometric mean speedups
over HEonGPU are 0.919x/0.899x, with ranges 0.796–0.991x/0.805–0.962x. All
ciphertexts agree exactly and kernel counts match, but no row wins. The earlier
52-pair native-RNS run used another physical 80 GB GPU; its absolute times must
not be used to calculate a version speedup. A separate intermediate 36-pair
run on the same GPU as the final run shows a 1.112x geometric mean improvement
in cuButterfly's own latency after direction specialization and renewed tuning.

The fresh N=16384, B=8 trace attributes 93.8–96.4% of added summed kernel time
to NTT execution, whose duration is 1.176–1.329x native. Tuning graphs are
excluded, and non-NTT launch signatures match. A separate 40 GB fixed-mapping
NCU control reduces bit-reversed global stores fourfold while increasing shared
read conflicts threefold. These results identify candidate coverage and core
efficiency issues; they do not establish a theoretical limit or hardware
saturation. Final validation passes 104 CPU-reference RNS cases, 12 adapter
cases and six legacy Hybrid2D regressions. See the
[complete report](../results/e2e_rns_coalesced_20261018/README.md) for raw data,
profiling limitations, library hashes and remaining scope.

## Register codelet and workload-distribution follow-up (2026-10-18)

FusedCTGS now admits a register-fused radix4 candidate alongside radix2, using
the same mapping validation, dispatch and request tuner. The 124 exact RNS and
12 adapter cases pass. All 52 CKKS Graph pairs remain ciphertext-exact with
matched kernel counts. On the same 80 GB GPU as the preceding fused-core run,
method-I/II geometric mean speedups improve to 0.961x/0.927x. Nine method-I
rows win; the N=65536 subset averages 1.016x, while N=16384 remains weaker.

Actual traces show only 2–12 polynomials per method-I NTT request. Ciphertext
batch repeats these scalar calls instead of increasing each NTT grid. In the
weakest scene, B=2 requests account for 43.5% of residual NTT kernel overhead.
The new trace assigns 90–96% of added kernel time to NTT, but native B=2
launches fewer warps than the selected replacement while running faster:
small-grid parallelism alone does not explain the backend gap.

Eighteen fixed-mapping controls now jointly warm both cores for at least 100ms
and alternate them in one process. Radix4 wins 11 and loses seven, supporting
optional selection. Earlier separate-process short-warmup controls remain
historical diagnostics with an explicit frequency-ramp confound. See the
[full workload and optimization report](../results/e2e_rns_workload_20261018/optimization.md)
for measurements, mechanism limits, launch geometry and unfinished batch/planner
integration. The saved research build is tested; the public conda installation
was not replaced in this milestone.

## Independent stage-mapping follow-up (2026-10-18)

The fused RNS candidate family now accepts local logs 2..10 and independent
per-pass rows/threads, exposing previously omitted 3+9/5+9 organizations.
All 148 exact RNS tests and 12 adapter tests pass. The new A100 40GB full
52-pair matrix averages 0.969x/0.925x for method I/II; all 12 method-I N=65536
rows measure above parity, averaging 1.050x. Four same-core, same-capacity
organization controls do not improve, so the native split is not forced.
One 80GB attempt is excluded because its metadata records another
project on the target GPU. A new runner guard blocks occupied starts, samples
occupancy during execution and excludes detected contention from summaries.
The [current report](../results/e2e_rns_workload_20261018/stage_mapping_optimization.md)
separates these stages and retains the unresolved application-context,
cross-ciphertext batching and global-planner work.

The target 80GB card subsequently became available. Guarded method-I/II full
reruns finish with 43/13 clean occupancy observations, 52 exact pairs and
0.972x/0.924x geometric means. The method-I N=65536 subset averages 1.055x
over 12 measured wins. The contaminated attempt remains excluded and unchanged.
