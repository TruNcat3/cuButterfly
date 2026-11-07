# E13 QuaRot FWHT Adapter

This directory contains the bounded E13 replacement for the pinned QuaRot
Hadamard call site.  The source checkout is
`results/e13_assets/sources/QuaRot` at commit
`5008669b08c1f11f9b64d52d16fddd47ca754c5a`.

## Replacement boundary

`QuaRotFWHTAdapter.matmul_hadU_cuda` is a drop-in replacement for the
`quarot.functional.hadamard.matmul_hadU_cuda` boundary.  The patch helper
changes only that symbol (including the package alias when it is available)
and restores it on exit.  Quantization, Linear4bit, attention, rotary
embedding, KV cache, and all other QuaRot functions remain outside this
adapter.

For an input with last dimension `n` and QuaRot factor `K`, the adapter:

1. validates `n % K == 0` and that `n // K` is a power of two;
2. runs a rank-1 cuButterfly FWHT over the flattened `K` sub-vectors;
3. multiplies by the pinned full-dimension scale `1 / sqrt(n)`;
4. applies the original `hadK.to(device).to(dtype) @ input` for `K > 1`;
5. restores the original shape.

The plan uses standard length and out-of-place execution.  No padding or
replacement of the `K=172` matrix is performed.  Plans are cached by dtype,
sub-transform length, flattened batch, device, and caller CUDA stream.  The
C handle stream is set from the current PyTorch stream before each execution.
The adapter supports FP16 and FP32 inference tensors only.

## Synthetic correctness gate

The gate loads only the pinned `functional/hadamard.py` file, so it does not
need to build QuaRot's private CUDA extension.  It compares cuButterfly with a
pure PyTorch radix-2 reference for:

- attention: `K=1`, `n=32`, FP16 and FP32;
- Llama-2-7B MLP decomposition: `K=172`, `n=11008`, sub-transform `64`,
  using the pinned `get_hadK(11008)` matrix, FP16 and FP32;
- one non-default caller CUDA stream;
- rejection of a non-power-of-two transform instead of implicit padding;
- patch scope and restoration of non-FWHT symbols.

Run it on the reserved GPU2 after confirming the UUID:

```bash
CUDA_VISIBLE_DEVICES=2 /home/wt/yes/envs/cubutterfly/bin/python \
  paper/experiments/e13/validate_quarot_adapter.py \
  --library build-e13-shared/libcuntt.so \
  --device cuda:0 \
  --expected-uuid GPU-0eae5f07-33de-321b-70a9-ffeae49309e6 \
  --output results/paper_profile_rc2/end_to_end/quarot_adapter_correctness.json
```

The checked-in report records the library and pinned-source hashes, device
identity, case tolerances, stream coverage, and outstanding dependencies.

To compare against the real Dao CUDA implementation used by the pinned
QuaRot function, add `--dao-reference`.  This calls the pinned
`matmul_hadU_cuda` directly, so its `fast_hadamard_transform.hadamard_transform`
path is exercised for both FP16 and FP32.  The default output is independent
of the synthetic report:

```bash
CUDA_VISIBLE_DEVICES=2 /home/wt/yes/envs/cubutterfly/bin/python \
  paper/experiments/e13/validate_quarot_adapter.py \
  --dao-reference \
  --library build-e13-shared/libcuntt.so \
  --device cuda:0 \
  --expected-uuid GPU-0eae5f07-33de-321b-70a9-ffeae49309e6 \
  --output results/paper_profile_rc2/end_to_end/quarot_adapter_dao_correctness.json
```

The Dao report records the installed package and CUDA extension paths/version
and retains the same four shape/dtype/stream cases.  It is still an operator
equivalence gate only; it is not a full QuaRot model or end-to-end result.

## Application hook

When the full QuaRot environment is available, use the context manager around
the existing application path without modifying the pinned checkout:

```python
from quarot_adapter import QuaRotFWHTAdapter, patched_quarot_hadamard

adapter = QuaRotFWHTAdapter(
    "build-e13-shared/libcuntt.so", compute_contract="public_fp32_final")
try:
    with patched_quarot_hadamard(adapter=adapter):
        output = existing_quarot_model(input_ids)
finally:
    adapter.close()
```

This example uses FP16 storage, FP32 computation, and fused final scaling and
rounding. The adapter's explicitly selected `native` and `fp32_composed` paths
remain available for historical comparisons; the application runner defaults to
`public_fp32_final` and Graph execution.

Alternatively, pass the imported `quarot.functional` module explicitly to
`patched_quarot_hadamard` or retain the object returned by
`install_quarot_fwht_adapter` and call `close()` after the run.

## Random-weight performance mode

The repaired runner initializes every Linear4bit with seeded Gaussian random
weights quantized symmetrically to signed int4 and positive per-row scales.
It records complete baseline/treatment state hashes and checks finite prefill
and every decode output before timing. Output comparisons are implementation
checks only. Supply `--atol` and `--rtol` explicitly to assess equivalence;
without both, the runner records errors as `numerical_tolerance_unassessed`
and may time finite outputs with `pass_timing_numerical_unassessed`; it does not claim numerical equivalence. RoPE uses actual attention position IDs
checked against the paged cache position, including layer-0 length updates.

CPU preparation probe (no vendor imports, CUDA, or 7B allocation):

```bash
/home/wt/yes/envs/cubutterfly/bin/python paper/experiments/e13/run_quarot_random_end_to_end.py \
  --prepare-check --output /tmp/quarot_random_prepare.json
```

GPU numerical report without timing (omitting tolerances intentionally leaves
equivalence unassessed):

```bash
CUDA_VISIBLE_DEVICES=2 /home/wt/yes/envs/cubutterfly/bin/python \
  paper/experiments/e13/run_quarot_random_end_to_end.py \
  --validation-only --prefill 32 --decode 2 --output /tmp/quarot_random_validation.json
```

FlashAttention remains the default and a missing dependency blocks execution.
Use `--attention-backend torch_sdpa` only to explicitly select the labeled
fallback. All validation and timing use inference mode; plan creation during
timed treatment calls causes failure. Timing requires finite outputs and exact same-backend Graph behavior checks. Cross-backend errors are report-only unless both tolerances are explicitly supplied. Variant order is explicit; the finite followup queue balances order across independent processes.

The earlier 2026-10-12 full 7B short-sequence validation confirmed matching state
hashes and finite outputs, but found large output differences. A separate
four-layer diagnosis found small local Hadamard differences amplified across
quantized layers, and unstable native decode outputs even without replacing
Hadamard. Neither result passes the application numerical gate. See the
[follow-up evidence record](../targeted_evaluation_followup_20261012.md).
`diagnose_quarot_random_numerics.py` records these local comparisons without
timing or changing precision; its reduced model is never a 7B result.

### Accumulation and warmup repair, 2026-10-12

The explicit `--compute-contract fp32_composed` option now matches Dao's FP32
accumulation and scale-before-FP16-writeback behavior. Inputs/outputs stay FP16.
The cuButterfly adapter converts to FP32, executes the existing FP32 FWHT,
applies the same FP32 scale, then converts to FP16 before the original K-factor
multiply. All conversions and scaling remain in the timed call. This is an
explicit composed compatibility path, not a fused FP16-storage/FP32-accumulate
kernel, and is recorded separately from the unchanged `native` option.

The independent CPU FP64 oracle found native cuButterfly local relative L2
5.37e-4--6.14e-4 versus Dao 2.07e-4--2.94e-4 on actual four-layer inputs.
After accumulation/scaling alignment, all 16 same-input Hadamard comparisons
and both reduced-model outputs were bitwise equal to Dao.

An isolated int4 KV-cache probe traced the native repeat discrepancy to
TorchScript quantization execution changing during first use. Disabling JIT
optimization for diagnosis made cached bytes exact and repeats deterministic;
warming the unchanged optimized model four complete sequences also made its
repeats bitwise equal. The runner retains optimization and adds explicit
validation warmup plus a zero-tolerance baseline repeat check. This does not
assert that all upstream cache workloads are now verified.

Full 7B validation on A100 40GB, batch1/prefill512/decode32, then passed
`--atol 0 --rtol 0`: all 33 output groups and baseline repeats were bitwise
equal, and model hashes matched. Evidence is
`results/paper_targeted_20261012/quarot/full512_validation.json`.
This 40GB validation is not an 80GB timing result. The repaired A100 80GB
timing run is separate, with 5 warmups and 10 calls per metric; variant order
is explicit via `--variant-order`. Formal speedups require its own passed gate.
Both paths use the labeled SDPA prefill fallback; decode remains the pinned
QuaRot paged-cache kernel. A model checkpoint is unnecessary for this random
state performance comparison, which carries no model-quality claim.

For a new hardware or workload point, pass
`--algorithm-policy measure` to perform bounded local FWHT candidate tuning
during setup. The runner records the policy and each plan's selection source;
the timed calls begin only after setup. A final calibrated replay should use
the resulting cache or registry and report `user_cache` or `measured_table`.
A `static_model` source is an explicit selection miss, not a calibrated
application result.

The two independent A100 80GB timing processes completed with the baseline
and treatment order exchanged. Both used the `fp32_composed` contract, zero
tolerance, five complete-sequence warmups, and ten calls per metric. Their
median GPU times were:

| Work | Baseline p1 / cuButterfly p1 (ms) | Baseline p2 / cuButterfly p2 (ms) |
| --- | ---: | ---: |
| Prefill 512 | 75.01 / 78.01 | 75.39 / 79.07 |
| Decode 32 | 2385.42 / 2430.00 | 2430.46 / 2480.06 |
| End-to-end | 2460.34 / 2509.74 | 2505.40 / 2556.89 |

The baseline/cuButterfly end-to-end ratios are 0.9803 and 0.9799. This
repaired path is therefore about 2% slower at this workload; it is a valid
correctness and overhead result, not an end-to-end speedup. Plan creation and
first use are outside the timed calls, while the FP32 conversion and scaling
composition are included. The complete reports are
`results/paper_targeted_20261012/quarot/full512_timing_p1.json` and `p2.json`.

### CUDA Graph stream contract, 2026-10-19

Graph execution requires the isolated current-stream QuaRot extension for
both backends. The pinned original launches private quantization, int4 GEMM
and paged-cache kernels on the legacy default stream. Successful capture
alone does not establish that those launches were captured: the original
`sym_quant` probe replayed zero nonzero bytes instead of 8192, while the
current-stream port replayed all 8192 and matched eager exactly. The failed
2026-10-18 B1/128/8 report selected `pinned_original` and omitted this port.

The runner now automatically selects
`results/e13_assets/quarot_graph_stream_build/_CUDA_graph.so` for Graph mode.
An explicit `--quarot-extension` overrides this location. Either path requires
an adjacent `build_receipt.json` with the current-stream protocol and matching
binary SHA256; a missing or mismatched receipt blocks execution. Build or
verify the reproducible port before GPU execution:

```bash
/home/wt/yes/envs/cubutterfly/bin/python \
  paper/experiments/e13/build_quarot_graph_extension.py --prepare-only
```

Omit `--prepare-only` to build it when needed. Graph validation always requires
zero error against the same backend's eager outputs for original and changed
inputs, a repeated restored input, and outputs retained across replay. These
checks apply even when cross-backend floating-point tolerances are unset.
KV metadata is prepared per sequence position before capture; each replay
resets the Python cache state and re-executes prefill and all decode updates.
Input copies and independent output clones are included in both backends'
timed complete calls. FWHT setup and bounded tuning are excluded.

```bash
CUDA_VISIBLE_DEVICES=GPU-2971e500-d3f6-d7e6-e5f2-cb67a06fff02 \
CUBUTTERFLY_COMPILE_MODE=research \
CUBUTTERFLY_SEARCH_BUDGET=4 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
CUBUTTERFLY_PLAN_CACHE="$PWD/results/e2e_alignment_20261019/quarot/plan_cache.json" \
/home/wt/yes/envs/cubutterfly/bin/python \
  paper/experiments/e13/run_quarot_random_end_to_end.py \
  --library build-contract-20261017/libcuntt.so \
  --batch-size 1 --prefill 128 --decode 8 --warmup 2 --calls 5 \
  --attention-backend torch_sdpa --compute-contract fp32_composed \
  --algorithm-policy measure --execution-mode graph \
  --output results/e2e_alignment_20261019/quarot/full7b_b1_p128_d8_graph.json
```

`diagnose_quarot_graph.py --layers 2` accepts the same runner arguments for
bounded diagnosis. Its reports explicitly mark the reduced model and cannot
serve as full 7B evidence. Reports without application tolerances retain
`numerical_unassessed` status and the complete cross-backend error distribution;
the independent Graph behavior checks remain exact.

The 2026-10-19 full 32-layer random 7B B1/128/8 run used this automatic
extension selection and a four-candidate search budget. All nine baseline
repeat output groups, all cross-backend output groups, and both backends'
original/changed/restored-repeat/output-lifetime Graph checks had zero error.
With no application thresholds selected, the report intentionally retains
`pass_timing_numerical_unassessed`. Five complete Graph calls after two warmups
measured median wall latency 154.17 ms for Dao/QuaRot and 159.45 ms for
cuButterfly (0.9669x); CUDA event medians were 154.14 and 159.42 ms.
This point is a performance negative result, with compilation/tuning outside
timing and FP32 composition, input copies and output clones included.
Evidence: `results/e2e_alignment_20261019/quarot/full7b_b1_p128_d8_graph.json`.
The earlier exhaustive-setup attempt was interrupted before measurement and
is separately recorded in `interrupted_setup.json` in the same directory.
The matched B4/128/8 completion also passed every exact Graph behavior check
and had zero cross-backend error. Median wall latency was 177.54 ms for the
baseline and 185.60 ms for cuButterfly (0.9565x); CUDA event medians were
177.51 and 185.54 ms, with peak allocations of 4.51 and 4.64 GB. Evidence:
`results/e2e_alignment_20261019/quarot/full7b_b4_p128_d8_graph.json`.

The historical random timings below were obtained before this initialization
and RoPE repair and do not satisfy the new numerical gate. They must not serve
as validated end-to-end performance evidence.

For the application latency comparison, an external checkpoint is not
required.  The baseline and treatment use matching hashed,
deterministically initialized random states with seed `20261003`; input token
IDs, prefill/decode lengths, KV-cache state and variant order are also frozen.
This mode reports only GPU-resident prefill, decode, combined latency and peak
memory.  It must not be used for perplexity, logits, token agreement or any
model-quality claim.

The random mode removes the checkpoint-authorization blocker.  The shared
environment has the QuaRot CUDA extension and pinned Transformers runtime; the
upstream FlashAttention build is still separate, so the checked-in random
timing uses an explicitly labeled PyTorch SDPA fallback.  The fallback is a
controlled latency experiment, not a replacement for the upstream attention
kernel in a final headline comparison.

## Historical E13 execution before the 2026-10-12 repair

The earlier correctness reports establish only the synthetic and Dao CUDA
operator gates. The application timings below predate the repaired random
state and full-model numerical checks, and require replacement. Any future quality statement
must use a separately authorized frozen checkpoint and must not reuse the
random-weight performance record.

The earlier random full-model smoke reached the application path using
the pinned 7B architecture, seed `20261003`, and a PyTorch SDPA attention
fallback because FlashAttention is not available in the shared CUDA 12.4
environment. The environment had the pinned Transformers 4.38 runtime and a
locally built QuaRot `_CUDA` extension; the fallback is marked in the report.
The invalid random state prevents using these timings as application evidence.
Its small payload (`prefill=32`,
`decode=2`) measured 74.17 ms versus 72.47 ms for prefill, 163.83 ms versus
162.45 ms for decode, and 238.44 ms versus 235.58 ms for the combined call.
The four cuButterfly FWHT plans took 448.7 ms to create and were excluded from
steady-state timing. The `512/32` and `2048/128` payloads were also executed
under this fallback, without the new numerical gate. For `512/32`, combined latency is 2668.60 ms
versus 2648.06 ms (1.008x); for `2048/128`, it is 10573.72 ms versus 10544.19
ms (1.003x).  Both reports exclude roughly 0.45 s of FWHT plan creation from
steady-state timing.

## Fused public-contract followup, 2026-11-07

The runner now defaults to `--compute-contract public_fp32_final` and
`--execution-mode graph`. This is `sameFP16storageFP32compute`: FP16 boundary
storage with FP32 FWHT compute and final writeback. The existing K-factor
matrix multiply remains unchanged and is included in the complete call;
the FP32 label does not describe that multiply or the rest of the model.
Explicit `native`, `fp32_composed`, and `eager` options replay older contracts.
Both synchronized host wall time and CUDA events are retained; summary ratios
use median wall times for the complete device-resident sequence, host replay,
input refresh, output ownership copies, and synchronization. Setup is excluded.
No application tolerance is selected by the followup; errors remain report-only.

The fixed config `configs/experiments/quarot_fused_followup_20261107.json`
contains exactly B1/B4 × baseline-first/treatment-first, prefill128/decode8,
seed20261003, validation warmup4, timing warmup2, **7 calls**, Graph and the
public FP32 contract. All four jobs share one explicit timezone-aware deadline.
Prepare hashes files and prints commands without importing Torch or creating a
model. Supply your built library and current-stream extension paths:

```bash
python scripts/run_quarot_fused_followup.py --mode prepare \
  --library "$QUAROT_LIBRARY" --quarot-extension "$QUAROT_EXTENSION" \
  --output results/quarot_fused_followup_20261107
python scripts/run_quarot_fused_followup.py --mode run \
  --library "$QUAROT_LIBRARY" --quarot-extension "$QUAROT_EXTENSION" \
  --gpu-uuid "$QUAROT_GPU_UUID" --until "$QUAROT_DEADLINE_ISO" \
  --output results/quarot_fused_followup_20261107
```

Run with the research Python environment. Each job uses `run_isolated_gpu.py`
and a fresh job directory. Compatible clean releases are skipped; mismatched
releases are rejected. Interrupted directories remain evidence and require a
fresh output root. The deadline is never extended automatically.

New execution identities also bind selection environment values, registry and
static profile contents, and the ranker inputs. Changing these inputs rejects
reuse. This finite queue does not accept a mutable custom plan cache or the
unsupported runtime overrides `CUBUTTERFLY_COST_MODEL` and
`CUBUTTERFLY_HARDWARE_PROFILE`; use the hardware-model/profile-root interfaces.
Older identities remain readable as historical evidence but cannot authorize
new execution under an unknown environment.
