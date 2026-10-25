# E13: Open-Source End-to-End Preparation

This document freezes one real open-source application per butterfly family. It
is a preparation record, not a performance or quality result. No application,
GPU kernel, model, or external dataset was run in this preparation pass.

The machine-readable contract is
[`experiments/end_to_end.json`](experiments/end_to_end.json), and the CPU-only
checker is [`tools/prepare_end_to_end.py`](tools/prepare_end_to_end.py).

Execution began on 2026-10-02. The original preparation record below is retained
for provenance. Current measured results and outstanding conditions are tracked
in [`e13_progress.json`](../results/paper_profile_rc2/end_to_end/e13_progress.json),
with reproducible [FNO execution conditions](experiments/e13/README_fno.md),
[QuaRot adapter gates](experiments/e13/README_quarot.md), and
[HEonGPU build status](experiments/heongpu_e13_build_status.md).
Passing an adapter gate or building an application does not complete its
end-to-end comparison.

The supplemental CKKS/TFHE workload contracts are tracked separately in
[`ckks_tfhe_workloads.json`](experiments/ckks_tfhe_workloads.json). The pinned
HEonGPU TFHE native reference has now been measured with complete public gate
and circuit calls. It remains supplemental until the cuButterfly replacement
passes the same exact-bit semantic gate; a missing fused replacement blocks
only that treatment result, not the native reference.

The [CKKS complete-operation comparison](experiments/README_ckks.md) covers key
switching, square/relinearize/rescale, square followed by key switching, and
slot rotation. The latest RNS candidates add short local stages and independent
per-pass mappings alongside register-fused radix4. A100 40GB and 80GB each
complete 52 CUDA Graph pairs with exactly equal integer ciphertexts and matching
kernel counts. Native/adapted latency geometric means for method I/II are
0.969x/0.925x on 40GB and 0.972x/0.924x on 80GB. All 12 method-I N=65536 rows
per card measure above parity, averaging 1.050x/1.055x respectively; method II
has no winning row. The [80GB report](../results/e2e_rns_workload_20261018/ckks_stage_mapping_80gb_comparison.md),
[40GB report](../results/e2e_rns_workload_20261018/ckks_stage_mapping_full_comparison.md) and
[mechanism diagnosis](../results/e2e_rns_workload_20261018/stage_mapping_optimization.md)
retain all pairs, negative controls and remaining deficits. An earlier 80GB
attempt encountered external GPU work and remains excluded; subsequent monitored
reruns provide the current 80GB results.
The valid preceding [80GB radix4 result](../results/e2e_rns_workload_20261018/ckks_comparison.md)
remains separate at 0.961x/0.927x, and is not mixed into a cross-device version comparison. The
[previous fused-core report](../results/e2e_rns_coalesced_20261018/README.md),
[earlier RNS/grouped comparison](../results/e2e_rns_20261018/ckks_comparison.md)
and [first report](../results/e2e_current_20261018/ckks_comparison.md) remain
separate historical evidence, including their hardware identities. These are complete CKKS operation boundaries,
not the complete regular-bootstrap or TFHE replacement.

## Selection

| Operator | Application and pinned source | Complete timing boundary | What remains unchanged |
| --- | --- | --- | --- |
| FFT | NeuralOperator FNO Darcy flow, commit [`00b7d86f`](https://github.com/neuraloperator/neuraloperator/tree/00b7d86f8d74ff0af55da53eb585fe26df9c71f0), MIT | GPU-resident complete FNO forward for a fixed device batch; dataset fetch/preprocess, H2D, postprocess, D2H, setup, and training are separate | `complex_data=false`, `rfftn` half-spectrum, non-last-axis shift, `irfftn` shape/normalization, model and checkpoint |
| FWHT | QuaRot Llama-2-7B int4 prefill/decode, commit [`5008669b`](https://github.com/spcl/QuaRot/tree/5008669b08c1f11f9b64d52d16fddd47ca754c5a), Apache-2.0 | GPU-resident full-model prefill/decode with the same KV-cache behavior; token preparation, host I/O, setup, and weight download are separate | Quantizer, int4 GEMM, flash attention, KV cache, `get_hadK`, `hadK` matrix, normalization, and dtype/device behavior |
| NTT | HEonGPU CKKS regular bootstrapping, commit [`1928a14b`](https://github.com/Alisah-Ozcan/HEonGPU/tree/1928a14bb002c5d7f1cb5534eca9fa19b555b4f2), Apache-2.0 | GPU-resident complete `regular_bootstrapping` call, including all accounted compatible NTT dispatches; ciphertext host I/O, keygen, encryption, decryption, decode, and context setup are separate | CKKS stages, RNS limbs, roots, `X^N+1`, per-polynomial layout, inverse scaling, keys, metadata, and caller CUDA stream |

The NTT choice is HEonGPU rather than PIRonGPU because its pinned source exposes
the CKKS regular-bootstrapping call and the GPU-NTT boundary directly. This is
an application-level CKKS bootstrapping workflow, not a claim about a complete
FHE service or model.

## Source and semantic contracts

### FFT: FNO Darcy flow

The upstream end-to-end example is
[`examples/models/plot_FNO_darcy.py`](https://github.com/neuraloperator/neuraloperator/blob/00b7d86f8d74ff0af55da53eb585fe26df9c71f0/examples/models/plot_FNO_darcy.py);
the configurable training entry is
[`scripts/train_darcy.py`](https://github.com/neuraloperator/neuraloperator/blob/00b7d86f8d74ff0af55da53eb585fe26df9c71f0/scripts/train_darcy.py).
The pinned `DarcyDataset` supports resolutions 16, 32, 64, 128, and 421 and
uses the open Zenodo record [12784353](https://zenodo.org/records/12784353),
licensed CC-BY-4.0. The record metadata and archive hashes for `darcy_64.tgz`
(187103764 bytes, MD5 `49027fc1972647915f6c0893a79bbc59`) and `darcy_128.tgz`
(373869283 bytes, MD5 `43e85ec9bff754d0f6d3925e26d5a8ac`) were checked without
downloading the archives. These 64x64 and 128x128 cases are the two main
payloads, with 1000 training and 100 held-out samples. The 16x16 training and
16-to-32 transfer paths remain adapter smoke cases only. No high-resolution
checkpoint has been acquired or fixed yet.

The default FNO path is real data: `complex_data=false`,
`torch.fft.rfftn`, `fftshift` on all but the last spatial axis, and
`torch.fft.irfftn` with the original `s` and normalization. The last spectral
dimension is `n//2+1`, and the implementation has an explicit Hermitian
symmetry path. cuButterfly is currently mainly C2C, so a real-FFT adapter is a
required implementation item. Setting `complex_data=true` would change the
baseline and is invalid.

### FWHT: QuaRot Llama-2-7B

The pinned upstream harness is
[`e2e/benchmark.py`](https://github.com/spcl/QuaRot/blob/5008669b08c1f11f9b64d52d16fddd47ca754c5a/e2e/benchmark.py).
Its default quantized constructor uses `no_init_weights()`, which is not a
reproducible random state and may not be used directly. For the performance
experiment, construct one deterministically initialized random state with seed
`20261003`, clone it into the baseline and treatment, and report latency,
memory and traces only. A separate quality experiment must use one authorized,
frozen Llama-2-7B quantized checkpoint for both variants and public WikiText-2
validation data. Upstream int4-versus-FP16 output is not the causal comparison
here.

The replacement is exactly the power-of-two call inside
`quarot.functional.matmul_hadU_cuda`. The `get_hadK` decomposition and small
matrix multiplication remain upstream. For Llama-2-7B, attention uses the
K=1 path; the MLP intermediate dimension 11008 is `172 * 64`, so the treatment
must run FWHT on the 64-wide subpath and retain the 172x172 factor. Zero-padding
11008 to a power of two, replacing the factor matrix, or dropping quantization
would violate the application contract.

The two timing cases are batch 1 with prefill/decode `(512, 32)` and
`(2048, 128)`. The latter matches the upstream default prefill size. The
primary metric is GPU-resident full-model prefill/decode after setup; token
preparation and host transfers are reported separately. Model access
authorization, extension build, quality evaluation, and GPU capacity remain
pending.

### NTT: HEonGPU CKKS regular bootstrap

The official entry is
[`1_ckks_regular_bootstrapping.cpp`](https://github.com/Alisah-Ozcan/HEonGPU/blob/1928a14bb002c5d7f1cb5534eca9fa19b555b4f2/example/bootstrapping/1_ckks_regular_bootstrapping.cpp).
The regular path in
[`operator.cu`](https://github.com/Alisah-Ozcan/HEonGPU/blob/1928a14bb002c5d7f1cb5534eca9fa19b555b4f2/src/lib/host/ckks/operator.cu)
uses the pinned GPU-NTT submodule at commit
[`8a4daf11`](https://github.com/Alisah-Ozcan/GPU-NTT/tree/8a4daf11c7cdea8dde45fa0f8f7183c138d42b3b).

The contract is per-RNS-limb word arithmetic in the negacyclic ring
`Z_q[X]/(X^N+1)`, `gpuntt::PerPolynomial`, `zero_padding=false`, per-modulus
forward/inverse roots, and inverse multiplication by `N^{-1}`. The
`ExecutionOptions.stream_` value reaches both NTT configurations and surrounding
kernels; an omitted option means `cudaStreamDefault`. The API does not make a
separate natural-order promise, so order must be frozen by a pinned GPU-NTT
reference test before replacement. cuButterfly's word64 boundary is one
modulus below `2^63`; generic multi-limb arithmetic is out of scope.

The small case is the official N=4096 demonstration (functional but explicitly
insecure parameters) and is smoke-only. N=16384 is retained only as a resource
exploration candidate, not as the selected main case: its security level,
multiplicative depth, complete Q/P chain, and memory feasibility require an
audit, and a different N may be selected. Copying the N=4096 list is not
allowed.

The replacement boundary covers the complete call graph
`regular_bootstrapping -> coeff_to_slot -> multiply_matrix` or
`multiply_matrix_less_memory` -> `exp_scaled` -> `slot_to_coeff`, including
nested key-switch and transform helpers. `GPU_INTT`/`GPU_INTT_Inplace`,
`GPU_NTT`/`GPU_NTT_Inplace`, and ordered NTT variants must be counted per run.
Only dispatches whose word64 modulus, roots/order, RNS layout, inverse scaling,
and stream contracts are proven equivalent may be replaced. Ordered variants
remain native-retained or unsupported until audited; `SmallForwardNTT` and
`SmallInverseNTT` are recorded separately if a trace reaches them. Any missing
or unsupported required dispatch makes the run a partial NTT ablation, not a
complete replacement claim.

## Preparation and blockers

The default checker performs no network access, GPU query, build, model download,
or application execution:

```bash
python paper/tools/prepare_end_to_end.py \
  --manifest paper/experiments/end_to_end.json \
  --output results/end_to_end_readiness.json
```

It validates pinned commit/license fields, one application per operator, two
payloads per application, semantic guardrails, and the NTT modulus boundary.
It emits every source/reference command as a `not_run` checklist entry. Any
network command is invoked separately as part of application setup and is
never run by the checker. This separation adds no approval requirement when
the application setup has already been authorized.

Before any GPU result can be recorded, the following blockers must be closed:

1. FNO needs the verified 64x64/128x128 archives, a frozen trained checkpoint,
   and a real-FFT/Hermitian adapter; baseline and treatment must use identical
   data, parameters, and checkpoint.
2. QuaRot performance mode needs a deterministic random-state runner, the
   pinned runtime/CUDA extension, and an adapter that changes only the
   power-of-two FWHT subpath. Authorized weights and frozen tokenizer/WikiText-2
   hashes are required only for quality claims.
3. HEonGPU needs a build and per-limb root/order/stream equivalence test; the
   resource candidate additionally needs a safe security/depth-complete Q/P
   chain and memory audit before any main payload is selected.

No blocker is converted into a measured result by this preparation record.

## Bounded measurement protocol

The manifest proposes three process trials, five complete warmup calls per
variant, and ten complete timed calls per variant and trial. Baseline/treatment
order is balanced with a frozen seed. These are proposed application trials,
not the existing operator matrix's 100/100 timing protocol. The primary metric
is synchronized wall time for the complete declared operation with inputs
already on the GPU. It includes per-call adaptation, allocation, CPU dispatch,
and synchronization. CUDA-event spans are secondary and must join all relevant
streams. Host input/output and reusable preparation are reported separately.

Every call must start with equivalent ciphertext, input, and KV state. An
application-required reset or allocation stays in both variants' timed paths.
Report the raw samples, three trial medians, their median, peak memory, and
quality for each application separately. Separate traces explain butterfly
time, adaptation cost, launches, and traffic. Application time is measured
directly and never replaced by a sum of profiled kernels. No performance result
or fully selected NTT main parameter set is supplied by this protocol.

## Paper-ready conservative summary

We selected three bounded open-source application paths for a first end-to-end
study: 64x64/128x128 real-input Darcy FNO from NeuralOperator, int4 Llama-2
inference from QuaRot, and CKKS regular bootstrapping from HEonGPU. Each path is
pinned to an audited repository commit and has two finite first-round payload
definitions; 16x16/32x32 FNO and N=4096 CKKS are smoke-only, while the NTT
large case remains resource exploration until a safe main payload is selected.
The FFT comparison must preserve
NeuralOperator's real `rfftn`/`irfftn` contract, including its half-spectrum and
Hermitian handling. The FWHT comparison replaces only QuaRot's power-of-two
Hadamard subpath; quantization, GEMM, and the non-power-of-two `172 x 64`
factorization remain unchanged. The NTT comparison covers the complete CKKS
bootstrap call and accounts for every compatible NTT dispatch while keeping
HEonGPU's negacyclic RNS, root, layout, normalization, and stream contracts.
Checkpoints, model authorization, parameter audits, adapters, GPU execution,
and quality measurements remain pending; this record therefore makes no
application speedup or quality claim.
