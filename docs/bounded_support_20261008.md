# Bounded semantic support milestone, 0.9.3

The user requested implementing the missing arithmetic lowerings and excluding
unreasonable size/batch combinations. This milestone changes the executable
library and the declared test population. It is not a completed performance
campaign or a proof of globally optimal mappings.

## Declared workload limits

`scripts/prepare_bounded_semantic_retest.py` prepares configuration, manifest and
`policy.json` without GPU work. The default scaling budget is **2^20 logical
elements per request**, including all batches. Individual transform caps are
2^20 for 1D, 1024x1024 for 2D, and 64x64x64 for 3D. These are experiment
choices, not library limits. The total budget can be changed explicitly with
`--max-elements`; the point caps remain declared in the preparation policy.

| Shape | Explicit batch choices |
| --- | --- |
| 256 | 1, 16, 256, 4096 |
| 4096 | 1, 16, 256 |
| 65536 | 1, 16 |
| 1048576 | 1 |
| 64x64 | 1, 16, 256 |
| 256x256 | 1, 16 |
| 1024x1024 | 1 |
| 16x16x16 | 1, 16, 256 |
| 64x64x64 | 1 |

Twenty of the previous 44 scaling shape/batch combinations remain; 24 are
excluded with explicit reasons. Selection uses only shape and batch, never
measured speed, baseline availability or current kernel support. No request
has its batch silently reduced. The semantic crossing, arbitrary-length and
embedding populations retain their original precision, direction, placement,
order and layout dimensions. Rank-eight/rank-nine boundary cases are separate
correctness/support checks and are not a performance population.

The generated manifest has 38,416 unique valid semantic requests. Its 232
static missing-lowering records are exclusively rank-nine boundary negatives;
the public execution bound remains eight. Static eligibility is not measured
correctness or resource feasibility. The 17,056 invalid Cartesian variants are
retained in the preflight audit, outside the valid-request count.

## Executable additions

| Previous gap | 0.9.3 implementation and scope |
| --- | --- |
| UINT64 subset/superset zeta | Native unsigned 64-bit arithmetic, forward/Möbius inverse, two-word warp exchange, xD resident dispatch, exact tuning checks and eight-byte resource accounting |
| FP32 storage with FP64 compute | Explicit cast to FP64 working buffers, matching FP64 child, cast back to original storage; public order/layout boundaries remain composable |
| FP16/BF16 real FFT | Explicit full-spectrum adapter using the same requested low complex storage and compute contract; power-of-two axes of length at least two |
| Compact real FFT rank four through eight | Existing FP32/FP64 per-axis real/complex composition extended to the shared rank bound |
| Standard arbitrary-length FFT beyond rank two | Exact-length rank-one child plans and generic per-axis permutations; no replacement by zero-padded mathematical transforms |

Numeric adapters report `kind=numeric-adapter`, child mapping, workspace and
conversion-inclusive execution. They are classified as adapted in campaign
results. Full-spectrum low real execution is not presented as a native compact
real kernel. Additional real output scaling follows the requested native or
FP32 arithmetic contract. Explicit mapping replay includes the adapter.

Default high-rank real planning would otherwise enumerate 7! axis orders at
rank eight. For more than three complex axes, its bounded candidate set is
cyclic rotations of forward and reverse traversal. Explicit replay accepts
any legal order. This is a search-budget policy, not a semantic approximation
or exhaustive-search claim.

## Reference and runner changes

Power-of-two NTT now uses an independent exact integer radix-two CPU reference.
Python arbitrary-width products avoid UINT64 modular multiplication overflow.
Small direct-DFT comparisons check roots, directions and multidimensional
composition. Non-power-of-two requests keep the direct reference. The default
16,777,216-operation budget counts butterflies for the radix-two path and DFT
terms otherwise. The declared power-of-two population needs at most 10,485,760
butterflies, so ordinary NTT loads no longer hit the former quadratic cap.

All outputs are still checked; no random sampling or batch reduction is used.
The real FFT runner now encodes each side using its actual scalar/complex
storage type. FP16/BF16 C2R references use the final quantized spectrum, and
low-precision R2C reference arrays retain their imaginary components.

## Verification and evidence

Target: A100 80GB PCIe, UUID
`GPU-2971e500-d3f6-d7e6-e5f2-cb67a06fff02`.
Library SHA256:
`78248f2c2e3a6965e0b1c5ba3c82229f09289672f7b158e56d4d9ea183bc03f2`.
Local artifacts are under `results/semantic_support_20261008_v0923/`.

- `gpu-contracts-r2.xml`: 334 checks completed, comprising 332 GPU cases and
  two CPU oracle checks. Covers new precision/word-width paths, rank4/rank8
  real FFT, arbitrary rank3 FFT, directions, normalization, independent layouts,
  placement, padding/input protection and exact mapping replay.
- `runner-smoke/` and `runner-smoke-r2/`: 13 unique requests executed through
  the actual campaign runner, including all-element N=65536 NTT for both
  word widths. Six integer requests pass exactly; seven floating requests
  remain `accuracy-review-required`. Baseline gaps remain explicit.
- `cpu-validation.xml`: 194 passed checks and 28 subtests covering the bounded
  population, independent exact NTT oracle, codec, descriptor, frontend and
  continuation protocol.
- The standalone UINT64 reference executable checks direct combinatorial
  sums, subtraction, wraparound, and forward/inverse round trips.

The first GPU test attempt incorrectly supplied a nonnull workspace to a
zero-workspace plan. The first runner test also incorrectly expected native
FP32 compact real execution to be labelled adapted. Both were test assertions,
were corrected, and the failed artifacts remain alongside successful reruns.

Numeric-adapter and arbitrary-axis diagnostics retain complete arrays. In the small low-real
numeric test set, the maximum observed relative L2 error was 0.03572% for FP16
and 0.15076% for BF16. These are bounded diagnostic observations, not adopted
precision thresholds or large-workload accuracy claims. Historical unscaled
FP16 overflow cases and the separately authorized scaled controls remain intact.

## Campaign transition and remaining work

The prior queue was stopped between shards, with no active measurement child,
after 35,343 records. Its manifests, failures and library were preserved.
`transition.json` and `previous_build/` record the transition. A new binary or
reference protocol is never appended to the old measurement identity.

The new bounded population is prepared; its full GPU conformance and performance
comparison have **not** been completed. The changed binary needs a fresh
calibration/measurement binding before comprehensive performance comparison.
No old whole-plan ranking is automatically certified for new adapters. Public
baseline bindings also remain narrower than library semantics, especially
low-precision real FFT and mixed compute. Missing baselines are not replaced
by higher precision, another operator or a different batch.

Preparation command, using a fresh output directory:

```bash
/home/wt/yes/envs/cubutterfly/bin/python scripts/prepare_bounded_semantic_retest.py \
  --library build-e13-cufftdx/libcuntt.so --output results/bounded-semantic-retest
```
