# Benchmark contracts

`benchmark/` is the canonical index for benchmark scope. Existing workload
JSON files and historical results remain in their original locations; this
directory records which one is authoritative for each evidence track.

Start with [`scopes/a100_80gb_research_v1.json`](scopes/a100_80gb_research_v1.json)
and validate it before starting or summarizing a run:

```bash
python scripts/validate_benchmark_scope.py benchmark/scopes/a100_80gb_research_v1.json
```

The current paper-oriented evidence boundary and remaining measurements are
listed in [`research_evidence_gap_20261017.md`](research_evidence_gap_20261017.md).
The current application-level positive, near-parity, and negative rows are
summarized in [`application_gap_review_20261021.md`](application_gap_review_20261021.md).
The [October 19 convergence record](research_convergence_20261019.md) fixes the
remaining CKKS automatic-selection acceptance set and separates its completion
from the outstanding paper and portability evidence.

New reports follow the repository-wide
[reporting and evidence contract](../docs/reporting_contract.md): scope and
question, protocol, complete result, attribution/limitations, and reproduction
metadata are kept together. The contract also defines the `measured`,
`derived`, `external`, `unsupported`, `diagnostic`, and `placeholder` labels
used when a report is summarized for the paper.

The scope preserves the complete semantic identity of a request: operator,
precision and accumulation, shape/rank, batch, direction, normalization,
input/output order, placement, layout and strides. Baseline gaps are recorded
as gaps. A runner must not substitute a smaller batch, different precision,
flattened rank, or unrelated operator.

Each result should identify `scope_id`, `track_id`, `case_id`, population kind
(`native`, `adapted`, or `cpu`), GPU UUID, library/build hash, and timing
protocol. Native vendor results and real adapter paths are separate cohorts.
Compilation and plan creation stay outside steady-state execution timing, while
required conversion, permutation and normalization stay inside it.

Create a new scope revision for a changed population. Do not edit an old scope
to make an existing result table appear to cover a new matrix.

## Unified campaign manifest

`benchmark/manifest.py` provides a runner-neutral entry point for new campaigns.
It expands three scopes (`operator`, `e2e`, and `profile`) while preserving the
semantic axes used by the existing suites.  A case may constrain `sizes`,
`batches`, `precisions`, `directions`, `layouts`, `operators`, and `scenes`.
Each target is an explicit `(backend, provider, execution)` tuple, so a CPU
baseline is never accidentally paired with a CUDA provider by a Cartesian
product.  The root `execution` object and per-target override record whether
GPU and CPU runs are requested.  The parser rejects unknown fields, unknown
case ids, invalid targets, and empty selections before any runner is started.

The bounded and broad examples are [`config/benchmark_quick.json`](../config/benchmark_quick.json)
and [`config/benchmark_full.json`](../config/benchmark_full.json).  The former
declares the `quick` mode and the latter declares the `full` mode, so selecting
the mode also selects the intended matrix size.  Validate and emit an auditable
case manifest with:

```bash
PYTHONPATH=. python -m benchmark.manifest \
  config/benchmark_quick.json --mode quick \
  --operator ntt --size 4096 --batch 1 \
  --output results/benchmark/quick/ntt.json
```

The output contains the resolved singular fields (`operator`, `size`,
`batch`, `precision`, `direction`, `layout`, `scene`, `backend`, and
`provider`) for every case.  Existing CUDA and CPU runners can consume this
file; the manifest layer does not select a device, compile a kernel, or alter
timing policy.  Compilation, search, and execution timing remain controlled by
the runner-specific contracts above.

For a campaign that must configure and execute heterogeneous binaries from one
file, use the schema-2 runner registry in
[`config/benchmark_weak_regions_20261022.json`](../config/benchmark_weak_regions_20261022.json):
`plan`, `ntt`, and `gpuntt` are declared under `runners`, while each case
declares its semantic input and optional `baselines`.  The corresponding
`scripts/benchmark_weak_regions.py` derives the command-line representation,
checks the configured executable, runs the primary and baseline calls, and
writes paired timing and correctness records.  This keeps executable-specific
flags out of the case definition.  A baseline that is unavailable is recorded
as such; it is never silently substituted with another shape or precision.
Each case may set `execution_mode` explicitly.  The default is `auto` for a
rank-one transform and `xd` for a multidimensional transform; forcing `xd` on
rank one is rejected because the public xD policy is multidimensional.  The
manifest-level `search_budget` is forwarded as `CUBUTTERFLY_SEARCH_BUDGET`,
so a bounded quick run cannot accidentally enumerate the entire candidate
space.  A row remains marked measured only when the requested policy actually
completed candidate timing.
Use `--only case-a,case-b` (or repeat `--only`) for a bounded replay of a
declared subset; unknown IDs are rejected before any GPU work starts.
The runner passes one output-local `tuning/cache.json` to plan construction for
both measurement and replay. The cache key still contains the complete
semantic and hardware fingerprint, so different ranks, batches, directions,
layouts, or compile policies cannot reuse one another's mapping.

The same registry is accepted by the comprehensive CUDA suites. Their
`runners` entries carry both `kind` and `executable`, while the legacy
`binaries` spelling remains readable for archived manifests. The shared
resolver is [`benchmark/runner_registry.py`](runner_registry.py); comprehensive
and external-baseline runners use it before constructing their own commands.

For example, select only the GPU NTT operator cohort and let the manifest choose
the default output path:

```bash
PYTHONPATH=. python -m benchmark.manifest \
  config/benchmark_full.json --mode full --scope operator \
  --operator ntt --precision word64 --size 16384 --size 32768 \
  --batch 1 --batch 4 --gpu
```

`quick` is the bounded smoke matrix and `full` is the broader declared matrix;
neither mode silently changes precision, shape, batch, layout, or provider.
`full` is a request inventory, not evidence that every declared combination is
executable or supported by every backend. Runners must report unsupported
requests and missing adapters explicitly; manifest expansion proves no runtime
support or measurement coverage.

### FFT Compute Types

`fft-h` and `fft-l` classify benchmark requests and comparisons; both emit the
same underlying `operator: "fft"` and reuse its kernels and search. FFT-H requires
FP32/FP64 input storage and FP32/FP64 core arithmetic. FFT-L covers FP16/BF16
input storage (including explicit FP32 arithmetic) and explicit low-precision
multiplication with higher-precision accumulation. Output dtype alone does not
determine the class. These groups never replace the full numerical contract.

Use `--operator fft-h`, `--operator fft-l`, or `--compute-type fft-l` to select a
group. Legacy `--operator fft` selects both groups at the requested precisions.
Cases accept the same aliases in `operators`, or a `compute_type`/`compute_types`
constraint alongside `operator: "fft"`. `compute_types` is an allowed set for
the derived classification, not an independent Cartesian axis: precisions
`["fp16", "fp32"]` with types `["fft-h", "fft-l"]` produce two rows.
A single explicit type or operator alias must match its precision contract;
conflicting types or unsupported FFT precisions fail before runner execution.

FFT rows retain `precision` and emit `storage_precision`, `compute_precision`,
`accumulation`, and resolved `accumulation_precision`. Optional per-case
`storage_precision` and `compute_precision` assert the contract rather than
override it. `accumulation` accepts `native` (the default) or `fp32` for
FP16/BF16 storage. The latter uses FP32 arithmetic but remains FFT-L.
`fp16-fp32` means FP32 storage, FP16 multiplication, and FP32 accumulation;
its runner policy remains `accumulation: "native"` and the resolved precision
is recorded separately. It remains FFT-L and requires the existing restricted WMMA core. The examples
do not add this restricted mode to general-size campaigns. TF32 has no supported
manifest execution contract and is rejected, never treated as FP32.

```bash
PYTHONPATH=. python -m benchmark.manifest config/benchmark_quick.json \
  --mode quick --operator fft-l --output /tmp/fft-l-quick.json
PYTHONPATH=. python -m benchmark.manifest config/benchmark_full.json \
  --mode full --operator fft --compute-type fft-h --precision fp64 \
  --scope operator --output /tmp/fft-h-full.json
```

The examples declare locally executable low-precision FFT cases only for
cuButterfly. The external comparison catalog separately records provider
precision support and locally measured precision cohorts under
`precision_support`; upstream low-precision availability does not imply a local
adapter or measurement. Empty `local_measured` lists mean
`not_verified_in_this_update`, not that historical measurements never existed;
exact campaign/raw records must be bound before result aggregation.
cuFFT/VkFFT list H/L support, FFTW remains CPU FFT-H,
tcFFT has a CPU-compiled FFT-L adapter with GPU and Graph validation pending;
FlashFFTConv remains source-only FFT-L
convolution context, and cuFFTDx remains a local processing core. Catalog
provider lists are bounded declared domains, not claims of exhaustive upstream
support. Frozen scopes and historical raw results are unchanged.

The bounded [FFT-L CPU preparation](fft_l_quick.md) reuses the semantic runner's
quantized sources and diagnostic reference, records exact vendor binding gaps,
and keeps raw and scaled-input cohorts separate. Preparation establishes no
GPU performance, fair comparison coverage, or numerical acceptance.
