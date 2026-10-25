# External CPU comparisons

The external CPU adapters fill comparison gaps for the **same semantic
requests** used by the GPU campaign. They do not replace GPU competitors or
the independent correctness oracle. CPU/GPU speedups remain separate from
GPU/GPU speedups, with no pooled average.

## Implemented scope

| Operator | External implementation | Current adapter contract |
| --- | --- | --- |
| FFT | FFTW 3.3.8 | C2C FP32/FP64, packed natural order, rank 1–8, full batch, forward/inverse; 1 and 8 threads reported separately |
| FWHT | FXT 2026.05.19 | FP32/FP64, forward/inverse; original single-thread Walsh core |
| Subset/superset zeta | FXT 2026.05.19 | uint32/uint64 modular arithmetic, forward/inverse; distinct original arithmetic/reversed-arithmetic cores |
| NTT | Intel HEXL v1.2.5 | External word64 core with explicit cyclic-root/order and per-axis adapters; word32 requests use exact storage widening/narrowing |
| Structured 2x2 | HazyResearch butterfly, commit `7217b5d93bc78e1229fed3761bcc70d943f604b7` | Original FP32/FP64 author CPU core, broadcast stage matrices and per-axis composition, forward/inverse |

These are external libraries or author implementations, not replacements
written in this repository and relabeled as competitors. The structured CPU
implementation is an author baseline; it is not a claim about the fastest CPU
implementation. HEXL performance on the current AMD EPYC host is not a claim
about its performance on Intel CPUs with different vector instruction support.
HEXL word32 rows explicitly remain **adapted storage**, not native word32.

For FWHT and zeta, flattening packed multidimensional requests is exact:
independent axes form a tensor product of the same binary transforms. FFT
and NTT do not use that shortcut. FFTW receives the actual rank/extents, while
the NTT and structured adapters execute the requested axes and include packing.

## Measurement and error contract

The runner consumes the existing checksummed manifest and prepared inputs;
it does not resample inputs, change batch, precision, shape or normalization.
All outputs are out of place. Required copies, cyclic twists, order conversion,
width conversion, scaling and allocations **inside execution** are timed.
Compilation, planning, setup allocation and reference calculation are excluded.
FFTW uses `FFTW_MEASURE`, with its input restored after planning.

CPU timing includes host dispatch and uses seven wall-clock samples of twenty
complete calls, after ten warmups. GPU graph replay remains a separately
identified resident-device measurement; neither measurement includes initial
host/device upload or download. This is not an application transfer-cost claim.

Integer results require exact equality of all output elements. Floating
outputs are compared with an independent CPU reference in the requested
precision, and retain lossless output/reference/error arrays and distributions.
There is **no floating acceptance threshold**. Historic wider references are
additional diagnostics only. Structured inverse coefficients are prepared in
FP64 and cast once to the requested precision, matching the public coefficient
contract; subsequent stage arithmetic and its CPU reference use that precision.

`results.json` records source/library hashes, external revisions, input hashes,
thread counts, CPU affinity, all timing samples and load averages. Resume rejects
changed identities. Invalid rows do not generate speedups, and CPU and GPU
results must have identical manifest identities before joining.

The separate `review_gpu_same_precision.py` reviews actual outputs retained by
the frozen GPU runner against the same-precision CPU references. It runs on the
CPU and writes a supplemental report; it does not alter original timings,
relabel incomplete measurements or require another GPU run.

## Current finite run

The 2026-10-13 installation campaign contains **66 distinct requests**. Its CPU
run completed **83/83 measurements**, because FFTW has two thread settings for
each of the 17 FFT requests. There are zero adapter gaps and zero execution or
integer-correctness failures in this population. The 29 integer records passed
exact comparison. All 54 floating records remain `accuracy-review-required`.

| Operator | Requests | CPU measurements | Largest observed relative L2 error |
| --- | ---: | ---: | ---: |
| FFT | 17 | 34 | 1.91e-7 |
| FWHT | 13 | 13 | 1.54e-7 |
| NTT | 13 | 13 | Exact |
| Subset zeta | 8 | 8 | Exact |
| Superset zeta | 8 | 8 | Exact |
| Structured 2x2 | 7 | 7 | 0 on these inputs |

Results are in
[`cpu_external_v2/results.json`](../results/install_compare_20261013/cpu_external_v2/results.json)
and the per-request
[`unified_comparison.csv`](../results/install_compare_20261013/cpu_external_v2/unified_comparison.csv).
The host is an AMD EPYC 7542, with this run pinned to logical CPUs 16–23 and
background math libraries limited to one thread. FFTW's own eight-thread
configuration is explicit. This is a **shared-host preliminary timing** run:
the largest within-row max/min sample ratio is 2.48 for FFTW single-thread.
An uncontended repeat is needed before using CPU speedups as final paper data.

Earlier protocol-debug output is preserved under `cpu_protocol_debug` and
marked invalid. It is excluded from these counts and all comparisons.
This finite population does not establish support for every stride/order,
real FFT, low floating precision, arbitrary lengths or other matrix families.

## Reproduction

Use `/home/wt/yes/envs/cubutterfly/bin/python` on this host. System FFTW FP32,
FP64 and thread runtimes are installed. FXT's original source and license are
under `results/install_compare_20261013/external/fxt`; the runner builds a
standalone benchmark bridge and records a manifest. Official sources are
[FFTW](https://www.fftw.org/), [FXT](https://www.jjj.de/fxt/fxtpage.html),
[HEXL](https://github.com/intel/hexl) and
[author Butterfly](https://github.com/HazyResearch/butterfly).

Build the two other adapters before measurement. Each accepts a different
explicit source root for another machine; `--fetch` fetches the pinned upstream
revision if it is absent. The C++ bridge sources live in `scripts`, not only in
an experiment output directory.

```bash
python scripts/cpu_ntt_baseline.py --root results/install_compare_20261013/external/hexl
python scripts/cpu_structured_baseline.py --root results/install_compare_20261013/external/butterfly
```

Run the already-prepared population into a **fresh output directory**:

```bash
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  taskset -c 16-23 python scripts/cpu_external_baselines.py \
  --manifest results/install_compare_20261013/manifest.json \
  --preparation results/install_compare_20261013/prepared \
  --output results/install_compare_20261013/cpu_external_repeat \
  --fxt-root results/install_compare_20261013/external/fxt \
  --hexl-root results/install_compare_20261013/external/hexl \
  --butterfly-root results/install_compare_20261013/external/butterfly \
  --gpu-results results/install_compare_20261013/measurement/results.json
```

Choose an appropriate affinity on other hosts and record it. Use `--resume`
only with the original output directory and matching identities. To update the
table after the GPU campaign produces results without rerunning CPU timings:

```bash
python scripts/cpu_external_baselines.py \
  --manifest results/install_compare_20261013/manifest.json \
  --preparation results/install_compare_20261013/prepared \
  --output results/install_compare_20261013/cpu_external_v2 \
  --gpu-results results/install_compare_20261013/measurement/results.json \
  --join-only
python scripts/review_gpu_same_precision.py \
  --manifest results/install_compare_20261013/manifest.json \
  --preparation results/install_compare_20261013/prepared \
  --gpu-results results/install_compare_20261013/measurement/results.json \
  --output results/install_compare_20261013/cpu_external_v2/gpu_same_precision_review.json
```

For the active campaign, `follow_cpu_gpu_comparison.py` is already running in
the background, bound to driver PID `1261063` and its process start time. When
that driver ends it performs the final join and supplemental review once. It
does not launch GPU work or restart an interrupted driver. Its own
[`join_status.json`](../results/install_compare_20261013/cpu_external_v2/join_status.json)
distinguishes a complete join, a partial checkpoint, and termination without a
checkpoint. The measured CPU JSON is checksummed at startup and cannot change
silently while this follower waits.
