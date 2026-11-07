# M1 scaling preparation, 2026-10-27

The subsequent separate GPU40 cohort completed all 18 workloads, 54 cells and
162 timed trials. See [the results and original-release audit](NARRATIVE_SCALING_RESULTS_20261027.md).
The CPU-only preparation state below is historical; interrupted 80GB and
whole-study 40GB attempts contribute no accepted data to that cohort.

## Delivered state

The finite M1 cohort is generated and all 18 batch-independent modules are ready
in `/home/wt/.cache/cubutterfly/modules`. The controlled runner prepared 18
requests, 54 cells and 36 organization comparisons. Preparation queried no GPU,
loaded no CUDA module and performed no GPU verification or timing. Its status is
`compile-ready-runtime-unverified`.

Inputs and outputs:

- [Cohort config](../../configs/experiments/narrative_scaling_20261027.json).
- [Preparation script](../../scripts/prepare_narrative_scaling.py).
- [Generated study](../../results/narrative_gap_20261027/m1/study.json).
- [Prepared commands/journal](../../results/narrative_gap_20261027/m1/prepared.json).
- [Source, binary, module and resource records](../../results/narrative_gap_20261027/m1/preparation.json).
- Generated CUDA source for every unique request is retained under
  `results/narrative_gap_20261027/m1/sources/` with a SHA256 in the preparation record.

The script uses the existing `compile_request` projection and `compile_mapping`
compiler, CUDA architecture sm80, NVCC 12.4.99, the existing module cache and four
CPU workers. It checks module SHA256 against the compiler manifest. Module
identities include current template/header/compiler inputs through the existing
compiler contract. No runtime, shared runner, existing campaign or calibration
artifact was edited for M1.

## Fixed cohort and interpretation

| Arithmetic | Precision | N | B | Physical groups | Threads |
| --- | --- | --- | --- | --- | --- |
| FFT | FP64 storage and arithmetic | 65536; 1048576 | 1; 4; 16 | 8+8; 10+10 | 128; 256 |
| FWHT | FP64 storage and arithmetic | 65536; 1048576 | 1; 4; 16 | 8+8; 10+10 | 128; 256 |
| NTT | word64, Shoup, q=576460756061519873 | 65536; 1048576 | 1; 4; 16 | 8+8; 10+10 | 128; 256 |

Every cell is forward, unnormalized and out-of-place with natural external
ordering, linear shared layout, shared-iterative backend, scalar radix-2
operator primitive, no overlap and batch tile count 1. FFT uses four-multiply
complex arithmetic and table coefficients; floating accumulation is native.
FP64 was selected before measurement as a separate cohort; no FP32 cell fills
an FP64 gap. NTT modulus is retained as a Python integer throughout generation;
tests also verify its exact decimal value in the prepared benchmark commands.

Three organization levels retain each physical partition:

| Variant | 8-stage physical group | 10-stage physical group |
| --- | --- | --- |
| `local0` | default shared publication after individual stages | same |
| `local2` | 2+2+2+2 | 2+2+2+2+2 |
| `local4` | 4+4 | 4+4+2 |

`shared_resident.cuh::run_leaf` supports widths 1..5 and applies the same
`Operator::apply` primitive inside each register leaf. Both requested width-4
forms compile for all three arithmetic families. There is no missing local4
lowering in this cohort. Each B sweep retains the same compiled module request;
54 public cells project to 18 unique modules.

This intervention measures local publication/register-residency scaling. Changing
local grouping also changes work ownership, synchronization and register demand;
it does not isolate prefetch, overlap, or a single latency-hiding mechanism.
At an 8-stage tile, width2 exposes 64 leaf owners and width4 exposes 16 owners
among 128 launched threads. At a 10-stage tile, width2 exposes 256 owners and
width4 exposes 64 owners among 256 launched threads; the width-2 remainder in
`local4` exposes 256 owners. These follow the source's one-thread-per-leaf loop,
not measured occupancy. Sparse owner counts are a property of these fixed
controls and may matter to their performance. Neither `local0` nor another level
is claimed to be a searched optimized conventional baseline. The cohort does not
measure the complete HHD space or automatic-library competitive performance.

## Static capacity and remaining gates

The static check consumes the existing exact-model A100 80GB PCIe hardware model
at `config/hardware/NVIDIA-A100-80GB-PCIe-sm80-84974239744B/hardware_model.json`.
Its memory capacity is 84974239744 bytes, SM count 108, maximum threads/CTA 1024,
and opt-in shared capacity/CTA 166912 bytes. This artifact provides capacity
facts, not a new performance calibration.

| Quantity | N65536 | N1048576 |
| --- | --- | --- |
| FFT dynamic shared/CTA | 8192 B | 32768 B |
| FWHT/NTT dynamic shared/CTA | 4096 B | 16384 B |
| CTAs per physical group at B1/4/16 | 256/1024/4096 | 1024/4096/16384 |

All cells pass these known shared/thread/memory checks. The largest conservative
device-memory planning envelope is 2214592512 bytes, using `(8*B+4)*N*value_bytes`;
it includes allocation headroom and is not a measured peak. Register allocation,
static shared allocation, spills and achieved occupancy remain unmeasured.
Successful compilation and known capacity checks do not certify GPU executability.

Root owns UUID/model/capacity binding, exclusive launch, runtime mapping/resource
audits, numerical gates and timing. The current benchmark's exact full-batch CPU
references remain a setup cost outside kernel timing. Protocol is three trials,
10 warmups, 20 repeats and full-batch preflight verification. Integer results
require exact equality. Complete floating error distributions and the user's
floating acceptance verdict remain unfinished; no threshold was changed. Normal
benchmark pass flags must not be presented as completing that verdict.

## Reproduction and identities

The executed preparation command was:

```bash
CUDA_VISIBLE_DEVICES='' /home/wt/yes/envs/cubutterfly/bin/python \
  scripts/prepare_narrative_scaling.py --workers 4 --timeout 600
```

Generation and preparation can be repeated into a fresh directory with
`--output PATH`; `--no-compile` only emits the study, static resource checks and
controlled journal. `--build-dir` changes the recorded benchmark directory.
The script refuses preparation in a directory containing `guard/` or `timing.json`.
It does not launch GPU work. Root can consume the exact per-cell preflight/timing
commands already stored in `prepared.json`, with these explicit module settings:

```bash
CUBUTTERFLY_COMPILE_MODE=research
CUBUTTERFLY_JIT_CACHE=/home/wt/.cache/cubutterfly/modules
CUBUTTERFLY_TEMPLATE_ROOT=/home/wt/git/cuButterfly
CUBUTTERFLY_JIT_PYTHON=/home/wt/yes/envs/cubutterfly/bin/python
CUBUTTERFLY_MATHDX_ROOT=/home/wt/git/cuButterfly/external/mathdx/nvidia/mathdx
CUBUTTERFLY_NVCC=/usr/local/cuda/bin/nvcc
```

These environment assignments are inputs for a separately guarded launch, not a
launch performed here. Build `build-e13-cufftdx` is recorded without rebuilding.
Its library SHA256 is
`fac10905e8c7d88a1c42f669545df6182e988409a297ee99f55d00e6d5e57778`.
The preparation record contains benchmark SHA256 values and source hashes;
root must preserve or explicitly rebind any later identity changes.

| Artifact | SHA256 |
| --- | --- |
| Config | `dbc845de39ff9173fa407eaf1d5ea1d97ffa44ef83c19e7df04518b1fc60fda7` |
| Generated study | `fa01b8953c0acb3f4183c5779921ca1f3d4ff6657c94cdd03bfb2de8baac9b31` |
| Prepared journal | `d4f9ffb08d2676714e1bdaf5cd684c38d6e49c81d8c59a453b775b3197374b79` |
| Hardware model | `27d360400478070efb4948cc7fe62b14b4cad51592f49fbd7008248d669e2206` |

The recorded preparation interval was 15.13 seconds with four workers. Stored
per-module original compile durations sum to 50.69 seconds; the helper does not
separately classify cache hits, so this sum is not claimed as fresh compilation
CPU cost. These are observed CPU preparation values, not GPU performance estimates.

The bounded GPU protocol would create 54 preflight and 162 timed processes:
3240 measured calls, 1620 warmup calls and 54 preflight calls. Each full call has
two physical-group kernels, totaling 9828 launches before resource probes/setup.
CPU references, initialization, allocation and compilation remain outside kernel
timing. No elapsed-time or speedup prediction follows from these counts.

CPU validation passed four targeted tests, including cohort shape, precision,
width-4 remainder, deduplication, exact NTT modulus/command values, capacity
rejection, worker limit and subprocess-free generation:

```bash
CUDA_VISIBLE_DEVICES='' /home/wt/yes/envs/cubutterfly/bin/python -m pytest -q \
  tests/test_prepare_narrative_scaling.py
```

## Separate Capacity Cohort Preparation

The preparation helper now accepts `--hardware-model PATH` as an explicit static
capacity input. It requires the model's compute capability to match configured
sm80 and applies the same shared/thread/memory checks to its actual capacities;
it no longer requires the original 80GB memory size. `preparation.json` records
the actual model path, SHA256, source (`explicit-override` or `config`) and full
hardware fields. The study, semantics, module requests and protocol remain fixed.

After root captures the actual 40GB model through the existing live-hardware
workflow, preparation into a fresh independent directory is:

```bash
CUDA_VISIBLE_DEVICES='' /home/wt/yes/envs/cubutterfly/bin/python \
  scripts/prepare_narrative_scaling.py \
  --hardware-model "$M1_GPU40_HARDWARE_MODEL" \
  --output results/narrative_gap_20261027/m1_gpu40 --workers 4 --timeout 600
```

`M1_GPU40_HARDWARE_MODEL` must identify the captured model file. This is a separate
hardware cohort, with its own capacity provenance and future timings. It does not
pool with or overwrite the original 80GB preparation or measurements. The source
hash in the original preparation continues to identify the earlier helper; the
new cohort records the updated helper identity. No existing M1 artifacts were
regenerated for this extension.

Six targeted CPU tests pass, including a fake 40GB model accepted without a GPU
probe, unchanged study/module requests, exact recorded model identity, and wrong
compute capability rejected before output generation. No live GPU model was
captured by this change.
