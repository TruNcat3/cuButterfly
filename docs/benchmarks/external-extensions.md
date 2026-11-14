# External transform extensions

This workflow adds fresh paired comparisons against complete external GPU-FFT,
HadaCore and tcFFT transforms. Each adapter preserves its pinned upstream
arithmetic and records the boundary work needed to produce the requested output.
These comparisons assess complete external implementations. A processing core,
such as a codelet registered inside cuButterfly's planner, instead participates
in cuButterfly's execution organization. Calling an external transform through
`build_core` does not establish processing-core integration or a fixed-core
organization ablation. See the [research principles](../research_principles.md)
for those evidence categories.

HadaCore is additionally tracked as a **candidate local-core strategy**. Its
register fragments, Tensor Core arithmetic and launch parameters can be adapted
to a dependency-closed tile, while cuButterfly continues to own stage grouping,
layout, boundary swizzles, residency and selection. The current pinned adapter
remains comparator-only. The optional local provider is a separate integration;
see the [local core contract](../hadacore_local_core_contract.md). This separation keeps the
external speed comparison and the later same-core organization ablation
auditable.

## Support and timing boundaries

The domains below describe the local adapters. Device resources and numerical
behavior still require verification on the selected GPU. No FP32 request may be
silently converted to FP16, and no batch may be padded or reduced to satisfy an
adapter's constraints.

| Adapter and detailed contract | Precision and legal core domain | Work included in the complete call |
| --- | --- | --- |
| [GPU-FFT](../../benchmarks/gpufft/README.md) | Rank-one cyclic C2C; matching FP32 or FP64 storage/arithmetic; power-of-two N from 2 through 2^28; positive batch with N×batch < 2^31; batch ≤ 65535 when N > 512. Both directions and optional inverse normalization. | Forward input copy, upstream transform and output bit reversal; inverse input bit reversal, upstream transform and output copy. Required normalization remains included. |
| [HadaCore](../../benchmarks/hadacore/README.md) | FP16 storage and native FP16 MMA, with FP16 accumulator/result type; flattened power-of-two FWHT N from 2 through 32768; N×batch divisible by 256 and ≤ 2147483392. Packed natural core buffers, either placement, both directions and optional inverse normalization. BF16 is outside this extension. | Upstream orthonormal FWHT plus a separate output rescale: FP32 scalar multiplication of the FP16 result followed by FP16 writeback. No output allocation or padding; this supplied-output integration differs from native PyTorch return-value timing. |
| [tcFFT](../../benchmarks/tcfft/README.md) | Compute-capability major 8 only (SM80 binary); unsupported on SM120 pending a port. Rank-one forward, unnormalized C2C; FP16 storage, multiplication and WMMA accumulator type. N ∈ {256, 512, 1024, 131072, 262144, 524288, 16777216, 33554432, 67108864, 134217728}; N×batch divisible by 8192 and within signed-int range; subsequent grid.y ≤ 65535. | Device mixed-radix input reorder, upstream FFT and device output copy. Private scratch protects aliased input/output. |

Tensor Core accumulator types do not specify hardware internal reduction order.
HadaCore uses rounded normalized coefficients inside each subtransform, then
restores the requested mathematical scale at the boundary. Its FP16 rounding
therefore differs from cuButterfly's per-butterfly FP16 arithmetic. Retain this
contract with every error report; neither implementation serves as the other's
numerical oracle.

The common semantic runner additionally times required order/layout gathers,
scatters, user placement handling and generic output scaling. Its separate
packed scratch buffers can implement a user in-place request; adapter metadata
preserves the requested and projected core placements. Unsupported rank or
precision remains an explicit gap. In particular, GPU-FFT does not substitute a
flattened one-dimensional FFT for a multidimensional FFT.

Compile the pinned sources using the linked adapter instructions before taking
GPU ownership. CPU compilation and reference generation do not launch GPU work
and stay outside execution timing. Plan creation, allocation, coefficient/index
preparation, first-use JIT and warmup also stay outside steady-state timing.
CPU preparation currently covers external-library compilation, inputs and
references. A public DEFAULT cache miss can still invoke CPU candidate ranking
and device resource probes during plan creation; it is not a measurement-only
GPU phase. The current planner has no persistent offline-ranking import. Reuse
a compatible completed plan cache where available and record any fallback.
Necessary copies, permutations and scale kernels stay inside. Preparation alone
establishes neither GPU correctness nor Graph support nor measured coverage.

## Frozen population

[`experiments/workloads/external_extensions.json`](../../experiments/workloads/external_extensions.json)
is the public request population, with 28 cases in three groups:

- `external-extension-fft-main17`: the 17 original FFT requests, with explicit
  cuFFT, VkFFT and GPU-FFT comparisons. Unsupported GPU-FFT ranks remain present.
- `external-extension-fwht-fp16-9`: nine forward, unnormalized FP16/native FWHT
  requests, N ∈ {256, 4096, 32768} and batch ∈ {1, 16, 256}, compared with HadaCore.
- `external-extension-semantic-controls`: two FP16 FWHT controls at N=4096:
  batch=16 normalized inverse and batch=256 forward in-place.

The FP16 FWHT group and controls are independent of the older FP32 FWHT cohort.
Do not combine their speedup aggregates or relabel the older data. This population
is a request inventory, not a statement that all requests have been measured.
tcFFT uses its separate restricted FP16 comparison path in
[`scripts/run_tcfft_comparison.py`](../../scripts/run_tcfft_comparison.py); it is
not a baseline name accepted by the extension runner or an extra case silently
added to this population.

## Configure and prepare

Run commands from the repository root in an environment with the repository's
Python frontend and numerical-reference dependencies available. The extension
config requires `library`, `gpu_uuid`, `environment` and `cases`. `library` names
the cuButterfly shared library; `gpu_uuid` identifies the target device, rather
than a transient index; `environment` maps strings to strings. Each case keeps
its canonical descriptor, ID, population labels and explicit baseline list.
`vkfft_library` selects the VkFFT bridge. GPU-FFT and HadaCore libraries are set
through `CUBUTTERFLY_GPUFFT_LIBRARY` and `CUBUTTERFLY_HADACORE_LIBRARY` in the
configured environment. cuFFT uses the CUDA library binding available to the
runner. A missing dependency remains a reported gap.

This example copies the full public population without changing its semantics.
Set `CUBUTTERFLY_GPU_UUID` to the chosen UUID first and adjust the relative build
paths for the local installation. If using a prepared registry/profile, export
`CUBUTTERFLY_REGISTRY` and `CUBUTTERFLY_PROFILE_ROOT` before generating the config.

```sh
python - <<'PY'
import json
import os
from pathlib import Path

population = json.loads(Path('experiments/workloads/external_extensions.json').read_text())
environment = {
    'CUBUTTERFLY_COMPILE_MODE': os.environ.get('CUBUTTERFLY_COMPILE_MODE', 'research'),
    'CUBUTTERFLY_GPUFFT_LIBRARY': 'build/gpufft/libgpufft_baseline.so',
    'CUBUTTERFLY_HADACORE_LIBRARY': 'build/hadacore/libhadacore_baseline.so',
}
for key in ('CUBUTTERFLY_REGISTRY', 'CUBUTTERFLY_PROFILE_ROOT'):
    if key in os.environ:
        environment[key] = os.environ[key]
config = {
    'library': 'build/libcuntt.so',
    'gpu_uuid': os.environ['CUBUTTERFLY_GPU_UUID'],
    'environment': environment,
    'vkfft_library': 'build/vkfft/libvkfft_xd_bridge.so',
    'cases': population['cases'],
}
output = Path('results/external-extensions')
output.mkdir(parents=True, exist_ok=True)
(output / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
PY
python scripts/run_comparator_extension.py --prepare \
  --config results/external-extensions/config.json \
  --output results/external-extensions/campaign
```

`--prepare` generates CPU inputs, references, encoded layouts and checksums. It
writes `preparation.json` and per-case NPZ arrays under `prepared/`. The primary
FFT reference preserves the requested storage/compute precision; native FP16
FWHT uses a CPU Sylvester transform rounded at each stage. The higher-precision
diagnostic reference is retained separately. `prepare_reference_limit` is an
optional config field; exceeding the reference budget leaves a preparation gap.
The default limit is 8388608 elements.

Public selection uses `DEFAULT` (`selection_policy=0`). Reuse the target's
compatible registry/cache and profile under their recorded environment instead
of inserting a hand-selected mapping. A registry miss follows the public
planner's fallback; this campaign does not rerun exhaustive tuning or establish
that a selected mapping is globally optimal. Preserve the registry snapshot and
build provenance used for the run; see the
[hardware profile guide](../hardware_profile_install.md).

## Run, resume and summarize

```sh
python scripts/run_comparator_extension.py --run \
  --output results/external-extensions/campaign
# Optional bounded execution: use an exact ID from the prepared population.
python scripts/run_comparator_extension.py --run \
  --output results/external-extensions/campaign --case fwht-fp16-n4096-b16
python scripts/summarize_comparator_extension.py \
  results/external-extensions/campaign \
  --output results/external-extensions/summary
```

`--run` loads the stored config and verifies prepared array hashes before Torch
import or CUDA initialization. `--case` selects one exact case for execution;
it does not change the prepared population. Unknown IDs are rejected. Completed
rows are reused only when their source/library/environment/preparation identity,
checksums and before/after exclusivity guards match. Rerun the same command to
resume; use a new campaign directory for incompatible inputs or identities.

The runner checks exclusive GPU use before initialization and around each case.
A busy device returns exit code **76**, writes `pending.json` and leaves the work
pending. It does not stop another process or record the interrupted case as a
valid measurement. Resume after the device becomes available.

The CPU-only summarizer writes `comparisons.csv`, `groups.csv` and `summary.md`.
It retains every scheduled backend, including unsupported, missing, failed and
pending rows. Fresh timing pairs report baseline latency divided by cuButterfly
latency; values above one favor cuButterfly. Floating results remain error
reports without a new accuracy acceptance threshold. Structural timing pairs
are not numerical acceptance. Investigation warnings or failures withhold the
whole affected group's geometric mean; individual observations remain visible.
Retain preparation manifests, raw rows, mappings, full error artifacts and all
negative results. Partial coverage cannot be presented as a completed matrix.

## Validate an external adapter independently

An external library can complete numerical and execution validation before
cuButterfly's matching selection/calibration is ready. This mode reuses the
prepared requests and references, creates no cuButterfly plan, and writes a
separate result directory:

```sh
python scripts/validate_external_comparator.py results/external-extensions/campaign \
  --backend gpu-fft --output results/external-extensions/gpufft-validation
```

`--backend hadacore` uses the same entry point. Only requests that explicitly
name that backend are selected. Unsupported requests remain visible. The
existing complete-call runner also exercises timing, but these timings are
validation diagnostics without a cuButterfly pair and must not enter speedup
aggregates. Out-of-place requests exercise Graph replay; in-place requests use
the existing single-event protocol and explicitly report a Graph-validation gap.
This entry point uses the same UUID checks, busy exit code and resumable identity
bindings as the paired campaign.


## Target-host FFT17 and optional MMA9

Use this separate extension after the [PRO6000 quick12/full66 flow](../hardware/pro6000_server96.md).
Run from the target checkout. The caller supplies absolute local paths below;
`TARGET_CUDA_ENV` is the target CUDA shell setup and `TARGET_PROFILE_ENV` is the
matching calibrated profile environment. Source the same environment for native
calibration, public validation, preparation and execution. The main run's
`runtime_environment.sh` alone does not publish a complete installed profile;
select the matching target profile explicitly. Do not source A100 profiles or
copy their registries, calibration or caches.

```bash
: "${TARGET_CUDA_ENV:?set target CUDA setup script}"
: "${TARGET_CONDA_PREFIX:?set target conda prefix}"
: "${TARGET_PROFILE_ENV:?set matching target profile environment script}"
: "${TARGET_LIBRARY:?set absolute target libcuntt.so path}"
: "${TARGET_VKFFT_LIBRARY:?set absolute target VkFFT bridge path}"
: "${CUBUTTERFLY_GPU_UUID:?set physical target GPU UUID}"
: "${EXTENSION_ROOT:?set fresh absolute target-local results directory}"
source "$TARGET_CUDA_ENV"
conda activate "$TARGET_CONDA_PREFIX"
source "$TARGET_PROFILE_ENV"
: "${CUDA_HOME:?target CUDA toolkit root required}"
: "${CUBUTTERFLY_REGISTRY:?target measured registry required}"
: "${CUBUTTERFLY_PROFILE_ROOT:?target profile tree required}"
export TARGET_LIBRARY TARGET_VKFFT_LIBRARY EXTENSION_ROOT CUBUTTERFLY_GPU_UUID CUDA_HOME
export PYTHONPATH="$PWD/python:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=
```

For PRO6000 use the runbook's CUDA 12.8 environment. Build pinned official
sources on CPU with the target compiler. Use fresh output directories; for
existing official checkouts, verify the same detached commits before building.
GPU-FFT accepts an explicit target architecture through its Python build API;
HadaCore's CLI accepts `--nvcc` and retains its pinned SM80/compute80 targets.

```bash
mkdir -p external
git clone --no-checkout https://github.com/Alisah-Ozcan/GPU-FFT.git external/GPU-FFT
git -C external/GPU-FFT checkout --detach 7f05858f7af7c0bbf55a77472426d9ac56ec6b23
python - <<'BUILD'
import os, sys
sys.path.insert(0, 'scripts')
from gpufft_baseline import build
build('build/gpufft-target', upstream='external/GPU-FFT',
      nvcc=os.environ['CUDA_HOME'] + '/bin/nvcc', arch='120')
BUILD
export CUBUTTERFLY_GPUFFT_LIBRARY="$PWD/build/gpufft-target/libgpufft_baseline.so"
```

HadaCore comparator and tile binaries carry compute80 PTX for target JIT. This
is not Blackwell validation: check target correctness, streams, Graph replay
where supported, resources and complete-call behavior before performance claims.
Use `validate_external_comparator.py --backend hadacore` on a separately prepared
FWHT campaign for comparator validation; FFT17 has no HadaCore requests.
The tcFFT bridge enforces `device.major == 8` and builds SM80 only. It remains
unsupported on SM120 pending a port; do not remove the runtime gate.

Generate FFT17 directly from the frozen population, retaining unsupported ranks.
Every library, UUID, registry and profile binding below belongs to this target.
`runtime_bindings: {}` requests the prepared runtime-input hash snapshot.

```bash
python - <<'CONFIG'
import json, os
from pathlib import Path
population = json.loads(Path('experiments/workloads/external_extensions.json').read_text())
keys = ('CUBUTTERFLY_REGISTRY', 'CUBUTTERFLY_PROFILE_ROOT',
        'CUBUTTERFLY_GPUFFT_LIBRARY')
environment = {k: os.environ[k] for k in keys}
environment['CUBUTTERFLY_COMPILE_MODE'] = os.environ.get('CUBUTTERFLY_COMPILE_MODE', 'research')
for k in ('CUBUTTERFLY_HARDWARE_MODEL',):
    if os.environ.get(k):
        environment[k] = os.environ[k]
config = dict(library=os.environ['TARGET_LIBRARY'],
              gpu_uuid=os.environ['CUBUTTERFLY_GPU_UUID'], environment=environment,
              vkfft_library=os.environ['TARGET_VKFFT_LIBRARY'], runtime_bindings={},
              cases=[c for c in population['cases']
                     if 'external-extension-fft-main17' in c['populations']])
assert len(config['cases']) == 17
root = Path(os.environ['EXTENSION_ROOT'])
root.mkdir(parents=True, exist_ok=True)
(root / 'fft17.json').write_text(json.dumps(config, indent=2) + '\n')
CONFIG
python scripts/run_comparator_extension.py --prepare \
  --config "$EXTENSION_ROOT/fft17.json" --output "$EXTENSION_ROOT/fft17"
# Set a future deadline, including timezone, e.g. 2026-11-15T08:00:00+08:00.
: "${EXTENSION_UNTIL:?set future ISO-8601 deadline with explicit timezone}"
unset CUDA_VISIBLE_DEVICES
python scripts/watch_comparator_extension.py --campaign "$EXTENSION_ROOT/fft17" \
  --state-dir "$EXTENSION_ROOT/watcher" --until "$EXTENSION_UNTIL"
python scripts/summarize_comparator_extension.py "$EXTENSION_ROOT/fft17" \
  --output "$EXTENSION_ROOT/fft17-summary"
```

The watcher waits for idle exclusivity, retries busy outcomes and stops at the
deadline or on errors. Inspect pending, unsupported and failed rows in the
summary; an expired deadline or partial matrix is not completed evidence.

MMA9 is an optional **explicit-policy variant** of the nine FP16 FWHT requests.
Build the optional HadaCore comparator on CPU only if pursuing this cohort:

```bash
export CUDA_VISIBLE_DEVICES=
git clone --no-checkout https://github.com/pytorch-labs/applied-ai.git external/hadacore
git -C external/hadacore checkout --detach 2391954b19988bd76cf3c2ea84c1ce74b68d568b
python scripts/hadacore_baseline.py --upstream external/hadacore \
  --output build/hadacore-target --nvcc "$CUDA_HOME/bin/nvcc"
export CUBUTTERFLY_HADACORE_LIBRARY="$PWD/build/hadacore-target/libhadacore_baseline.so"
```

Then build the tile provider with `--nvcc "$CUDA_HOME/bin/nvcc"` and complete
target-local tile validation/MMA calibration, paired STRICT native calibration
and public AUTO validation as specified by the
[local-core guide](../hadacore_local_core_contract.md). Pass `TARGET_LIBRARY` and
the target comparator library to those commands in place of their example
paths. Only after those checks pass, export `CUBUTTERFLY_HADACORE_PROFILE` to the
resulting target paired profile. Keep the same ordinary profile environment.
Generate a new config and campaign; never relabel STRICT rows as MMA rows:

```bash
export CUDA_VISIBLE_DEVICES=
python - <<'MMA'
import copy, json, os
from pathlib import Path
root = Path(os.environ['EXTENSION_ROOT'])
config = json.loads((root / 'fft17.json').read_text())
config['environment']['CUBUTTERFLY_HADACORE_LIBRARY'] = os.environ['CUBUTTERFLY_HADACORE_LIBRARY']
config['environment']['CUBUTTERFLY_HADACORE_PROFILE'] = os.environ['CUBUTTERFLY_HADACORE_PROFILE']
population = json.loads(Path('experiments/workloads/external_extensions.json').read_text())
cases = copy.deepcopy([c for c in population['cases']
                       if 'external-extension-fwht-fp16-9' in c['populations']])
assert len(cases) == 9
for case in cases:
    case['id'] += '-allow-fp16-mma'
    case['descriptor']['semantics']['arithmetic_policy'] = 1  # ALLOW_FP16_MMA
    case['populations'] = ['external-extension-fwht-fp16-mma9']
config['cases'] = cases
(root / 'mma9.json').write_text(json.dumps(config, indent=2) + '\n')
MMA
python scripts/run_comparator_extension.py --prepare \
  --config "$EXTENSION_ROOT/mma9.json" --output "$EXTENSION_ROOT/mma9"
unset CUDA_VISIBLE_DEVICES
python scripts/watch_comparator_extension.py --campaign "$EXTENSION_ROOT/mma9" \
  --state-dir "$EXTENSION_ROOT/mma9-watcher" --until "$EXTENSION_UNTIL"
python scripts/summarize_comparator_extension.py "$EXTENSION_ROOT/mma9" \
  --output "$EXTENSION_ROOT/mma9-summary"
```

`ALLOW_FP16_MMA` permits AUTO to compare validated native and MMA candidates;
it does not force MMA selection. Preserve selected mappings and reasons, and
report this cohort separately from STRICT FWHT and FFT17. Missing target
calibration remains missing coverage, even when an A100 profile exists.

## Dated validation evidence

The [2026-11-14 comparator checkpoint](../../experiments/reports/EXTERNAL_COMPARATOR_CLOSURE_20261114.md)
separates HadaCore local/native/public validation, two calibrated tcFFT paired
requests, and independent GPU-FFT adapter diagnostics. It records the numerical
contracts and unsupported requests. These bounded results do not complete the
full extension performance matrix; adapter diagnostic times remain outside
paired speedup aggregates.


For bounded unattended resumption, use
[`watch_comparator_extension.py`](../../scripts/watch_comparator_extension.py)
with one or more prepared `--campaign` directories, `--state-dir` and a required
`--until` ISO-8601 timestamp with explicit timezone. It waits for exclusive GPU
use until that deadline, retries busy outcomes, and stops on execution or row
errors. Its status and logs preserve the next resumable work. See the checkpoint
for a two-campaign example; CPU preparation remains separate from measurement.

Optionally put `"runtime_bindings": {}` in a preparation config to snapshot the
selected libraries, registry/profile JSON inputs and referenced provider libraries.
The runner checks their population and hashes before CUDA initialization and
again before publishing a row. Reprepare in a new compatible campaign when those
inputs change. Omitting this optional field preserves the existing workflow.
