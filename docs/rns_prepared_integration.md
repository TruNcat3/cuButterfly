# Prepared RNS NTT integration

The RNS API prepares a word64 NTT plan once, then executes it on caller-owned
CUDA buffers. Applications can replace their NTT callback while retaining
their other operators. The interface preserves the requested moduli, forward
roots, polynomial offsets, direction, ring, ordering, and placement. Inputs
must contain canonical residues; it performs no implicit precision conversion,
reordering, batch reduction, or host/device copy.

The contracts are defined in [rns_ntt.hpp](../include/cuntt/rns_ntt.hpp),
[rns_tuning.hpp](../include/cuntt/rns_tuning.hpp), and the optional
[Python frontend](../python/cubutterfly/rns.py). Run the commands below from
the repository root, replacing the example application and library paths with
your installed artifacts. Select the intended CUDA device before preparation.

## Export and extract actual requests

The existing HEonGPU scalar and stage adapters use a shared
[preparation helper](../adapters/heongpu/rns_preparation.hpp). Enable export
during an application preparation run:

```bash
mkdir -p work/rns
CUBUTTERFLY_RNS_REQUEST_EXPORT=work/rns/observations.ndjson \
CUBUTTERFLY_RNS_CACHE=work/rns/cache \
CUBUTTERFLY_RNS_CACHE_POLICY=prefer \
  ./path/to/ckks-application
```

This application run needs its normal CUDA environment. Export appends one
NDJSON record after each successful preparation, with schema
`cubutterfly-rns-request-v1`, origin `scalar` or `stage`, placement, the complete
config, the selected mapping, and `cache_hit`. Use a fresh output filename for
a distinct collection, or deliberately combine append runs. The parent
directory must exist. The export represents requests that reached these
adapter preparation paths, rather than every potential application workload.

Once an export exists, extraction needs only Python's standard library:

```bash
python scripts/prepare_rns_requests.py extract \
  --input work/rns/observations.ndjson \
  --output work/rns/requests
```

The script validates schemas, fields, types, and enum names, then writes
`manifest.json` and one complete config JSON per unique request. The identity
includes the full config and placement, retaining domain/root values and
polynomial order and offsets. Mapping and origin observations are preserved
in the manifest. Counts describe **prepare observations**, not application
calls or kernel executions. Extraction performs no CUDA loading or arithmetic
validation.

## Prepare the cache and require replay

On the target GPU, prepare every extracted request using the built shared
library:

```bash
python scripts/prepare_rns_requests.py prepare \
  --manifest work/rns/requests/manifest.json \
  --library ./build/libcuntt.so \
  --cache-directory work/rns/cache \
  --output work/rns/preparation.json \
  --policy prefer --repetitions 5
```

`prefer` replays a compatible entry or runs bounded tuning on a miss.
`refresh` tunes and replaces the entry. `require` fails when the required entry
is missing; corrupt or incompatible records also fail. Use `--request-id ID`
(repeatable) or `--limit N` for an explicit subset; the default is the entire
manifest. Results record selected IDs, cache keys, cache hits, mapping, source,
and preparation wall time. Each request failure is recorded, other requests
continue, and the command exits nonzero if any fail.

Replay through the same application adapter with a populated cache:

```bash
CUBUTTERFLY_RNS_CACHE=work/rns/cache \
CUBUTTERFLY_RNS_CACHE_POLICY=require \
  ./path/to/ckks-application
```

Keep the repetition protocol consistent: the adapter uses the default of five.
An offline run with a different `--repetitions` value creates a different cache
identity. A subset preparation covers only the selected requests; an application
that reaches another request will fail under `require`.

Set `CUBUTTERFLY_RNS_TUNE_REPETITIONS` to an integer from 1 to 1000 to choose a
larger bounded confirmation budget for installation or research tuning. The
value participates in cache identity, so prepare the complete request set with
the same value before using `CUBUTTERFLY_RNS_CACHE_POLICY=require`. This changes
the search budget only; it does not select a fixed mapping or alter steady-state
execution. Preparation remains outside operator and application timing.

## Select the shared-memory exchange policy

`RnsMapping.exchange` defaults to `RnsExchange::Block`. Set it to
`RnsExchange::DependencyScoped` on an otherwise valid `FusedCTGS` or
`GpuNttMerge` mapping to let local thread ownership select each exchange
boundary's barrier. A boundary uses a warp barrier only when every shared
value stays within its owning warp; other boundaries keep the CTA barrier.
Both policies preserve arithmetic and ordering. `Hybrid2D` accepts only
`Block`; each core family's existing ring, arithmetic, geometry, and offset
restrictions still apply. This policy controls shared-memory synchronization;
it does not add a register/shuffle exchange kernel or extend other butterfly APIs.

Mapping JSON uses `"exchange": "block"` or
`"exchange": "dependency_scoped"`. Schema-v1 mappings without this field
decode as `block`. Bounded tuning measures both policies where their barrier
schedules differ; a DependencyScoped candidate with no warp boundaries is
deduplicated against Block. Preparation saves the selected policy in the
cache and restores it on replay.

Both fused families accept 32/64/128/256 threads per pass. The single-warp
choice participates in bounded tuning alongside the existing candidates;
it is not selected by GPU model or application name. FusedCTGS tuning also
allows different thread counts in the two passes. A 32-thread pass keeps
shared-memory exchanges within one warp under `dependency_scoped`; this does
not remove shared-memory traffic or introduce a register-only core. Hybrid2D
retains its 64/128/256-thread contract, and Merge retains its additional
provider-specific 512-thread option.

Explicit `set_mapping()` validates and prepares scratch before publishing the
new mapping and exchange masks. If allocation fails, the previous executable
mapping, selection metadata and workspace remain intact. Preparing an in-place
native-boundary fused mapping can release its unused private scratch; execution
still performs no allocation.

The fused two-interval lowering specializes each kernel's first/last and
prefix/suffix role at compile time. Forward and inverse execution derive those
roles from the same stage partition. This removes runtime role branches and
redundant address calculations without adding a mapping dimension or changing
the requested order, arithmetic, or placement. Register use depends on the
particular template; specialization does not guarantee a faster candidate.
Reprepare affected requests after changing this implementation.

The added trailing field changes the C++ `RnsMapping` layout: rebuild existing
C++ RNS consumers. The opaque C ABI and Python frontend do not access this
structure's layout, but must load the matching updated shared library.

## Integrate a Python NTT callback

Importing `cubutterfly.rns` loads neither CUDA nor a tensor framework. Load the
library explicitly with `Library(path)` or `CUBUTTERFLY_LIBRARY`. `Plan` accepts
a config dict or a JSON object string. Call `prepare()` once before `execute()`;
`info()` reports the actual request, mapping, and cache provenance.

The following function illustrates a specific application contract:
`app.ntt(input_address, output_address, stream=0)` receives raw device addresses,
and `app.run_and_wait()` finishes all work and releases captured graphs before
returning. Supply an adapter for the actual application's signature, return
values, ownership, and NTT semantics; the helper does not infer them.

```python
from cubutterfly.rns import CachePolicy, Library, Placement, Plan, replace_ntt_callback


def run_with_rns(app, config, library_path, cache_directory):
    with Plan(Library(library_path), config) as plan:
        plan.prepare(placement=Placement.IN_PLACE,
                     cache_directory=cache_directory,
                     policy=CachePolicy.REQUIRE, repetitions=5)

        def ntt(input_address, output_address, *, stream=0):
            plan.execute(input_address, output_address, stream=stream)

        with replace_ntt_callback(app, "ntt", ntt):
            return app.run_and_wait()
```

This example prepares an in-place plan: input and output addresses must match.
Use `Placement.SEPARATE` for separate buffers. Addresses must be integers or
`ctypes.c_void_p`, nonzero and uint64 aligned; stream zero denotes the default
stream. The caller selects the device and supplies allocations of sufficient
size for the complete offsets. Execution enqueues work without synchronization,
allocation, tuning, or copying. Use one stream at a time, and keep buffers,
streams, and plans alive until every launch and graph use has completed.
Close plans explicitly or through a context manager after that point.

The callback helper saves and restores the explicit object's attribute,
including on exceptions. Preserve the native callback as the baseline by
running the same application outside the replacement context, with matched
inputs, batch, execution mode, and memory boundaries. Replace only the NTT;
retain the application's other operators and necessary conversions in both
measurements. This Python mechanism cannot intercept statically linked C++
HEonGPU calls: those still require the existing compiled adapter integration.

## Cache identity and measurement boundaries

Cache compatibility includes the complete request, placement, hardware model
and capacity/compute properties, runtime/build fingerprint, CUDA driver and
runtime versions, and timing protocol with repetitions. The record stores the
device UUID and index as provenance; UUID is not part of this compatibility
key. Loading compares the full identity, rather than trusting a filename hash.
Treat a missing required entry as a request to prepare the matching environment,
not as permission to change the requested semantics.

The search is bounded CUDA Graph timing with finalist confirmation. Its records
are explicitly `timing_only_not_mathematically_validated`. A cache hit establishes
compatible measured provenance, not exact mathematical correctness, exhaustive
optimality, or installation-wide calibration. Validate NTT outputs exactly and
verify complete application behavior separately.

Keep construction, preparation, tuning, request export, and Graph creation
outside steady-state operator and application timing. The preparation script's
wall time covers construction and preparation; it is not NTT latency. A required
cache replay avoids tuning but still performs normal plan construction and
resource preparation before execution.

## Compare a real request with native GPU-NTT

The CKKS experiment wrapper builds `rns_request_compare` against its pinned
GPU-NTT archive. Select explicit IDs from the extracted manifest:

```bash
python experiments/workloads/run_rns_request_comparison.py \
  --manifest work/rns/requests/manifest.json \
  --request-id rns-REPLACE_WITH_MANIFEST_ID \
  --binary build-ckks/rns_request_compare \
  --cuntt-library build/libcuntt.so \
  --cache-dir work/rns/cache \
  --output-directory work/rns/comparison \
  --gpu-uuid GPU-REPLACE_WITH_TARGET_UUID --thermal-guard
```

Repeat `--request-id` for additional requests. The default requires prepared
cache entries with five tuning repetitions. Each provider executes the same
request and receives exact CPU checks on changed and restored inputs. Timing
uses alternating backend order over ten rounds and twenty Graph-contained
transforms per sample. Preparation, uploads, CPU references, and Graph creation
are outside events. Samples, medians, mappings, hardware/build provenance, and
sampled GPU occupancy checks are retained separately for each ID.

This diagnostic currently requires dense in-place, N-aligned physical rows,
canonical word64 residues, negacyclic native CT/GS boundaries, and logN 12..20.
Logical polynomial order may be permuted. Unsupported requests fail explicitly;
the tool adds no hidden packing. These are individual-request operator results,
not complete CKKS latency or an execution-frequency-weighted application score.

The default `--native-dispatch modulus-ordered` uses GPU-NTT's explicit
per-row modulus lookup. For dense physical rows whose domain sequence is
`physical_row % domain_count`, use `--native-dispatch contiguous` in a new
output directory to compare with its ordinary forward/inverse API. Incompatible
sequences fail without fallback. This control keeps the prepared cuButterfly
mapping unchanged and records the native API and counts in CSV/metadata.
The two native paths implement equivalent semantics but may have different
costs. Neither reconstructs the original application's dispatch automatically:
the exported semantic config does not retain its original API or index tables.
Keep this native-dispatch control separate from the default comparison and
from complete application measurements.

### Diagnose exchange as a single variable

After the automatic comparison above, replay the same prepared winner with
each exchange policy. Keep the request, cache, library, device, and timing
arguments fixed; use `require` so both runs load the same cached mapping:

```bash
for exchange in block dependency_scoped; do
  python experiments/workloads/run_rns_request_comparison.py \
    --manifest work/rns/requests/manifest.json \
    --request-id rns-REPLACE_WITH_MANIFEST_ID \
    --binary build-ckks/rns_request_compare \
    --cuntt-library build/libcuntt.so \
    --cache-dir work/rns/cache --cache-policy require \
    --output-directory "work/rns/exchange-$exchange" \
    --gpu-uuid GPU-REPLACE_WITH_TARGET_UUID --thermal-guard \
    --tune-repetitions 5 --repetitions 20 --rounds 10 \
    --exchange-override "$exchange"
done
```

The override changes only the prepared winner's `exchange` after preparation
and leaves the cache entry intact. It requires a winner that supports the
requested policy. Results retain the prepared and actual mappings and label
the controlled library result `component_control`; the wrapper verifies that
all other mapping fields match. Exact checks and the native GPU-NTT control
still run. This measures the exchange choice within one prepared geometry;
it is not a comparison of independently tuned automatic winners. Keep these
records separate from the automatic run and inspect the measured samples
before drawing a performance conclusion.

### Diagnose cache sensitivity

`--cache-regime warm` retains the ordinary benchmark behavior. For an artificial
cache-access control, use `--cache-regime l2-displaced --repetitions 1`. It reads
and writes an independent scratch buffer twice the device's reported L2 size
before each timed transform. Inputs, roots, and prepared mappings are unchanged;
the scratch work and stream synchronization are outside the events. Both
backends receive the same control. Unknown L2 capacity or excessive scratch
requirements fail explicitly.

Compare this with a separate `warm --repetitions 1` run using the same request,
library, cache, native dispatch and round count. Do not compare it directly with
the default twenty-transform average. Both regimes drain the stream before
timing. Events surround a Graph launch, so short samples can still include host
submission gaps; this diagnostic is not a pure sum of kernel durations.

CSV and metadata retain the L2 and scratch byte counts, original mapping and
`artificial_l2_displacement_not_application_cache_replay` scope. Displaced
results are component controls, never headline application results. Accessing
twice the L2 capacity does not prove that every cache line was evicted or
reconstruct the application's intervening work. Use the complete application
trace to establish whether a measured sensitivity explains its behavior.

This is an additive RNS path. Its cache is not connected to the general
descriptor/global plan cache or the global installation cost model. Existing
non-RNS application operators continue through their own paths. See the
research principles for matched end-to-end evidence
requirements; this guide makes no performance claim.
