# Independent CPU preparation for installation search

Installation search now accepts a frozen candidate inventory and a persistent
CPU score database. Completed GPU cells and measurements retain their original
identities. This changes preparation and recovery, not the candidate space,
adaptive model updates, top-four resolution rule, or timing protocol.

The workflow separates three operations:

1. `capture_calibration_cpu_inputs.py` snapshots the existing device profile,
   training journal, descriptors and seeds. It exports each pending workload's
   full ordered inventory from the current library. This capture can query CUDA
   hardware and is **not CPU-only**; it performs no timed measurements. Completed
   cells are skipped. Inventories are compressed JSONL with checksums.
2. `prepare_calibration_cpu.py` hides all GPUs, scores the full inventory under
   the captured model, and saves scores, stable rankings and preparation status.
   It precompiles a configurable priority set through the existing broker.
   Saved `needs-compilation` configurations enter the queue first, including
   those from completed cells; their measurement records remain intact.
   The default top 32 plus exploration and seed candidates is a compilation
   priority, never a limit on search coverage. Unprojectable requests and
   compiler failures remain visible. Preparation is not GPU correctness evidence.
3. `calibrate_local_hardware.py --cpu-preparation-input INPUT
   --cpu-preparation-dir OUTPUT` checks the preparation receipt and consumes
   the exact inventory and scores. `watch_idle_campaign.py
   --cpu-preparation-status OUTPUT/status.json` requires both `scoring_complete`
   and `compile_ready` before starting the GPU idle window.

An explicit inventory-only mode uses `--cpu-preparation-input INPUT
--cpu-preparation-inventory-only` without `--cpu-preparation-dir`. It validates
the frozen inventory and reuses existing compiled modules, then computes exact
scores for the current model in memory. It reads no old score database and
claims no score-cache hits from an earlier model. The refresh driver exposes
the same mode as `--calibration-cpu-input INPUT
--calibration-cpu-inventory-only`, without `--calibration-cpu-dir`.

For an interrupted refresh, `prepare_refresh_calibration.py` provides one entry
point for these preparation steps. After the owned measurement driver stops,
it restores the campaign's recorded build/JIT environment and checks the
checkpoint's measurement identity. It first freezes and compiles saved
`needs-compilation` configurations, then captures the full inventories for
pending workloads and runs offline scoring and priority precompilation.
Capture may query the selected GPU's properties; both compilation stages and
scoring hide CUDA devices. This entry point performs no timed measurements.

```bash
python scripts/prepare_refresh_calibration.py --campaign CAMPAIGN \
  --output OUTPUT --jobs 4 --compile-top 32
```

The resulting full input is `OUTPUT/capture/input.json`, with scores and
readiness in `OUTPUT/prepared`. The earlier compilation-only receipt lives in
`OUTPUT/compilation_recovery`; it does not claim scoring completion. Point the
watcher at `OUTPUT/prepared/status.json`.

The shared SQLite database uses complete model content, build/tool context,
candidate ID and descriptor content as keys. It commits batches of at most
65,536 scores in primary-key order, with the existing one-second limit on
subsequent writes and SQLite `synchronous=FULL`. Online search keeps exact
full-inventory scores in memory and, when using a score database, persists only
the shortlist and selected points before GPU work. Offline preparation still
persists every score. Interruptions reuse compatible committed scores and
recompute missing scores exactly; successful measurements and selection records
remain durable. Resolved descriptors invalidate only their own entries, and a
changed model or profile selects a different score epoch. Stable ties use the
online search's original stratified order; an offline ranking never replaces it.

The bounded [CPU/I/O comparison](../tests/benchmark_calibration_cpu_cache_io.py)
used 16,384 representative candidate keys under three model epochs. Sorting and
batching reduced cache workload time from 6.584 to 1.208 seconds and process
`write_bytes`, including close-time checkpointing, from 298,545,152 to 21,356,544
bytes (92.85%). All 49,152 persisted scores matched bit-for-bit after reopening.
These figures measure CPU/cache work, not GPU speedup.

Future adaptive rankings cannot be computed exactly before their GPU feedback
exists. The frozen preparation supplies the current model and ahead-of-time
modules; later rounds fit their new models and recompute exact scores. This path
does not freeze the model, approximate missing feedback, or claim that every
future compilation is already covered. Actual plan/resource descriptions still
require the GPU-capable probe.

Example after capturing `INPUT`:

```bash
python scripts/prepare_calibration_cpu.py --input INPUT --output OUTPUT \
  --broker EXISTING_BROKER --jobs 4 --compile-top 32
```

`--skip-compile` publishes scoring results alone. `--compile-only` compiles saved
`needs-compilation` configurations without fitting a model or scoring inventories;
use a separate output directory for that receipt. Offline compilation temporarily
clears the installation compilation-budget variables and restores them afterward.
Repeating full preparation reuses compatible committed scores. `status.json`
separates scoring completion, compilation readiness and GPU verification;
`compilation.json` records gaps.
Pending requests, projection failures and compiler failures prevent
`compile_ready`; a linked lowering that needs no JIT can be ready.

For research recovery, `calibrate_local_hardware.py --resume-search
--retry-compilation-gaps` explicitly revisits selected, unmeasured compilation
gaps, including gaps in completed cells. It retains successful measurements,
selection order and prior failed attempts, then confirms the current finalists.
Supply the matching `--cpu-preparation-input` and `--cpu-preparation-dir` to
consume offline results, or use the inventory-only mode above.
`run_full_semantic_refresh.py` accepts the corresponding input/directory pair or
the input/inventory-only pair; either enables gap retries and sets the research
continuation's `--compile-seconds` to `0`.
Later adaptive GPU feedback can still require CPU scoring and new compilation.

The capture tool can create a new manifest with `--rebind-score-context` after a
CPU implementation update. It checks all frozen inputs and unchanged measurement,
partition and projection identities, retains the original manifest, and records
the context transition without re-exporting candidates.

The October 15 local campaign captured 23 cells, skipped 10 already complete
cells, and exported 695,798 candidate records for the remaining 13. Its active
input is `results/install_compare_repair_20261014/cpu_inputs/active.json`; CPU
progress and outputs are under the sibling `cpu_preparation` directory.

Validation: 77 affected CPU tests passed, and ten actual persisted candidate
scores were read back through the online cache with exact agreement to fresh
predictions and no repeated feature preparation. A broader test run also found
the existing `test_compiled_product_preserves_asymmetric_axes_and_budget_coverage`
assertion incompatible with the previously expanded batch-tile stratification:
its budget of 24 no longer covers every partition/twiddle/layout combination.
This change does not alter that stratification or relax the assertion; the
separate coverage-test mismatch remains open.
