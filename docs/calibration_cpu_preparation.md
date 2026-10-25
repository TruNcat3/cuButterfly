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
   The default top 32 plus exploration and seed candidates is a compilation
   priority, never a limit on search coverage. Unprojectable requests and
   compiler failures remain visible. Preparation is not GPU correctness evidence.
3. `calibrate_local_hardware.py --cpu-preparation-input INPUT
   --cpu-preparation-dir OUTPUT` checks the preparation receipt and consumes
   the exact inventory and scores. `watch_idle_campaign.py
   --cpu-preparation-status OUTPUT/status.json` waits for `scoring_complete`
   before starting the GPU idle window. Compilation may continue independently.

The shared SQLite database uses complete model content, build/tool context,
candidate ID and descriptor content as keys. It commits partial batches and
flushes the shortlist before GPU work. Interruptions can reuse completed CPU
scores, and resolved descriptors invalidate only their own entries. A changed
model or profile selects a different score epoch. Stable ties use the online
search's original stratified order; an offline ranking never replaces it.

Future adaptive rankings cannot be computed exactly before their GPU feedback
exists. The frozen preparation supplies the current model and ahead-of-time
modules; later rounds fit their new models and persist new scores. This path
does not freeze the model, approximate missing feedback, or claim that every
future compilation is already covered. Actual plan/resource descriptions still
require the GPU-capable probe.

Example after capturing `INPUT`:

```bash
python scripts/prepare_calibration_cpu.py --input INPUT --output OUTPUT \
  --broker EXISTING_BROKER --jobs 4 --compile-top 32
```

`--skip-compile` publishes scoring results alone. Repeating the command reuses
compatible committed scores. `status.json` separates scoring completion,
compilation readiness and GPU verification; `compilation.json` records gaps.
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
