# Reproducible experiments

This directory holds executable experiments independently of the manuscript.

- `workloads/` contains workload definitions, target configurations and application runners. The `e13/` runners cover FFT filtering and QuaRot; the CKKS and TFHE runners use pinned HEonGPU dependencies.
- `tools/` contains CPU preparation, GPU measurement, mechanism controls, profile analysis, result export and plotting programs.
- `snapshots/` contains frozen JSON/CSV evidence used by result checks and visualizations.
- `provenance/` preserves source manifests and historical evidence identifiers. Historical source paths and hashes describe the original measurements; directory relocation does not revalidate or remeasure them.

Start with the public configuration-driven runner in `benchmark/README.md` for operator comparisons. Use `python experiments/tools/prepare_experiments.py --help` for the bounded experiment queue, `prepare_end_to_end.py --help` for application preparation, and `prepare_core_dataflow_ablation.py --help` for controlled mechanism experiments. Preparation and compilation run on the CPU; measurement requires an available GPU. Check each program's help for explicit input and output locations.

Frozen snapshots are reusable evidence, not new measurements. Some historical preparation programs require raw result bundles that are not included in a source checkout; they report the missing inputs rather than reconstructing or inventing results. Store new measurements in a separate results directory and retain their semantic, hardware, build and timing identities.

The [PRO6000 fixed-mapping sample report](reports/PRO6000_FIXED_MAPPING_SAMPLES_20261117.md)
publishes target measurements for 25 matching seed requests and highlights four
requests with nominal A100 wins and matching published mapping fields. It retains
target losses, integer baseline failures and the 41 unmatched Full66 requests;
it does not close the comprehensive platform acceptance.

The [PRO6000 FFT17/MMA9 report](reports/PRO6000_EXTENSIONS_20261117.md)
adds the complete finite extension cohorts: 17 FFT requests with 47 measured
external pairs and four provider gaps, plus nine ALLOW_FP16_MMA comparisons
after local native/tile calibration and 68 public AUTO checks. Each cohort
keeps its own GPU/build/profile identity and floating error-review scope.

The [PRO6000 Full66 report](reports/PRO6000_FULL66_20261118.md) covers the
complete declared main request population across all six operators. It retains
per-request public MEASURE selection, actual mappings, original checkpoint
identities, unsupported comparators and three GPU-NTT integer failures. Measured
requests preserve seven complete-call samples; a CPU audit replays the recorded
metrics from lossless actual/reference arrays. Floating accuracy review and the
original formal driver's status remain explicit. The snapshot also includes the
measured workflow/build patch and CPU planning source receipts.

The [PRO6000 raw performance download](reports/PRO6000_RAW_DATA_20261118.md)
provides CSV tables and 1,729 original timing samples, plus the published raw
JSON records in a small ZIP with checksums. Large output arrays and binaries
remain separate on the measurement host; numerical review and failures are retained.

The [bounded-search regression report](reports/PRO6000_SEARCH_FIX_20261118.md)
records the subsequent FP64 N=4096 diagnosis and new-build automatic-search fix,
with separate GPU0 paired measurements and original search/timing records.
These two-request results do not replace the frozen main or extension cohorts.

The [PRO6000 Figure 9 extension](reports/PRO6000_FIGURE9_20261118.md)
adds all 83 local CPU Full66 measurements, nine explicit FP16-I/O/FP32-CPU
adapted references and a full independent final-build FFT17 paired rerun.
It publishes every request/backend cell, CPU-normalized figures, source/input
identities, support boundaries and the root/order/residue audit of the three
retained GPU-NTT failures. Original and final-build cohorts remain separate.
Verify with `python3 experiments/tools/report_pro6000_figure9.py --check`.

The [finite Figure9 follow-up](reports/PRO6000_FIGURE9_FOLLOWUP_20261118.md)
separates backend status from campaign status, adds the predeclared legal
cyclic NTT8 population (six valid GPU-NTT pairs, two retained external failures),
and confirms only the 83 plot CPU denominators with three independent processes.
CPU medians remain provisional because selected cores had observed interference.
The standalone record package includes all earlier source checkpoints and
verifier dependencies; it runs the same `--check` from an empty directory
without GPU, external Python packages, output-array payloads or binaries.
