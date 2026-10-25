# Synthetic Evaluation Layouts

These three 7.0 x 2.2 inch figures organize pending experiments. Every figure
contains `SYNTHETIC EXAMPLE — NOT MEASURED`; every CSV row and export metadata
records `synthetic_example_not_measured`. The fixed seed is 20261009. Values
include wins and losses and are not predictions, measurements, or evidence.

Run with the existing CPU environment:

```sh
/home/wt/yes/envs/cubutterfly/bin/python paper/tools/render_draft_evaluation.py
```

The first run refuses existing outputs. `--overwrite` explicitly replaces only
the named draft outputs. `--input-dir DIR` accepts replacement synthetic CSVs
with the same columns and mandatory synthetic source. The draft renderer never
reads live measurement files, frozen snapshots, calibration, or GPU state.

| Figure | Organization | Interpretation boundary |
| --- | --- | --- |
| `draft/draft_mechanism.pdf` | HBM and resident resources; dependency schematic; warm latency | Fixed stagewise/resident/aligned/overlap core. Resource units are separate. The schematic is neither a kernel trace nor a time axis. |
| `draft/draft_scaling_selection.pdf` | FP32/FP64 N x batch heatmaps; selection regret; preparation cost | Matched FFT contract; explicit missing hatch. Regret is selected / best measured in a declared pool, not a global optimum. |
| `draft/draft_application_results.pdf` | FNO/FFT, QuaRot/FWHT, HEonGPU/NTT, two scales each | Baseline-normalized end-to-end components; total speedup diamonds; preparation excluded and separately reported. No invented error bars. |

PDF and SVG contain vector chart elements. PNG is a 200 dpi preview. Typography
uses the paper's DejaVu Sans default and its blue/green/brown/gray palette, with
7 pt minimum labels at the intended size. The synthetic warning uses 9 pt bold.
`paper/generated/draft/manifest.json` records input/output hashes, units,
normalizations, missing-value policy, required measured columns and replacement
entry. CSVs provide the complete accessible data alternative; the manifest
contains a description of each layout.

Measured replacement needs a reviewed evidence renderer with actual artifact
provenance, paired timing trials, correctness, contract, build and device
identity. Keep profiler counters separate from unprofiled warm latency, define
resident allocation scope, and reconcile component timings with independent
end-to-end timing before publishing a stack. Preparation and compilation must
have explicit timing boundaries. Never remove the watermark or relabel these
example values to manufacture evidence. Working-draft inclusion is controlled
outside this renderer; an evidence build must hide synthetic figures or use
clearly marked pending frames.

This scaffold complements the existing measured RC2 absolute-throughput figure,
the historical NCU profile comparison, and the measured core-ablation table.
It does not regenerate either artifact or assert publication-rule compliance.
No measurement, GPU, TeX, or build inputs are modified by this tool.
