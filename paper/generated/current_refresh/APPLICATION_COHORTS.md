# Frozen application cohort supplement

`application_closeout.json` contains all 14 comparison rows from the audited
nine-process A100 80GB application queue, plus four historical FNO hotpath
rows (Graph/eager kept separate). Each row retains absolute latency, ratio,
source path and SHA256, and library identity. Source paths are provenance
labels; the renderer does not open them.

`application_cohorts.tex` is the compact eight-request external Graph table.
Queue entries show every process's ratio range; FNO shows the recorded median
of two process medians. `application_closeout.tex` retains all 18 individual
comparison/mode rows, including four same-padding filter diagnostics and eager
FNO. The JSON also retains individual FNO process ratios. No best-of-process
or cross-build aggregate is used.

The application figure's upper panel preserves the original snapshot and four
legacy FNO/QuaRot observations. Its lower panel adds all 14 audited queue
ratios and two historical FNO Graph aggregate ratios. Both panels label their
own ratio ranges. Closure points use library `fac109…`; FNO hotpath uses
`ddb7d7…`. Neither measures the installed FP16 library `775a85…`. Filter
floating errors remain report-only; bootstrap uses `sec_level=none` and has
unverified source/binary correspondence. Random weights do not establish
trained-model quality. These limits remain in the exported JSON.

CPU refresh from the original local evidence (explicit evidence acceptance):

```bash
python paper/tools/export_application_closeout.py \
  --fp16-acceptance results/a100_fp16_acceptance_20261105
```

CPU redraw in a source release, without a `results` directory:

```bash
python paper/tools/render_current_refresh.py --figures-only --frozen-inputs
```

The latter uses `snapshot.json`, `fp16_comparison.json`,
`application_closeout.json`, and the existing repository-frozen
`paper/generated/application_historical_complete_call.json` and
`motivation_organization_a100.csv`. The figure manifest hashes these inputs.
Main66, cyclic90, CKKS and numeric macros are unchanged. Original raw files and
legacy observations remain archived.
