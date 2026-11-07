# Frozen application cohort supplement

`application_closeout.json` contains all 14 comparison rows from the audited
nine-process A100 80GB application queue, plus four historical FNO hotpath
rows (Graph/eager kept separate). Each row retains absolute latency, ratio,
source path and SHA256, and library identity. Source paths are provenance
labels; the renderer does not open them.

`application_cohorts.tex` contains the original eight-request external Graph table
and a separate block for two fused QuaRot requests, when their frozen rows exist.
Queue entries show every process's ratio range; FNO shows the recorded median
of two process medians. `application_closeout.tex` retains all 18 individual
comparison/mode rows, including four same-padding filter diagnostics and eager
FNO. The JSON also retains individual FNO process ratios. No best-of-process
or cross-build aggregate is used.

The application figure's upper panel preserves the original snapshot and two
numerically checked historical FNO observations. Two early QuaRot observations
remain archived but are excluded because random-state preparation was invalid
and outputs were not checked. The figure manifest records both exclusions.
Its lower panel adds all 14 audited queue
ratios and two historical FNO Graph aggregate ratios, plus four fused QuaRot
processes (Qf, green). Both panels label their
own ratio ranges. Closure points use library `fac109…`; FNO hotpath uses
`ddb7d7…`. The fused QuaRot follow-up uses `775a85…`; it is not pooled with them. Filter
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

`quarot_fused_e2e.json` preserves all four full-model process records, both clock
samples, numerical errors, selected plans, and stored identities. Its exporter
recomputes wall/event medians and checks clean completion before emitting the
two table rows and two numeric macros. B1 and B4 each use both process orders;
the table shows the range, not the best process. This follow-up keeps FP16 storage,
FP32 FWHT arithmetic, the same 7B random-weight model, and full Graph-call boundary.
The figure manifest also hashes this frozen input. Regenerate it from local raw
evidence with `python paper/tools/export_quarot_fused_followup.py`.
