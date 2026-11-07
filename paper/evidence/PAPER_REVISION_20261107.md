# Paper prose revision, November 7

Later same-day follow-up: the [matched path and full-model report](../../docs/quarot_fused_path_20261107.md)
adds four clean 7B processes on the fused path, with ratios 1.004–1.005 at B1
and 1.015–1.017 at B4. They appear separately in the application table and
figure; the earlier negative records remain. A performance-contract audit also
corrects CKKS/bootstrap mean/median labels and discloses the supplied-output
copy in the main DaoFHT comparison. Native-return FWHT for those ten requests
is still unmeasured. The follow-up PDFs are 15 pages, including references;
the application table and figure were checked at their rendered sizes.
The same audit removes two early QuaRot traces from plotted performance evidence:
they preceded the random-state repair and had no output gate. Their records and
hashes remain archived. The two historical FNO records do have numerical gates;
their ratios now use unrounded raw medians (three-decimal labels are unchanged).
The earlier prose-only revision and its checks follow.

This revision changes prose in abstract, introduction, evaluation, related work
and conclusion. It performs no new experiment and does not change plotted data
or headline operator, NTT, or CKKS aggregates.

The contribution is the butterfly-specific derivation of dependency-closed
regions, communication/readiness relations and resource demands, used by a
hierarchical hybrid organization method across arithmetic cores, workloads and
hardware constraints. General spatial/temporal dataflow, search itself and
universal hardware optimality are not claimed as contributions.

The four evaluation subsections remain Experimental setup, Performance
comparisons, Ablations and Analysis. Build/device differences are consolidated
in setup. Repeated protocol descriptions and repeated scaling ratios are
shortened; unfavorable FP16, large-length FFT/FWHT, small-batch NTT, deeper
grouping, overlap, pipeline and cross-core results remain visible. Floating
acceptance thresholds remain unset. No Blackwell measurement is asserted.

Application additions use the independently frozen
[application closeout snapshot](../generated/current_refresh/application_closeout.json)
and its generated table. Four random-weight QuaRot process measurements remain
losses (rounded range 0.957–0.968). Their recorded FP16 adapter composes
FP32 conversion, FWHT, scaling and narrowing; no trace-based attribution
of that overhead is claimed. See the [QuaRot audit](../../docs/quarot_attribution_20261107.md). FNO Graph aggregates at 64²/128², B4, are
1.030/1.056 from two processes per size; eager results remain separate. The
original four historical records and 46-comparison population are retained;
no cross-build or cross-protocol mean is introduced. Full build/source evidence
remains in the [A100 closeout record](A100_CLOSEOUT_20261107.md).

Before revision, evaluation.tex contained 3,094 whitespace-delimited words.
The revised file is approximately 15% shorter, including the new application
cohort discussion and table caption. This is a prose-size measure, not rendered
page reduction. Local checks cover braces, duplicate labels, input paths,
subsection structure, headline macros and preserved negative-result ranges.
Integration built both 14-page PDFs with no unresolved citations or cross-references. The rendered operator, application and ablation pages were inspected; plots and tables fit their columns. The new application table uses Scenario, Type, Baseline and Ratio. The publication checkout passes 351 focused CPU tests; 11 additional tests cover explicit local-artifact link handling. No new accepted QuaRot GPU measurement was obtained: its bounded idle window was interrupted by another GPU process.
