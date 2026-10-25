# Current Evidence Supplement, 2026-10-19

This supplement records evidence added to the evaluation prose. It does not
change the frozen figure inputs, existing hash manifests, or the 36-request
screen plus independent 12-request NTT extension headline statistics.
Historical measurements retain their original build and timing contracts.

| Independent cohort | Evidence used | Interpretation and boundary |
| --- | --- | --- |
| October 17 operator review | [Report](../../docs/install_comparison_results_20261017.md); [analysis summary](../../results/install_compare_repair_20261014/analysis/summary.json) | 66 requests: 29 integer exact and 37 finite floating outputs with same-precision error distributions. GPU baseline gaps and floating acceptance remain explicit; CPU results do not fill GPU columns. |
| Current mechanism v2 | [Six paired timings](../../results/mechanism_current_20261017_v2/summary.csv); [report](../../results/mechanism_current_20261017_v2/report.json); [profile manifest](../../results/mechanism_current_20261017_v2/profile/manifest.json) | FFT/FWHT/word64 NTT at N4096/B4 and N262144/B1. Stagewise/resident8 ratios 10.87–22.44 reflect controlled launch and global-boundary costs, not external-library superiority. The paired profile is a FWHT diagnostic. |
| Constrained selector | [FFT](../../results/selector_ablation_20261017/fft_n16384_b16/), [FWHT](../../results/selector_ablation_20261017/fwht_n16384_b4/), [NTT](../../results/selector_ablation_20261017/ntt_n16384_b4/) | Three mechanisms disabled independently; eight scheduled candidates per space. FFT/FWHT differences are small; NTT resident-fusion/successor-layout regrets are approximately 0.7%/2.9%. NTT successor-layout accepts 8 restricted versus 6 unrestricted candidates after two legality rejections. Independent winner confirmation and broader coverage remain open. |
| Legal current core controls | [Study](../../results/selector_ablation_20261017/core_organization_v2/study.json); [six-row summary](../../results/selector_ablation_20261017/core_organization_v2/summary.csv) | Correctness flags pass. FWHT organization, temporal FWHT core, and specialized NTT core pairs use different supported shapes; they do not complete either operator's two-core/two-organization factorial. The earlier NTT N16384 exploratory controls are not adopted. |
| October 18 fixed-filter Graph | [1024-edge report](../../results/e2e_current_20261018/fft_n1024_b16/image1024_filter31_batch16.current_graph_fast.json); [2048-edge report](../../results/e2e_current_20261018/fft_n2048_b16/image2048_filter31_batch16.current_graph_fast.json) | Independent B16, FP32, shared 31x31 fixed-filter cohort. cuButterfly/CuPy/Torch medians are 1406.293/1500.160/1532.587 us and 5730.816/5903.872/6007.040 us, respectively. All three cache spectra and match fast padding; old nine-point recomputed-filter losses remain separate. |
| October 19 CKKS stage batching | [Report](../../results/ckks_stage_batch_20261019/README.md); [complete table](../../results/ckks_stage_batch_20261019/comparison.md); [artifacts](../../results/ckks_stage_batch_20261019/artifacts/README.md) | 36 configurations, 144 paths and 108 paired comparisons; ciphertexts agree elementwise. All 24 B8/32 configurations beat native loop and native-core batch; all 12 B1 forced-batch cases lose to native loop. Two N65536 rotations prefer cuButterfly loop to cuButterfly batch. |

The fixed-filter reports use actual padding 1080x1080 and 2100x2100,
`filter_spectrum_cached=true` and `execution_mode=graph`. Their inherited
input `contract.filter_contract` still describes the original recomputed
filter; the execution fields and the cached branches of
[the runner](../experiments/e13/run_fft_filter_end_to_end.py) determine the
measured boundary. All backends prepare their invariant spectrum outside
timing. The Graph call includes input device copy, image padding, forward
transform, product, inverse transform, cropping and independent output copy;
plan selection and capture are excluded. Changed/restored input and prior
output preservation checks pass. Each report has one process, five event-timed
samples of three complete replays after two warmups. Finite same-precision
error distributions remain unassessed against a user acceptance threshold.
These two measurements do not establish general filter superiority or a
like-for-like speedup over the old nine-point cohort.

The CKKS table in the manuscript reproduces the report's separate geometric
means for each method/N and baseline. Method I uses N16384/65536 and method II
uses N16384, across key switching, square/relinearize/rescale, that chain plus
key switching, and rotation. Both batch backends share graph organization,
gather/scatter and non-NTT nodes; native-core batch is an integration control,
not an upstream batch API. Its comparison controls application organization
but still permits different NTT implementations and mappings.

The final thermally guarded files use twenty rotated measurement rounds,
ten complete Graph replays per sample, and dynamic clocks. Packing, NTT,
writeback and non-NTT work are timed; setup, tuning, capture, encryption and
checking are excluded. Q=[50,40,40], P=[50] or [50,50], scale=2^40 and
sec_level=none define an integration context, not production-security
validation. Complete operations and short chains do not constitute full
CKKS or TFHE bootstrap. Loop/batch application selection is not established
by these measurements; RNS tuning remains request-local rather than an
installation-model hit.

Remaining evidence includes independent selected-winner confirmation,
broader fixed-core factorial coverage, online search/setup costs, current
cross-architecture migration, floating numerical acceptance, missing matched
GPU baselines, and unfinished application/Graph and full-bootstrap work.
Historical image-filter, FNO and QuaRot losses remain in the evaluation.
The existing figure hash manifest continues to describe only its own frozen
inputs and is not a manifest for this supplement.

Validation for this text update: `make -C paper all evidence` passed. Both
outputs have 14 Letter pages and match their reader copies. Their final logs
contain no undefined references/citations, missing characters, or overfull
boxes. Underfull spacing and the existing imported-PDF version warnings
remain. Figures were not regenerated; this check did not include a new visual
inspection of the PDFs. The older 13-page validation record describes its
dated revision, not this build.
