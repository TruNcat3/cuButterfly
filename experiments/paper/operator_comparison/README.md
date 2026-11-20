# Complete-transform comparison on two platforms

```sh
python render.py --output-dir output
```

The Figure 9 renderer reconstructs 268 A100 and 143 RTX PRO 6000 observations
from measured CPU and GPU times, checks all 411 request/backend values against
the frozen numerical export, and writes PDF, SVG, PNG, CSV and a provenance
manifest. The shared logarithmic axis includes every observation, including
CPU/GPU ratios below one. The dashed line denotes CPU parity; boxes show
quartiles, the median and min/max whiskers, with individual requests overlaid.
They describe workload distributions, not confidence intervals.

`data/` preserves the full supplied PRO 6000 backend, request, CPU, coverage and
statistics tables, including failed and unsupported records. The A100 inputs
include measured paired times, numerical statuses, protocol metadata and the
original matched-population audit. Only personal locations and device/host
identifiers are replaced. `provenance.json` separates original source hashes
from hashes of the actual bundled bytes.

The normalization is `CPU time / backend complete-call time`. All backends
for a request use the same named CPU denominator. FFT uses FFTW with eight
threads. FP16 FWHT uses the explicitly adapted FP16-input/output, FP32-compute
CPU baseline, including conversion and scaling. GPU Tensor Core arithmetic
is an allowed numerical option; this comparison does not assert identical
internal rounding. Floating-point error-review statuses remain in the input
records. CPU measurements were collected on shared hosts and are qualified
accordingly.

A100 primary groups use the intersection of available backends. FFT8 is a
separate supplemental measurement population, and cyclic NTT is separate from
the main population. PRO 6000 uses `fft17-final`, `full66` and `mma9-original`
as independently collected cohorts. Its boxes retain usable measurements per
backend, so box populations can differ; GPU/GPU geometric means must instead
use matched pairs from the statistics tables. No rows are pooled across
platforms. The separate PRO 6000 cyclic NTT8 follow-up remains in the stored
tables but is not mixed into these CPU-normalized boxes.

This command is offline regeneration. To collect new platform measurements,
use the repository's platform runner and explicit workload manifests, install
the requested CPU/GPU providers, calibrate locally, and retain unsuccessful
requests in coverage accounting. A new run is a new cohort, not a replacement
for these frozen records.
