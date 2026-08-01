# V100 Calibrated Mapping Selector

## Scope

The first selector ranks already measured internal candidates for a covered
V100 semantic contract. It is deliberately not a cross-GPU model and does not
invent support for unseen operators, lengths, precision, layout, or output
order.

The selector reads the generated scaling manifest, measured scaling summary,
and V100 hardware profile. It reports the mapping family separately from the
processing unit, checks basic hardware legality, interpolates log-latency over
the remaining batch measurements, and attaches resource-oriented explanations
such as batch concurrency and working-set size relative to L2.

## Validation

Every complete `(operator, precision, logN, batch)` shape is held out before
its candidates are ranked. Thus the target timing is not used in its own
prediction.

| Metric | V100 result |
|:--|--:|
| evaluated multi-candidate shapes | 49 |
| operators | FWHT, NTT, XOR-zeta |
| top-1 accuracy | 93.88% |
| top-3 recall | 100.00% |
| geometric-mean regret | 1.0049x |
| worst regret | 1.1532x |
| top-1 candidate reduction | 50.00% |

NTT selects the measured winner for all 14 shapes. The three misses are FWHT
`logN=15`, batch 4 at the online-to-warp crossover, and the extrapolated
largest-batch points for FWHT and XOR-zeta `logN=8`. These are useful failure
boundaries rather than rows to hide.

Current groups contain only two internal candidates. Top-3 recall is therefore
an acceptance check, not evidence of aggressive top-3 pruning. FFT is also
excluded from the internal validation because the scaling manifest currently
contains one cuButterfly implementation per FFT length; cuFFT and VkFFT remain
external baselines rather than architecture candidates.

The separate FFT pipeline evaluation closes that gap over 48 internal
candidates per shape. Static-only ranking is insufficient, but
leave-one-batch-out calibration reaches 1.0007x top-3 geometric-mean regret and
1.0020x worst regret while returning 6.2% of candidates. See
[`v100_fft_pipeline_model_report.md`](../results/v100_fft_pipeline_model_report.md).

## Commands

```bash
python3 scripts/select_mapping.py --evaluate --top-k 3 \
  --evaluation-output results/v100_mapping_selector_evaluation.csv \
  --metrics-output results/v100_mapping_selector_metrics.json

python3 scripts/select_mapping.py \
  --operator fwht --precision fp32 --logN 15 --batch 16 --top-k 2
```

An unseen shape fails with `measurement is required`; it is not silently
assigned the nearest measured configuration.
