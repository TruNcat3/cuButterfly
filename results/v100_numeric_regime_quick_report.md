# V100 Numeric-Regime Quick Screening

## Scope

This is the first screening pass for the v0.5 numeric-regime study, not a final
cross-library performance table. It ran 736 generated candidates on the same
Tesla V100-SXM2-16GB with three trials per case, 20 warmups, 50 timed repeats,
and a correctness preflight. The matrix covers FP16/BF16 storage with native or
FP32 accumulation, FP32, FP64, uint32 subset-zeta, and uint32/uint64 NTT at the
quick-tier lengths and resource-derived batch anchors.

BF16 native arithmetic is emulated on SM70 by FP32 computation followed by
BF16 narrowing and is labeled `emulated_native=1`. These rows characterize the
numeric contract and mapping response; they are not native V100 BF16 claims.

## Screening Results

| Quantity | Result |
|:--|--:|
| timed samples | 2,208 |
| candidate cases | 736 |
| comparable `(operator, numeric contract, logN, batch)` regimes | 294 |
| detected events | 257 |
| winner crossovers | 96 |
| confirmed crossovers | 29 |
| ambiguous crossovers | 67 |
| numeric-pipeline winner changes | 41 |
| grid-wave boundaries | 108 |
| L2 working-set boundaries | 12 |

A crossover is confirmed only when the new winner is at least 2% faster and
its three-trial range does not overlap the alternative. The high ambiguous
fraction is expected from a screening protocol and identifies the points that
need the full timing protocol, not evidence that those crossovers are absent.

The selected mappings vary across the screen: temporal radix-4 wins 102
regimes, online radix-4 68, hierarchical radix-4 44, online radix-2 36,
temporal radix-2 31, NTT Hybrid2D radix-4 9, and NTT Hybrid2D radix-2 4. These
counts are not a global ranking because the generated batch neighborhoods and
legal length ranges are conditional. Their useful implication is that neither
one radix nor one mapping family dominates all numeric/workload regimes.

## Selector Gate

| Metric | Observed | Promotion gate |
|:--|--:|--:|
| top-1 accuracy | 0.586 | diagnostic only |
| top-3 recall | 1.000 | >= 0.950 |
| geometric-mean regret | 1.047 | <= 1.020 |
| worst regret | 2.078 | <= 1.100 |

All four complete holdout categories have coverage, but regret fails both
promotion gates. The runtime status is therefore `measurement-required`.
Top-3 recall shows the descriptors prune candidates without losing the winner
in this screen; the poor top-1 regret shows that nearest-neighbor latency
transport does not yet model boundary behavior accurately enough for dispatch.

## Next Measurements

1. The 29 confirmed crossover neighborhoods have now been rerun with the full
   protocol: 23 reproduce, 3 reverse, and 3 become statistically ambiguous.
2. Rerun the 67 original ambiguous crossovers only where the runner-up gap or numeric
   contract makes the scientific comparison useful.
3. Collect NCU for winner changes unexplained by grid waves, working-set ratio,
   or the static residency model.
4. Fit piecewise regime models and repeat the complete-regime holdouts before
   changing the public runtime selector.

See [the confirmed follow-up report](v100_numeric_confirmed_followup_report.md)
for the five-trial results.

Raw and derived artifacts are
`v100_numeric_regime_quick_raw.csv`, `v100_numeric_regime_events.csv`,
`v100_numeric_regimes.json`, `v100_numeric_selector_holdouts.csv`, and
`v100_numeric_selector_metrics.json` in this directory. The expanded manifest
is regenerated from `config/v100_numeric_regime_space.json`.
