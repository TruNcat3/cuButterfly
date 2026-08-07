# V100 Numeric-Regime Confirmed Follow-Ups

| Status | Events |
|:--|--:|
| confirmed-reversed | 3 |
| confirmed-same-direction | 23 |
| not-confirmed | 3 |

A full-protocol event requires at least a 2% median gap and non-overlapping five-trial ranges.

The full protocol repeats 149 candidate/neighbor cases for 745 timed samples.
Twenty-three of the 29 quick-screen directions reproduce with the stricter
criterion. Three batch events reverse direction and remain statistically
separated; three more batch events retain a median preference but their trial
ranges overlap. All nine length-axis events reproduce in the same direction.

| Axis | Same direction | Reversed | Not confirmed |
|:--|--:|--:|--:|
| length | 9 | 0 | 0 |
| batch | 14 | 3 | 3 |

This result supports a piecewise length/resource regime, but shows that a
single quick timing near a grid-wave boundary is insufficient for dispatch.
The three reversals are FP32 Structured 2x2 at `logN=15,batch=8`, FP16/FP32 FFT
at `logN=8,batch=1281`, and uint64 NTT at `logN=15,batch=241`. They should be
treated as boundary neighborhoods rather than mislabeled as stable winners.

The three unconfirmed cases are FP16/FP32 FFT at `logN=15,batch=5`, uint32
subset-zeta at `logN=8,batch=1281`, and FP64 FFT at `logN=8,batch=961`.
Their median gaps exceed 2%, but the five-trial ranges overlap, so the evidence
does not support a deterministic winner.

The remaining counter capture is reduced to these six unstable events. Run
`sudo -E scripts/profile_numeric_boundaries_ncu.sh`; it emits 12 paired profiles
and `results/ncu_numeric_boundaries/summary.csv`.

The capture is complete. [Paired NCU attribution](v100_numeric_boundary_ncu_analysis.md)
shows that three radix-4 reversals exchange a larger register footprint for
fewer instructions, the Structured reversal comes from online composition and
barrier work, and the FP64 near-tie is the only pair that changes the actual
resident-CTA capacity bound.
