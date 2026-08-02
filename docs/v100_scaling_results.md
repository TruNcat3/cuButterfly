# V100 Orthogonal Length/Batch Scaling

This experiment separates transform length from batch instead of fixing only
`N * batch`. It measures 184 cases and 920 independent process trials across
FFT, NTT, FWHT, and XOR-zeta on the Tesla V100-SXM2-16GB. Every implementation
passes a batch-1 correctness preflight. The compact source space is
[`v100_scaling_space.json`](../config/v100_scaling_space.json); its generated
manifest is [`v100_scaling_suite.json`](../config/v100_scaling_suite.json).

## Protocol

- fixed `logN`, independently swept power-of-two batch;
- no case exceeds `2^24` resident points;
- 1000 warmups, 100 measured repetitions, and five process trials;
- randomized case/trial order and durable per-sample output;
- peak and `Batch@90%` exclude unstable rows and rows below 0.020 ms;
- the V100 application clock is 1312 MHz and boost clock is 1530 MHz, so
  process-level bimodality remains explicit rather than silently filtered.

The complete generated table is
[`v100_scaling_full_report.md`](../results/v100_scaling_full_report.md), with
raw samples in [`v100_scaling_full_raw.csv`](../results/v100_scaling_full_raw.csv).

## Saturation Points

`Batch@90%` is the smallest measured batch reaching 90% of an implementation's
qualified peak measured point throughput.

| Operator | logN | Implementation | Batch@90% | Peak batch | Peak Gpoint/s |
|:--|--:|:--|--:|--:|--:|
| FFT | 8 | cuButterfly cuFFTDx | 4,096 | 65,536 | 51.206 |
| FFT | 14 | cuButterfly cuFFTDx direct | 1,024 | 1,024 | 44.142 |
| FFT | 18 | cuButterfly online | 2 | 16 | 20.450 |
| FFT | 20 | cuButterfly online | 4 | 8 | 18.412 |
| FWHT | 8 | warp-register | 16,384 | 65,536 | 102.056 |
| FWHT | 15 | warp-register | 256 | 512 | 86.876 |
| FWHT | 20 | online radix-4 | 2 | 16 | 13.609 |
| NTT | 12 | Hybrid2D radix-4 | 1,024 | 4,096 | 31.167 |
| NTT | 16 | Hybrid2D radix-4 | 64 | 256 | 15.687 |
| NTT | 20 | compact stage | 2 | 16 | 12.769 |
| XOR-zeta | 8 | radix-4 | 16,384 | 65,536 | 95.802 |
| XOR-zeta | 20 | online radix-4 | 2 | 16 | 13.617 |

The required batch falls rapidly as a transform exposes more internal
parallelism. It is therefore incorrect to treat one fixed batch or one fixed
total-point workload as a hardware-independent mapping property.

## FFT Against cuFFT

Ratios are cuButterfly throughput divided by cuFFT throughput at the same
length, batch, precision, direction, placement, and stride.

| logN | Batch range | Observed ratio | Interpretation |
|--:|:--|:--|:--|
| 8 | 4,096-65,536 | 0.994x-1.000x | parity after timing-floor and saturation filtering |
| 14 | 1-64 | 0.578x-0.872x | direct core lacks enough independent CTAs |
| 14 | 256-1,024 | 0.998x-1.069x | catches and then exceeds cuFFT after saturation |
| 18 | 2-16 | 1.030x-1.150x | online composition wins in the intermediate-batch region |
| 18 | 32-64 | 0.921x-0.840x | cuFFT reaches a higher large-batch ceiling |
| 20 | 1-16 | 0.778x-0.959x | gap remains; best ratio occurs at batch 2 |

This exposes two different FFT limitations. At `logN=14`, the local direct
unit is strong but needs batch concurrency. At `logN=18`, cuButterfly saturates
early near 20.45 Gpoint/s while cuFFT continues to 23.75 Gpoint/s. At
`logN=20`, both early saturation and a lower ceiling remain.

## Mapping Crossovers

The best processing unit is not constant over batch:

| Workload | Low-batch choice | Crossover | Saturated choice |
|:--|:--|:--|:--|
| FWHT FP32 `logN=15` | online radix-4 | between batch 4 and 16 | warp-register, up to 6.445x at batch 512 |
| FWHT FP32 `logN=20` | online radix-4 | none in measured range | online, about 1.52x for batch 2-16 |
| NTT 30-bit `logN=12` | radix-4 | none | radix-4, 1.11x-1.18x |
| NTT 60-bit `logN=16` | radix-4 | none | radix-4, advantage grows to 1.104x |
| XOR-zeta `logN=8` | radix-4 | none | radix-4, advantage contracts from 1.165x to 1.022x |

The FWHT crossover is direct evidence that batch unfolding belongs in the
architecture mapping. The shrinking XOR radix advantage shows the opposite
effect: once memory throughput dominates, local arithmetic organization has
less influence.

## Timing Quality

Of 184 cases, 101 are stable, 13 are stable with one reported outlier, and 69
fall below the 0.020 ms timing floor. One original `fft14_cufft_b64` row is
unstable because it crosses V100 clock modes; an independent five-trial
confirmation is stable but lands on the other clock mode, so the original row
is excluded from saturation selection rather than replaced. No low-duration
row is used to claim a latency advantage.

## Next Profiling Targets

1. FFT `logN=14`, batch 16/256/1024: attribute the direct-unit concurrency
   threshold to CTA count, occupancy, and launch amortization.
2. FFT `logN=18`, batch 2/16/64: explain why cuButterfly saturates early while
   cuFFT continues scaling.
3. FFT `logN=20`, batch 2/8/16: separate the intermediate-batch near-parity
   point from the lower large-batch throughput ceiling.
4. FWHT `logN=15`, batch 4/16: measure the online-to-warp mapping crossover.

