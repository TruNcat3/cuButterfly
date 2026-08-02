# V100 Orthogonal Length/Batch Scaling

`Batch@90%` is the smallest measured batch reaching 90% of that
implementation's peak measured `Gpoint/s` at fixed transform length.
Rows below the configured 0.020 ms timing floor are retained but are not
used as stable latency claims.

## Saturation Summary

| Operator | Numeric | logN | Implementation | Batch@90% | Peak batch | Peak Gpoint/s | Batch-1 ms |
|:--|:--|--:|:--|--:|--:|--:|--:|
| fft | fp32 | 8 | cuFFT | 1 | 1 | 0.095 | 0.002683 |
| fft | fp32 | 18 | VkFFT | 1 | 1 | 14.191 | 0.018473 |
| fft | fp32 | 18 | cuButterfly-cuFFTDx-online | 1 | 1 | 15.920 | 0.016466 |
| fft | fp32 | 20 | cuButterfly-cuFFTDx-online | 1 | 1 | 13.586 | 0.077179 |
| fwht | fp32 | 8 | cuButterfly-warp-register | 64 | 64 | 6.867 | 0.002580 |
| fwht | fp32 | 15 | cuButterfly-online-radix4 | 1 | 1 | 4.250 | 0.007711 |
| xor-zeta | uint32 | 8 | cuButterfly-radix2 | 64 | 64 | 5.333 | - |
| xor-zeta | uint32 | 8 | cuButterfly-radix4 | 1 | 1 | 0.101 | 0.002540 |
| xor-zeta | uint32 | 20 | cuButterfly-online-radix4 | 1 | 1 | 11.652 | 0.089989 |

## Detailed Scaling

| Operator | logN | Batch | Implementation | Median ms | Gpoint/s | Peak fraction | vs basis | Region | Timing quality |
|:--|--:|--:|:--|--:|--:|--:|--:|:--|:--|
| fft | 8 | 1 | cuFFT | 0.002683 | 0.095 | 1.000 | 1.000x | saturated | below-timing-floor |
| fft | 18 | 1 | VkFFT | 0.018473 | 14.191 | 1.000 | 0.891x | saturated | below-timing-floor |
| fft | 18 | 1 | cuButterfly-cuFFTDx-online | 0.016466 | 15.920 | 1.000 | 1.000x | saturated | below-timing-floor |
| fft | 20 | 1 | cuButterfly-cuFFTDx-online | 0.077179 | 13.586 | 1.000 | 1.000x | saturated | unstable |
| fwht | 8 | 1 | cuButterfly-warp-register | 0.002580 | 0.099 | 0.014 | 1.000x | launch-limited | below-timing-floor |
| fwht | 8 | 64 | cuButterfly-warp-register | 0.002386 | 6.867 | 1.000 | 1.000x | saturated | below-timing-floor |
| fwht | 15 | 1 | cuButterfly-online-radix4 | 0.007711 | 4.250 | 1.000 | 1.000x | saturated | below-timing-floor |
| xor-zeta | 8 | 64 | cuButterfly-radix2 | 0.003072 | 5.333 | 1.000 | 1.000x | saturated | below-timing-floor |
| xor-zeta | 8 | 1 | cuButterfly-radix4 | 0.002540 | 0.101 | 1.000 | 1.000x | saturated | below-timing-floor |
| xor-zeta | 20 | 1 | cuButterfly-online-radix4 | 0.089989 | 11.652 | 1.000 | 1.000x | saturated | stable |
