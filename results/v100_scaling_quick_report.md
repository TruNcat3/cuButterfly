# V100 Orthogonal Length/Batch Scaling

`Batch@90%` is the smallest measured batch reaching 90% of that
implementation's peak measured `Gpoint/s` at fixed transform length.
Rows below the configured 0.020 ms timing floor are retained but are not
used as stable latency claims.

## Saturation Summary

| Operator | Numeric | logN | Implementation | Batch@90% | Peak batch | Peak Gpoint/s | Batch-1 ms |
|:--|:--|--:|:--|--:|--:|--:|--:|
| fft | fp32 | 8 | VkFFT | 16384 | 16384 | 50.178 | 0.002335 |
| fft | fp32 | 8 | cuButterfly-cuFFTDx | 16384 | 16384 | 49.775 | 0.002888 |
| fft | fp32 | 8 | cuFFT | 16384 | 16384 | 50.067 | 0.002683 |
| fft | fp32 | 14 | VkFFT | 16 | 16 | 24.568 | 0.008264 |
| fft | fp32 | 14 | cuButterfly-cuFFTDx-direct | 256 | 256 | 34.557 | 0.020521 |
| fft | fp32 | 14 | cuFFT | 256 | 256 | 34.653 | 0.011889 |
| fft | fp32 | 18 | VkFFT | 16 | 16 | 20.856 | 0.018483 |
| fft | fp32 | 18 | cuButterfly-cuFFTDx-online | 16 | 16 | 20.434 | 0.016476 |
| fft | fp32 | 18 | cuFFT | 16 | 16 | 19.845 | 0.022979 |
| fft | fp32 | 20 | VkFFT | 4 | 4 | 20.311 | 0.060918 |
| fft | fp32 | 20 | cuButterfly-cuFFTDx-online | 4 | 4 | 17.799 | 0.071690 |
| fft | fp32 | 20 | cuFFT | 4 | 4 | 19.968 | 0.063590 |
| fwht | fp32 | 8 | cuButterfly-shared-radix4 | 16384 | 16384 | 86.614 | 0.002560 |
| fwht | fp32 | 8 | cuButterfly-warp-register | 16384 | 16384 | 98.085 | 0.002048 |
| fwht | fp32 | 15 | cuButterfly-online-radix4 | 128 | 128 | 13.257 | 0.009032 |
| fwht | fp32 | 15 | cuButterfly-warp-register | 128 | 128 | 64.739 | 0.017603 |
| fwht | fp32 | 20 | cuButterfly-hierarchical-radix4 | 1 | 1 | 10.640 | 0.098550 |
| fwht | fp32 | 20 | cuButterfly-online-radix4 | 4 | 4 | 13.121 | 0.090153 |
| ntt | uint32 | 12 | cuNTT-Hybrid2D-radix2 | 1024 | 1024 | 25.682 | 0.006881 |
| ntt | uint32 | 12 | cuNTT-Hybrid2D-radix4 | 1024 | 1024 | 30.065 | 0.005990 |
| ntt | uint64 | 16 | cuNTT-Hybrid2D-radix2 | 64 | 64 | 13.808 | 0.015862 |
| ntt | uint64 | 16 | cuNTT-Hybrid2D-radix4 | 64 | 64 | 15.064 | 0.015514 |
| ntt | uint64 | 20 | cuNTT-compact-stage | 4 | 4 | 12.278 | 0.094986 |
| xor-zeta | uint32 | 8 | cuButterfly-radix2 | 16384 | 16384 | 84.401 | 0.002847 |
| xor-zeta | uint32 | 8 | cuButterfly-radix4 | 16384 | 16384 | 87.784 | 0.002499 |
| xor-zeta | uint32 | 20 | cuButterfly-hierarchical-radix4 | 1 | 1 | 10.683 | 0.098150 |
| xor-zeta | uint32 | 20 | cuButterfly-online-radix4 | 4 | 4 | 13.132 | 0.095048 |

## Detailed Scaling

| Operator | logN | Batch | Implementation | Median ms | Gpoint/s | Peak fraction | vs basis | Region | Timing quality |
|:--|--:|--:|:--|--:|--:|--:|--:|:--|:--|
| fft | 8 | 1 | VkFFT | 0.002335 | 0.110 | 0.002 | 1.149x | launch-limited | below-timing-floor |
| fft | 8 | 64 | VkFFT | 0.002458 | 6.667 | 0.133 | 1.125x | launch-limited | below-timing-floor |
| fft | 8 | 16,384 | VkFFT | 0.083589 | 50.178 | 1.000 | 1.002x | saturated | stable |
| fft | 8 | 1 | cuButterfly-cuFFTDx | 0.002888 | 0.089 | 0.002 | 0.929x | launch-limited | below-timing-floor |
| fft | 8 | 64 | cuButterfly-cuFFTDx | 0.003123 | 5.246 | 0.105 | 0.885x | launch-limited | below-timing-floor |
| fft | 8 | 16,384 | cuButterfly-cuFFTDx | 0.084265 | 49.775 | 1.000 | 0.994x | saturated | stable |
| fft | 8 | 1 | cuFFT | 0.002683 | 0.095 | 0.002 | 1.000x | launch-limited | below-timing-floor |
| fft | 8 | 64 | cuFFT | 0.002765 | 5.926 | 0.118 | 1.000x | launch-limited | below-timing-floor |
| fft | 8 | 16,384 | cuFFT | 0.083773 | 50.067 | 1.000 | 1.000x | saturated | stable |
| fft | 14 | 1 | VkFFT | 0.008264 | 1.983 | 0.081 | 1.439x | launch-limited | below-timing-floor |
| fft | 14 | 16 | VkFFT | 0.010670 | 24.568 | 1.000 | 1.266x | saturated | below-timing-floor |
| fft | 14 | 256 | VkFFT | 0.171694 | 24.429 | 0.994 | 0.705x | saturated | stable |
| fft | 14 | 1 | cuButterfly-cuFFTDx-direct | 0.020521 | 0.798 | 0.023 | 0.579x | launch-limited | stable |
| fft | 14 | 16 | cuButterfly-cuFFTDx-direct | 0.020777 | 12.617 | 0.365 | 0.650x | launch-limited | stable |
| fft | 14 | 256 | cuButterfly-cuFFTDx-direct | 0.121375 | 34.557 | 1.000 | 0.997x | saturated | stable |
| fft | 14 | 1 | cuFFT | 0.011889 | 1.378 | 0.040 | 1.000x | launch-limited | below-timing-floor |
| fft | 14 | 16 | cuFFT | 0.013507 | 19.409 | 0.560 | 1.000x | ramp | below-timing-floor |
| fft | 14 | 256 | cuFFT | 0.121037 | 34.653 | 1.000 | 1.000x | saturated | stable |
| fft | 18 | 1 | VkFFT | 0.018483 | 14.183 | 0.680 | 1.243x | ramp | below-timing-floor |
| fft | 18 | 4 | VkFFT | 0.063805 | 16.434 | 0.788 | 1.075x | ramp | stable |
| fft | 18 | 16 | VkFFT | 0.201103 | 20.856 | 1.000 | 1.051x | saturated | stable |
| fft | 18 | 1 | cuButterfly-cuFFTDx-online | 0.016476 | 15.911 | 0.779 | 1.395x | ramp | below-timing-floor |
| fft | 18 | 4 | cuButterfly-cuFFTDx-online | 0.059453 | 17.637 | 0.863 | 1.154x | ramp | stable |
| fft | 18 | 16 | cuButterfly-cuFFTDx-online | 0.205261 | 20.434 | 1.000 | 1.030x | saturated | stable |
| fft | 18 | 1 | cuFFT | 0.022979 | 11.408 | 0.575 | 1.000x | ramp | stable |
| fft | 18 | 4 | cuFFT | 0.068608 | 15.284 | 0.770 | 1.000x | ramp | stable |
| fft | 18 | 16 | cuFFT | 0.211354 | 19.845 | 1.000 | 1.000x | saturated | stable |
| fft | 20 | 1 | VkFFT | 0.060918 | 17.213 | 0.847 | 1.044x | ramp | stable |
| fft | 20 | 4 | VkFFT | 0.206500 | 20.311 | 1.000 | 1.017x | saturated | stable |
| fft | 20 | 1 | cuButterfly-cuFFTDx-online | 0.071690 | 14.627 | 0.822 | 0.887x | ramp | unstable |
| fft | 20 | 4 | cuButterfly-cuFFTDx-online | 0.235653 | 17.799 | 1.000 | 0.891x | saturated | stable |
| fft | 20 | 1 | cuFFT | 0.063590 | 16.490 | 0.826 | 1.000x | ramp | stable |
| fft | 20 | 4 | cuFFT | 0.210053 | 19.968 | 1.000 | 1.000x | saturated | stable |
| fwht | 8 | 1 | cuButterfly-shared-radix4 | 0.002560 | 0.100 | 0.001 | 0.800x | launch-limited | below-timing-floor |
| fwht | 8 | 64 | cuButterfly-shared-radix4 | 0.002570 | 6.375 | 0.074 | 0.821x | launch-limited | below-timing-floor |
| fwht | 8 | 16,384 | cuButterfly-shared-radix4 | 0.048425 | 86.614 | 1.000 | 0.883x | saturated | stable |
| fwht | 8 | 1 | cuButterfly-warp-register | 0.002048 | 0.125 | 0.001 | 1.000x | launch-limited | below-timing-floor |
| fwht | 8 | 64 | cuButterfly-warp-register | 0.002109 | 7.769 | 0.079 | 1.000x | launch-limited | below-timing-floor |
| fwht | 8 | 16,384 | cuButterfly-warp-register | 0.042762 | 98.085 | 1.000 | 1.000x | saturated | stable |
| fwht | 15 | 1 | cuButterfly-online-radix4 | 0.009032 | 3.628 | 0.274 | 1.000x | launch-limited | below-timing-floor |
| fwht | 15 | 16 | cuButterfly-online-radix4 | 0.044657 | 11.740 | 0.886 | 0.394x | ramp | stable |
| fwht | 15 | 128 | cuButterfly-online-radix4 | 0.316375 | 13.257 | 1.000 | 0.205x | saturated | stable |
| fwht | 15 | 1 | cuButterfly-warp-register | 0.017603 | 1.862 | 0.029 | 0.513x | launch-limited | below-timing-floor |
| fwht | 15 | 16 | cuButterfly-warp-register | 0.017592 | 29.803 | 0.460 | 1.000x | launch-limited | below-timing-floor |
| fwht | 15 | 128 | cuButterfly-warp-register | 0.064788 | 64.739 | 1.000 | 1.000x | saturated | stable |
| fwht | 20 | 1 | cuButterfly-hierarchical-radix4 | 0.098550 | 10.640 | 1.000 | 0.915x | saturated | stable |
| fwht | 20 | 4 | cuButterfly-hierarchical-radix4 | 0.488223 | 8.591 | 0.807 | 0.655x | ramp | stable |
| fwht | 20 | 1 | cuButterfly-online-radix4 | 0.090153 | 11.631 | 0.886 | 1.000x | ramp | stable |
| fwht | 20 | 4 | cuButterfly-online-radix4 | 0.319652 | 13.121 | 1.000 | 1.000x | saturated | stable |
| ntt | 12 | 1 | cuNTT-Hybrid2D-radix2 | 0.006881 | 0.595 | 0.023 | 1.000x | launch-limited | below-timing-floor |
| ntt | 12 | 16 | cuNTT-Hybrid2D-radix2 | 0.008458 | 7.748 | 0.302 | 1.000x | launch-limited | below-timing-floor |
| ntt | 12 | 1,024 | cuNTT-Hybrid2D-radix2 | 0.163318 | 25.682 | 1.000 | 1.000x | saturated | stable |
| ntt | 12 | 1 | cuNTT-Hybrid2D-radix4 | 0.005990 | 0.684 | 0.023 | 1.149x | launch-limited | below-timing-floor |
| ntt | 12 | 16 | cuNTT-Hybrid2D-radix4 | 0.007639 | 8.579 | 0.285 | 1.107x | launch-limited | below-timing-floor |
| ntt | 12 | 1,024 | cuNTT-Hybrid2D-radix4 | 0.139510 | 30.065 | 1.000 | 1.171x | saturated | stable |
| ntt | 16 | 1 | cuNTT-Hybrid2D-radix2 | 0.015862 | 4.132 | 0.299 | 0.978x | launch-limited | below-timing-floor |
| ntt | 16 | 16 | cuNTT-Hybrid2D-radix2 | 0.087378 | 12.000 | 0.869 | 0.944x | ramp | stable |
| ntt | 16 | 64 | cuNTT-Hybrid2D-radix2 | 0.303759 | 13.808 | 1.000 | 0.917x | saturated | stable |
| ntt | 16 | 1 | cuNTT-Hybrid2D-radix4 | 0.015514 | 4.224 | 0.280 | 1.000x | launch-limited | below-timing-floor |
| ntt | 16 | 16 | cuNTT-Hybrid2D-radix4 | 0.082463 | 12.716 | 0.844 | 1.000x | ramp | stable |
| ntt | 16 | 64 | cuNTT-Hybrid2D-radix4 | 0.278436 | 15.064 | 1.000 | 1.000x | saturated | stable |
| ntt | 20 | 1 | cuNTT-compact-stage | 0.094986 | 11.039 | 0.899 | 1.000x | ramp | stable |
| ntt | 20 | 4 | cuNTT-compact-stage | 0.341606 | 12.278 | 1.000 | 1.000x | saturated | stable |
| xor-zeta | 8 | 1 | cuButterfly-radix2 | 0.002847 | 0.090 | 0.001 | 0.878x | launch-limited | below-timing-floor |
| xor-zeta | 8 | 64 | cuButterfly-radix2 | 0.002929 | 5.594 | 0.066 | 0.874x | launch-limited | below-timing-floor |
| xor-zeta | 8 | 16,384 | cuButterfly-radix2 | 0.049695 | 84.401 | 1.000 | 0.961x | saturated | stable |
| xor-zeta | 8 | 1 | cuButterfly-radix4 | 0.002499 | 0.102 | 0.001 | 1.000x | launch-limited | below-timing-floor |
| xor-zeta | 8 | 64 | cuButterfly-radix4 | 0.002560 | 6.400 | 0.073 | 1.000x | launch-limited | below-timing-floor |
| xor-zeta | 8 | 16,384 | cuButterfly-radix4 | 0.047780 | 87.784 | 1.000 | 1.000x | saturated | stable |
| xor-zeta | 20 | 1 | cuButterfly-hierarchical-radix4 | 0.098150 | 10.683 | 1.000 | 0.968x | saturated | stable |
| xor-zeta | 20 | 4 | cuButterfly-hierarchical-radix4 | 0.487117 | 8.610 | 0.806 | 0.656x | ramp | stable |
| xor-zeta | 20 | 1 | cuButterfly-online-radix4 | 0.095048 | 11.032 | 0.840 | 1.000x | ramp | unstable |
| xor-zeta | 20 | 4 | cuButterfly-online-radix4 | 0.319406 | 13.132 | 1.000 | 1.000x | saturated | stable |
