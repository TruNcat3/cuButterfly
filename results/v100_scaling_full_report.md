# V100 Orthogonal Length/Batch Scaling

`Batch@90%` is the smallest measured batch reaching 90% of that
implementation's peak measured `Gpoint/s` at fixed transform length.
Rows below the configured 0.020 ms timing floor are retained but are not
used as stable latency claims or peak/saturation references.

## Saturation Summary

| Operator | Numeric | logN | Implementation | Batch@90% | Peak batch | Peak Gpoint/s | Batch-1 ms |
|:--|:--|--:|:--|--:|--:|--:|--:|
| fft | fp32 | 8 | VkFFT | 4096 | 4096 | 51.848 | 0.002314 |
| fft | fp32 | 8 | cuButterfly-cuFFTDx | 4096 | 65536 | 51.206 | 0.002898 |
| fft | fp32 | 8 | cuFFT | 4096 | 65536 | 51.295 | 0.002683 |
| fft | fp32 | 14 | VkFFT | 256 | 1024 | 25.533 | 0.008253 |
| fft | fp32 | 14 | cuButterfly-cuFFTDx-direct | 1024 | 1024 | 44.142 | 0.020552 |
| fft | fp32 | 14 | cuFFT | 1024 | 1024 | 41.289 | 0.011878 |
| fft | fp32 | 18 | VkFFT | 2 | 64 | 23.506 | 0.018534 |
| fft | fp32 | 18 | cuButterfly-cuFFTDx-online | 2 | 16 | 20.450 | 0.016435 |
| fft | fp32 | 18 | cuFFT | 32 | 64 | 23.752 | 0.023050 |
| fft | fp32 | 20 | VkFFT | 8 | 16 | 22.726 | 0.061051 |
| fft | fp32 | 20 | cuButterfly-cuFFTDx-online | 4 | 8 | 18.412 | 0.071404 |
| fft | fp32 | 20 | cuFFT | 8 | 16 | 23.259 | 0.063570 |
| fwht | fp32 | 8 | cuButterfly-shared-radix4 | 16384 | 65536 | 94.607 | 0.002488 |
| fwht | fp32 | 8 | cuButterfly-warp-register | 16384 | 65536 | 102.056 | 0.002058 |
| fwht | fp32 | 15 | cuButterfly-online-radix4 | 32 | 512 | 13.480 | 0.007690 |
| fwht | fp32 | 15 | cuButterfly-warp-register | 256 | 512 | 86.876 | 0.017592 |
| fwht | fp32 | 20 | cuButterfly-hierarchical-radix4 | 1 | 1 | 10.646 | 0.098499 |
| fwht | fp32 | 20 | cuButterfly-online-radix4 | 2 | 16 | 13.609 | 0.090071 |
| ntt | uint32 | 12 | cuNTT-Hybrid2D-radix2 | 1024 | 4096 | 26.331 | 0.006871 |
| ntt | uint32 | 12 | cuNTT-Hybrid2D-radix4 | 1024 | 4096 | 31.167 | 0.005990 |
| ntt | uint64 | 16 | cuNTT-Hybrid2D-radix2 | 32 | 256 | 14.210 | 0.015872 |
| ntt | uint64 | 16 | cuNTT-Hybrid2D-radix4 | 64 | 256 | 15.687 | 0.015534 |
| ntt | uint64 | 20 | cuNTT-compact-stage | 2 | 16 | 12.769 | 0.095058 |
| xor-zeta | uint32 | 8 | cuButterfly-radix2 | 16384 | 65536 | 93.698 | 0.002888 |
| xor-zeta | uint32 | 8 | cuButterfly-radix4 | 16384 | 65536 | 95.802 | 0.002478 |
| xor-zeta | uint32 | 20 | cuButterfly-hierarchical-radix4 | 1 | 1 | 10.677 | 0.098212 |
| xor-zeta | uint32 | 20 | cuButterfly-online-radix4 | 2 | 16 | 13.617 | 0.090112 |

## Detailed Scaling

| Operator | logN | Batch | Implementation | Median ms | Gpoint/s | Peak fraction | vs basis | Region | Timing quality |
|:--|--:|--:|:--|--:|--:|--:|--:|:--|:--|
| fft | 8 | 1 | VkFFT | 0.002314 | 0.111 | 0.002 | 1.159x | launch-limited | below-timing-floor |
| fft | 8 | 4 | VkFFT | 0.002355 | 0.435 | 0.008 | 1.139x | launch-limited | below-timing-floor |
| fft | 8 | 16 | VkFFT | 0.002355 | 1.739 | 0.034 | 1.143x | launch-limited | below-timing-floor |
| fft | 8 | 64 | VkFFT | 0.002458 | 6.667 | 0.129 | 1.113x | launch-limited | below-timing-floor |
| fft | 8 | 256 | VkFFT | 0.002775 | 23.616 | 0.455 | 1.140x | launch-limited | below-timing-floor |
| fft | 8 | 1,024 | VkFFT | 0.005786 | 45.310 | 0.874 | 0.784x | ramp | below-timing-floor |
| fft | 8 | 4,096 | VkFFT | 0.020224 | 51.848 | 1.000 | 1.093x | saturated | stable |
| fft | 8 | 16,384 | VkFFT | 0.083558 | 50.196 | 0.968 | 1.003x | saturated | stable |
| fft | 8 | 65,536 | VkFFT | 0.326113 | 51.446 | 0.992 | 1.003x | saturated | stable |
| fft | 8 | 1 | cuButterfly-cuFFTDx | 0.002898 | 0.088 | 0.002 | 0.926x | launch-limited | below-timing-floor |
| fft | 8 | 4 | cuButterfly-cuFFTDx | 0.003082 | 0.332 | 0.006 | 0.870x | launch-limited | below-timing-floor |
| fft | 8 | 16 | cuButterfly-cuFFTDx | 0.003103 | 1.320 | 0.026 | 0.868x | launch-limited | below-timing-floor |
| fft | 8 | 64 | cuButterfly-cuFFTDx | 0.003103 | 5.280 | 0.103 | 0.881x | launch-limited | below-timing-floor |
| fft | 8 | 256 | cuButterfly-cuFFTDx | 0.003348 | 19.575 | 0.382 | 0.945x | launch-limited | below-timing-floor |
| fft | 8 | 1,024 | cuButterfly-cuFFTDx | 0.005181 | 50.597 | 0.988 | 0.876x | saturated | below-timing-floor |
| fft | 8 | 4,096 | cuButterfly-cuFFTDx | 0.022098 | 47.451 | 0.927 | 1.000x | saturated | stable |
| fft | 8 | 16,384 | cuButterfly-cuFFTDx | 0.084275 | 49.769 | 0.972 | 0.994x | saturated | stable |
| fft | 8 | 65,536 | cuButterfly-cuFFTDx | 0.327639 | 51.206 | 1.000 | 0.998x | saturated | stable |
| fft | 8 | 1 | cuFFT | 0.002683 | 0.095 | 0.002 | 1.000x | launch-limited | below-timing-floor |
| fft | 8 | 4 | cuFFT | 0.002683 | 0.382 | 0.007 | 1.000x | launch-limited | below-timing-floor |
| fft | 8 | 16 | cuFFT | 0.002693 | 1.521 | 0.030 | 1.000x | launch-limited | below-timing-floor |
| fft | 8 | 64 | cuFFT | 0.002734 | 5.993 | 0.117 | 1.000x | launch-limited | below-timing-floor |
| fft | 8 | 256 | cuFFT | 0.003164 | 20.712 | 0.404 | 1.000x | launch-limited | below-timing-floor |
| fft | 8 | 1,024 | cuFFT | 0.004536 | 57.788 | 1.127 | 1.000x | saturated | below-timing-floor |
| fft | 8 | 4,096 | cuFFT | 0.022098 | 47.451 | 0.925 | 1.000x | saturated | stable |
| fft | 8 | 16,384 | cuFFT | 0.083794 | 50.055 | 0.976 | 1.000x | saturated | stable |
| fft | 8 | 65,536 | cuFFT | 0.327076 | 51.295 | 1.000 | 1.000x | saturated | stable |
| fft | 14 | 1 | VkFFT | 0.008253 | 1.985 | 0.078 | 1.439x | launch-limited | below-timing-floor |
| fft | 14 | 4 | VkFFT | 0.008489 | 7.720 | 0.302 | 1.407x | launch-limited | below-timing-floor |
| fft | 14 | 16 | VkFFT | 0.010680 | 24.545 | 0.961 | 1.264x | saturated | below-timing-floor |
| fft | 14 | 64 | VkFFT | 0.047227 | 22.203 | 0.870 | 0.585x | ramp | stable |
| fft | 14 | 256 | VkFFT | 0.171725 | 24.425 | 0.957 | 0.705x | saturated | stable |
| fft | 14 | 1,024 | VkFFT | 0.657070 | 25.533 | 1.000 | 0.618x | saturated | stable |
| fft | 14 | 1 | cuButterfly-cuFFTDx-direct | 0.020552 | 0.797 | 0.018 | 0.578x | launch-limited | stable |
| fft | 14 | 4 | cuButterfly-cuFFTDx-direct | 0.020603 | 3.181 | 0.072 | 0.580x | launch-limited | stable |
| fft | 14 | 16 | cuButterfly-cuFFTDx-direct | 0.020777 | 12.617 | 0.286 | 0.650x | launch-limited | stable-with-outlier |
| fft | 14 | 64 | cuButterfly-cuFFTDx-direct | 0.031693 | 33.085 | 0.750 | 0.872x | ramp | stable-with-outlier |
| fft | 14 | 256 | cuButterfly-cuFFTDx-direct | 0.121395 | 34.551 | 0.783 | 0.998x | ramp | stable |
| fft | 14 | 1,024 | cuButterfly-cuFFTDx-direct | 0.380078 | 44.142 | 1.000 | 1.069x | saturated | stable |
| fft | 14 | 1 | cuFFT | 0.011878 | 1.379 | 0.033 | 1.000x | launch-limited | below-timing-floor |
| fft | 14 | 4 | cuFFT | 0.011940 | 5.489 | 0.133 | 1.000x | launch-limited | below-timing-floor |
| fft | 14 | 16 | cuFFT | 0.013496 | 19.423 | 0.470 | 1.000x | launch-limited | below-timing-floor |
| fft | 14 | 64 | cuFFT | 0.027648 | 37.926 | 0.919 | 1.000x | saturated | unstable |
| fft | 14 | 256 | cuFFT | 0.121098 | 34.636 | 0.839 | 1.000x | ramp | stable |
| fft | 14 | 1,024 | cuFFT | 0.406333 | 41.289 | 1.000 | 1.000x | saturated | stable |
| fft | 18 | 1 | VkFFT | 0.018534 | 14.144 | 0.602 | 1.244x | ramp | below-timing-floor |
| fft | 18 | 2 | VkFFT | 0.023583 | 22.232 | 0.946 | 1.191x | saturated | stable |
| fft | 18 | 4 | VkFFT | 0.063816 | 16.431 | 0.699 | 1.075x | ramp | stable |
| fft | 18 | 8 | VkFFT | 0.111780 | 18.761 | 0.798 | 1.099x | ramp | stable |
| fft | 18 | 16 | VkFFT | 0.201257 | 20.841 | 0.887 | 1.049x | ramp | stable |
| fft | 18 | 32 | VkFFT | 0.372163 | 22.540 | 0.959 | 1.018x | saturated | stable |
| fft | 18 | 64 | VkFFT | 0.713728 | 23.506 | 1.000 | 0.990x | saturated | stable |
| fft | 18 | 1 | cuButterfly-cuFFTDx-online | 0.016435 | 15.950 | 0.780 | 1.403x | ramp | below-timing-floor |
| fft | 18 | 2 | cuButterfly-cuFFTDx-online | 0.025969 | 20.189 | 0.987 | 1.082x | saturated | stable |
| fft | 18 | 4 | cuButterfly-cuFFTDx-online | 0.059628 | 17.585 | 0.860 | 1.150x | ramp | stable |
| fft | 18 | 8 | cuButterfly-cuFFTDx-online | 0.108851 | 19.266 | 0.942 | 1.129x | saturated | stable |
| fft | 18 | 16 | cuButterfly-cuFFTDx-online | 0.205097 | 20.450 | 1.000 | 1.030x | saturated | stable |
| fft | 18 | 32 | cuButterfly-cuFFTDx-online | 0.411238 | 20.398 | 0.997 | 0.921x | saturated | stable |
| fft | 18 | 64 | cuButterfly-cuFFTDx-online | 0.840888 | 19.952 | 0.976 | 0.840x | saturated | stable |
| fft | 18 | 1 | cuFFT | 0.023050 | 11.373 | 0.479 | 1.000x | launch-limited | stable |
| fft | 18 | 2 | cuFFT | 0.028088 | 18.666 | 0.786 | 1.000x | ramp | stable |
| fft | 18 | 4 | cuFFT | 0.068598 | 15.286 | 0.644 | 1.000x | ramp | stable |
| fft | 18 | 8 | cuFFT | 0.122870 | 17.068 | 0.719 | 1.000x | ramp | stable |
| fft | 18 | 16 | cuFFT | 0.211149 | 19.864 | 0.836 | 1.000x | ramp | stable |
| fft | 18 | 32 | cuFFT | 0.378737 | 22.149 | 0.933 | 1.000x | saturated | stable |
| fft | 18 | 64 | cuFFT | 0.706355 | 23.752 | 1.000 | 1.000x | saturated | stable |
| fft | 20 | 1 | VkFFT | 0.061051 | 17.175 | 0.756 | 1.041x | ramp | stable-with-outlier |
| fft | 20 | 2 | VkFFT | 0.117484 | 17.851 | 0.785 | 1.049x | ramp | stable-with-outlier |
| fft | 20 | 4 | VkFFT | 0.206705 | 20.291 | 0.893 | 1.017x | ramp | stable |
| fft | 20 | 8 | VkFFT | 0.384266 | 21.830 | 0.961 | 0.993x | saturated | stable |
| fft | 20 | 16 | VkFFT | 0.738243 | 22.726 | 1.000 | 0.977x | saturated | stable |
| fft | 20 | 1 | cuButterfly-cuFFTDx-online | 0.071404 | 14.685 | 0.798 | 0.890x | ramp | stable-with-outlier |
| fft | 20 | 2 | cuButterfly-cuFFTDx-online | 0.128553 | 16.314 | 0.886 | 0.959x | ramp | stable |
| fft | 20 | 4 | cuButterfly-cuFFTDx-online | 0.235581 | 17.804 | 0.967 | 0.892x | saturated | stable |
| fft | 20 | 8 | cuButterfly-cuFFTDx-online | 0.455598 | 18.412 | 1.000 | 0.837x | saturated | stable |
| fft | 20 | 16 | cuButterfly-cuFFTDx-online | 0.927017 | 18.098 | 0.983 | 0.778x | saturated | stable |
| fft | 20 | 1 | cuFFT | 0.063570 | 16.495 | 0.709 | 1.000x | ramp | stable |
| fft | 20 | 2 | cuFFT | 0.123228 | 17.018 | 0.732 | 1.000x | ramp | stable |
| fft | 20 | 4 | cuFFT | 0.210237 | 19.950 | 0.858 | 1.000x | ramp | stable |
| fft | 20 | 8 | cuFFT | 0.381512 | 21.988 | 0.945 | 1.000x | saturated | stable |
| fft | 20 | 16 | cuFFT | 0.721326 | 23.259 | 1.000 | 1.000x | saturated | stable |
| fwht | 8 | 1 | cuButterfly-shared-radix4 | 0.002488 | 0.103 | 0.001 | 0.827x | launch-limited | below-timing-floor |
| fwht | 8 | 4 | cuButterfly-shared-radix4 | 0.002488 | 0.412 | 0.004 | 0.856x | launch-limited | below-timing-floor |
| fwht | 8 | 16 | cuButterfly-shared-radix4 | 0.002540 | 1.613 | 0.017 | 0.851x | launch-limited | below-timing-floor |
| fwht | 8 | 64 | cuButterfly-shared-radix4 | 0.002550 | 6.425 | 0.068 | 0.827x | launch-limited | below-timing-floor |
| fwht | 8 | 256 | cuButterfly-shared-radix4 | 0.002949 | 22.223 | 0.235 | 0.820x | launch-limited | below-timing-floor |
| fwht | 8 | 1,024 | cuButterfly-shared-radix4 | 0.004352 | 60.235 | 0.637 | 0.814x | ramp | below-timing-floor |
| fwht | 8 | 4,096 | cuButterfly-shared-radix4 | 0.014971 | 70.040 | 0.740 | 0.750x | ramp | below-timing-floor |
| fwht | 8 | 16,384 | cuButterfly-shared-radix4 | 0.048486 | 86.505 | 0.914 | 0.882x | saturated | stable |
| fwht | 8 | 65,536 | cuButterfly-shared-radix4 | 0.177336 | 94.607 | 1.000 | 0.927x | saturated | stable |
| fwht | 8 | 1 | cuButterfly-warp-register | 0.002058 | 0.124 | 0.001 | 1.000x | launch-limited | below-timing-floor |
| fwht | 8 | 4 | cuButterfly-warp-register | 0.002130 | 0.481 | 0.005 | 1.000x | launch-limited | below-timing-floor |
| fwht | 8 | 16 | cuButterfly-warp-register | 0.002161 | 1.895 | 0.019 | 1.000x | launch-limited | below-timing-floor |
| fwht | 8 | 64 | cuButterfly-warp-register | 0.002109 | 7.769 | 0.076 | 1.000x | launch-limited | below-timing-floor |
| fwht | 8 | 256 | cuButterfly-warp-register | 0.002417 | 27.115 | 0.266 | 1.000x | launch-limited | below-timing-floor |
| fwht | 8 | 1,024 | cuButterfly-warp-register | 0.003543 | 73.989 | 0.725 | 1.000x | ramp | below-timing-floor |
| fwht | 8 | 4,096 | cuButterfly-warp-register | 0.011233 | 93.348 | 0.915 | 1.000x | saturated | below-timing-floor |
| fwht | 8 | 16,384 | cuButterfly-warp-register | 0.042762 | 98.085 | 0.961 | 1.000x | saturated | stable |
| fwht | 8 | 65,536 | cuButterfly-warp-register | 0.164393 | 102.056 | 1.000 | 1.000x | saturated | stable |
| fwht | 15 | 1 | cuButterfly-online-radix4 | 0.007690 | 4.261 | 0.316 | 1.000x | launch-limited | below-timing-floor |
| fwht | 15 | 4 | cuButterfly-online-radix4 | 0.015309 | 8.562 | 0.635 | 1.000x | ramp | below-timing-floor |
| fwht | 15 | 16 | cuButterfly-online-radix4 | 0.044698 | 11.730 | 0.870 | 0.394x | ramp | stable-with-outlier |
| fwht | 15 | 32 | cuButterfly-online-radix4 | 0.084255 | 12.445 | 0.923 | 0.266x | saturated | stable |
| fwht | 15 | 128 | cuButterfly-online-radix4 | 0.317020 | 13.230 | 0.981 | 0.205x | saturated | stable |
| fwht | 15 | 256 | cuButterfly-online-radix4 | 0.625777 | 13.405 | 0.994 | 0.171x | saturated | stable |
| fwht | 15 | 512 | cuButterfly-online-radix4 | 1.244590 | 13.480 | 1.000 | 0.155x | saturated | stable |
| fwht | 15 | 1 | cuButterfly-warp-register | 0.017592 | 1.863 | 0.021 | 0.437x | launch-limited | below-timing-floor |
| fwht | 15 | 4 | cuButterfly-warp-register | 0.017654 | 7.424 | 0.085 | 0.867x | launch-limited | below-timing-floor |
| fwht | 15 | 16 | cuButterfly-warp-register | 0.017613 | 29.767 | 0.343 | 1.000x | launch-limited | below-timing-floor |
| fwht | 15 | 32 | cuButterfly-warp-register | 0.022385 | 46.843 | 0.539 | 1.000x | ramp | stable |
| fwht | 15 | 128 | cuButterfly-warp-register | 0.064840 | 64.687 | 0.745 | 1.000x | ramp | stable |
| fwht | 15 | 256 | cuButterfly-warp-register | 0.107244 | 78.220 | 0.900 | 1.000x | saturated | stable-with-outlier |
| fwht | 15 | 512 | cuButterfly-warp-register | 0.193116 | 86.876 | 1.000 | 1.000x | saturated | stable |
| fwht | 20 | 1 | cuButterfly-hierarchical-radix4 | 0.098499 | 10.646 | 1.000 | 0.914x | saturated | stable |
| fwht | 20 | 2 | cuButterfly-hierarchical-radix4 | 0.257331 | 8.150 | 0.766 | 0.650x | ramp | stable |
| fwht | 20 | 4 | cuButterfly-hierarchical-radix4 | 0.488110 | 8.593 | 0.807 | 0.655x | ramp | stable |
| fwht | 20 | 8 | cuButterfly-hierarchical-radix4 | 0.947814 | 8.850 | 0.831 | 0.659x | ramp | stable |
| fwht | 20 | 16 | cuButterfly-hierarchical-radix4 | 1.868196 | 8.980 | 0.844 | 0.660x | ramp | stable |
| fwht | 20 | 1 | cuButterfly-online-radix4 | 0.090071 | 11.642 | 0.855 | 1.000x | ramp | stable |
| fwht | 20 | 2 | cuButterfly-online-radix4 | 0.167301 | 12.535 | 0.921 | 1.000x | saturated | stable |
| fwht | 20 | 4 | cuButterfly-online-radix4 | 0.319857 | 13.113 | 0.964 | 1.000x | saturated | stable |
| fwht | 20 | 8 | cuButterfly-online-radix4 | 0.624742 | 13.427 | 0.987 | 1.000x | saturated | stable |
| fwht | 20 | 16 | cuButterfly-online-radix4 | 1.232824 | 13.609 | 1.000 | 1.000x | saturated | stable |
| ntt | 12 | 1 | cuNTT-Hybrid2D-radix2 | 0.006871 | 0.596 | 0.023 | 1.000x | launch-limited | below-timing-floor |
| ntt | 12 | 4 | cuNTT-Hybrid2D-radix2 | 0.007158 | 2.289 | 0.087 | 1.000x | launch-limited | below-timing-floor |
| ntt | 12 | 16 | cuNTT-Hybrid2D-radix2 | 0.008468 | 7.739 | 0.294 | 1.000x | launch-limited | below-timing-floor |
| ntt | 12 | 64 | cuNTT-Hybrid2D-radix2 | 0.015483 | 16.931 | 0.643 | 1.000x | ramp | below-timing-floor |
| ntt | 12 | 256 | cuNTT-Hybrid2D-radix2 | 0.048230 | 21.741 | 0.826 | 1.000x | ramp | stable-with-outlier |
| ntt | 12 | 1,024 | cuNTT-Hybrid2D-radix2 | 0.164413 | 25.511 | 0.969 | 1.000x | saturated | stable-with-outlier |
| ntt | 12 | 4,096 | cuNTT-Hybrid2D-radix2 | 0.637164 | 26.331 | 1.000 | 1.000x | saturated | stable |
| ntt | 12 | 1 | cuNTT-Hybrid2D-radix4 | 0.005990 | 0.684 | 0.022 | 1.147x | launch-limited | below-timing-floor |
| ntt | 12 | 4 | cuNTT-Hybrid2D-radix4 | 0.006369 | 2.572 | 0.083 | 1.124x | launch-limited | below-timing-floor |
| ntt | 12 | 16 | cuNTT-Hybrid2D-radix4 | 0.007629 | 8.590 | 0.276 | 1.110x | launch-limited | below-timing-floor |
| ntt | 12 | 64 | cuNTT-Hybrid2D-radix4 | 0.013855 | 18.921 | 0.607 | 1.118x | ramp | below-timing-floor |
| ntt | 12 | 256 | cuNTT-Hybrid2D-radix4 | 0.042824 | 24.486 | 0.786 | 1.126x | ramp | stable |
| ntt | 12 | 1,024 | cuNTT-Hybrid2D-radix4 | 0.139889 | 29.983 | 0.962 | 1.175x | saturated | stable |
| ntt | 12 | 4,096 | cuNTT-Hybrid2D-radix4 | 0.538307 | 31.167 | 1.000 | 1.184x | saturated | stable |
| ntt | 16 | 1 | cuNTT-Hybrid2D-radix2 | 0.015872 | 4.129 | 0.291 | 0.979x | launch-limited | below-timing-floor |
| ntt | 16 | 4 | cuNTT-Hybrid2D-radix2 | 0.031549 | 8.309 | 0.585 | 0.933x | ramp | stable |
| ntt | 16 | 16 | cuNTT-Hybrid2D-radix2 | 0.087163 | 12.030 | 0.847 | 0.945x | ramp | stable-with-outlier |
| ntt | 16 | 32 | cuNTT-Hybrid2D-radix2 | 0.160543 | 13.063 | 0.919 | 0.928x | saturated | stable |
| ntt | 16 | 64 | cuNTT-Hybrid2D-radix2 | 0.305172 | 13.744 | 0.967 | 0.916x | saturated | stable |
| ntt | 16 | 128 | cuNTT-Hybrid2D-radix2 | 0.591452 | 14.183 | 0.998 | 0.914x | saturated | stable |
| ntt | 16 | 256 | cuNTT-Hybrid2D-radix2 | 1.180631 | 14.210 | 1.000 | 0.906x | saturated | stable |
| ntt | 16 | 1 | cuNTT-Hybrid2D-radix4 | 0.015534 | 4.219 | 0.269 | 1.000x | launch-limited | below-timing-floor |
| ntt | 16 | 4 | cuNTT-Hybrid2D-radix4 | 0.029430 | 8.907 | 0.568 | 1.000x | ramp | stable-with-outlier |
| ntt | 16 | 16 | cuNTT-Hybrid2D-radix4 | 0.082381 | 12.728 | 0.811 | 1.000x | ramp | stable-with-outlier |
| ntt | 16 | 32 | cuNTT-Hybrid2D-radix4 | 0.149043 | 14.071 | 0.897 | 1.000x | ramp | stable |
| ntt | 16 | 64 | cuNTT-Hybrid2D-radix4 | 0.279552 | 15.004 | 0.956 | 1.000x | saturated | stable |
| ntt | 16 | 128 | cuNTT-Hybrid2D-radix4 | 0.540559 | 15.518 | 0.989 | 1.000x | saturated | stable |
| ntt | 16 | 256 | cuNTT-Hybrid2D-radix4 | 1.069466 | 15.687 | 1.000 | 1.000x | saturated | stable |
| ntt | 20 | 1 | cuNTT-compact-stage | 0.095058 | 11.031 | 0.864 | 1.000x | ramp | stable |
| ntt | 20 | 2 | cuNTT-compact-stage | 0.179866 | 11.660 | 0.913 | 1.000x | saturated | stable |
| ntt | 20 | 4 | cuNTT-compact-stage | 0.342262 | 12.255 | 0.960 | 1.000x | saturated | stable |
| ntt | 20 | 8 | cuNTT-compact-stage | 0.665016 | 12.614 | 0.988 | 1.000x | saturated | stable |
| ntt | 20 | 16 | cuNTT-compact-stage | 1.313854 | 12.769 | 1.000 | 1.000x | saturated | stable |
| xor-zeta | 8 | 1 | cuButterfly-radix2 | 0.002888 | 0.089 | 0.001 | 0.858x | launch-limited | below-timing-floor |
| xor-zeta | 8 | 4 | cuButterfly-radix2 | 0.002867 | 0.357 | 0.004 | 0.872x | launch-limited | below-timing-floor |
| xor-zeta | 8 | 16 | cuButterfly-radix2 | 0.002898 | 1.413 | 0.015 | 0.873x | launch-limited | below-timing-floor |
| xor-zeta | 8 | 64 | cuButterfly-radix2 | 0.002918 | 5.615 | 0.060 | 0.877x | launch-limited | below-timing-floor |
| xor-zeta | 8 | 256 | cuButterfly-radix2 | 0.003308 | 19.811 | 0.211 | 0.888x | launch-limited | below-timing-floor |
| xor-zeta | 8 | 1,024 | cuButterfly-radix2 | 0.004823 | 54.353 | 0.580 | 0.900x | ramp | below-timing-floor |
| xor-zeta | 8 | 4,096 | cuButterfly-radix2 | 0.016077 | 65.222 | 0.696 | 0.921x | ramp | below-timing-floor |
| xor-zeta | 8 | 16,384 | cuButterfly-radix2 | 0.049715 | 84.367 | 0.900 | 0.961x | saturated | stable |
| xor-zeta | 8 | 65,536 | cuButterfly-radix2 | 0.179057 | 93.698 | 1.000 | 0.978x | saturated | stable |
| xor-zeta | 8 | 1 | cuButterfly-radix4 | 0.002478 | 0.103 | 0.001 | 1.000x | launch-limited | below-timing-floor |
| xor-zeta | 8 | 4 | cuButterfly-radix4 | 0.002499 | 0.410 | 0.004 | 1.000x | launch-limited | below-timing-floor |
| xor-zeta | 8 | 16 | cuButterfly-radix4 | 0.002529 | 1.620 | 0.017 | 1.000x | launch-limited | below-timing-floor |
| xor-zeta | 8 | 64 | cuButterfly-radix4 | 0.002560 | 6.400 | 0.067 | 1.000x | launch-limited | below-timing-floor |
| xor-zeta | 8 | 256 | cuButterfly-radix4 | 0.002939 | 22.299 | 0.233 | 1.000x | launch-limited | below-timing-floor |
| xor-zeta | 8 | 1,024 | cuButterfly-radix4 | 0.004342 | 60.374 | 0.630 | 1.000x | ramp | below-timing-floor |
| xor-zeta | 8 | 4,096 | cuButterfly-radix4 | 0.014807 | 70.816 | 0.739 | 1.000x | ramp | below-timing-floor |
| xor-zeta | 8 | 16,384 | cuButterfly-radix4 | 0.047790 | 87.765 | 0.916 | 1.000x | saturated | stable |
| xor-zeta | 8 | 65,536 | cuButterfly-radix4 | 0.175124 | 95.802 | 1.000 | 1.000x | saturated | stable |
| xor-zeta | 20 | 1 | cuButterfly-hierarchical-radix4 | 0.098212 | 10.677 | 1.000 | 0.918x | saturated | stable |
| xor-zeta | 20 | 2 | cuButterfly-hierarchical-radix4 | 0.256819 | 8.166 | 0.765 | 0.651x | ramp | stable |
| xor-zeta | 20 | 4 | cuButterfly-hierarchical-radix4 | 0.487014 | 8.612 | 0.807 | 0.656x | ramp | stable |
| xor-zeta | 20 | 8 | cuButterfly-hierarchical-radix4 | 0.945746 | 8.870 | 0.831 | 0.660x | ramp | stable |
| xor-zeta | 20 | 16 | cuButterfly-hierarchical-radix4 | 1.863813 | 9.002 | 0.843 | 0.661x | ramp | stable |
| xor-zeta | 20 | 1 | cuButterfly-online-radix4 | 0.090112 | 11.636 | 0.855 | 1.000x | ramp | stable-with-outlier |
| xor-zeta | 20 | 2 | cuButterfly-online-radix4 | 0.167076 | 12.552 | 0.922 | 1.000x | saturated | stable |
| xor-zeta | 20 | 4 | cuButterfly-online-radix4 | 0.319273 | 13.137 | 0.965 | 1.000x | saturated | stable |
| xor-zeta | 20 | 8 | cuButterfly-online-radix4 | 0.623821 | 13.447 | 0.988 | 1.000x | saturated | stable |
| xor-zeta | 20 | 16 | cuButterfly-online-radix4 | 1.232067 | 13.617 | 1.000 | 1.000x | saturated | stable |
