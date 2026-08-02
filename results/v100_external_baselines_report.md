# V100 Matching-Protocol External Baselines

All rows use 1000 warmups, 100 repetitions, five independent process trials, and correctness checks.

Rows below 0.020 ms are retained but are not used for stable latency claims.

| Operator | logN | Batch | Output | Implementation | Median ms | vs cuButterfly/cuNTT | Timing quality |
|:--|--:|--:|:--|:--|--:|--:|:--|
| fwht | 15 | 16 | natural | Dao-AILab-FHT | 0.015483 | 1.148x | below-timing-floor |
| fwht | 15 | 16 | natural | cuButterfly-warp-register | 0.017777 | 1.000x | below-timing-floor |
| fwht | 15 | 256 | natural | Dao-AILab-FHT | 0.116429 | 0.924x | stable |
| fwht | 15 | 256 | natural | cuButterfly-warp-register | 0.107551 | 1.000x | stable |
| fwht | 15 | 4 | natural | Dao-AILab-FHT | 0.015606 | 1.142x | below-timing-floor |
| fwht | 15 | 4 | natural | cuButterfly-warp-register | 0.017818 | 1.000x | below-timing-floor |
| fwht | 15 | 512 | natural | Dao-AILab-FHT | 0.207882 | 0.930x | stable |
| fwht | 15 | 512 | natural | cuButterfly-warp-register | 0.193270 | 1.000x | stable-with-outlier |
| fwht | 8 | 4,096 | natural | Dao-AILab-FHT | 0.011561 | 0.978x | below-timing-floor |
| fwht | 8 | 4,096 | natural | cuButterfly-warp-register | 0.011305 | 1.000x | below-timing-floor |
| fwht | 8 | 65,536 | natural | Dao-AILab-FHT | 0.164506 | 0.999x | stable |
| fwht | 8 | 65,536 | natural | cuButterfly-warp-register | 0.164280 | 1.000x | stable |
| ntt | 16 | 256 | natural | GPU-NTT-Merge-natural | 1.522080 | 0.702x | stable |
| ntt | 16 | 256 | natural | cuNTT-Hybrid2D-radix4 | 1.068636 | 1.000x | stable |
| ntt | 16 | 4 | natural | GPU-NTT-Merge-natural | 0.032881 | 0.902x | stable-with-outlier |
| ntt | 16 | 4 | natural | cuNTT-Hybrid2D-radix4 | 0.029645 | 1.000x | stable-with-outlier |
| ntt | 16 | 64 | natural | GPU-NTT-Merge-natural | 0.386632 | 0.718x | stable |
| ntt | 16 | 64 | natural | cuNTT-Hybrid2D-radix4 | 0.277678 | 1.000x | stable |
| ntt | 20 | 16 | bit-reversed | GPU-NTT-Merge-native | 1.568900 | 0.839x | stable |
| ntt | 20 | 16 | bit-reversed | cuNTT-compact-stage | 1.315605 | 1.000x | stable-with-outlier |
| ntt | 20 | 2 | bit-reversed | GPU-NTT-Merge-native | 0.204872 | 0.878x | stable |
| ntt | 20 | 2 | bit-reversed | cuNTT-compact-stage | 0.179855 | 1.000x | stable |
| ntt | 20 | 4 | bit-reversed | GPU-NTT-Merge-native | 0.399534 | 0.856x | stable |
| ntt | 20 | 4 | bit-reversed | cuNTT-compact-stage | 0.341914 | 1.000x | stable |
