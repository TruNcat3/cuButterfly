# V100 Three-Way Library Comparison

All local rows use 1000 warmups, 100 repetitions, five randomized process trials, and correctness checks.
GPU-NTT rows are imported from the archived matching-protocol run on the same V100; they were not interleaved with this refresh.
Ratios above one mean the searched cuButterfly/cuNTT configuration has higher throughput.

| Operator | Numeric | logN | Batch | High-performance library | Library ms | Base ms | Searched ms | Search/base | Search/library | Result | Stability |
|:--|:--|--:|--:|:--|--:|--:|--:|--:|--:|:--|:--|
| fft | fp32 | 8 | 16,384 | cuFFT | 0.083814 | 0.112087 | 0.084316 | 1.329x | 0.994x | parity | stable / stable-with-outlier / stable |
| fft | fp32 | 14 | 256 | cuFFT | 0.121754 | 0.719780 | 0.120996 | 5.949x | 1.006x | parity | stable / stable / stable |
| fft | fp32 | 18 | 16 | cuFFT | 0.211630 | 0.510269 | 0.195635 | 2.608x | 1.082x | faster | stable / stable / stable |
| fft | fp32 | 20 | 4 | cuFFT | 0.210309 | 0.767959 | 0.214589 | 3.579x | 0.980x | parity | stable / stable / stable |
| fwht | fp32 | 8 | 65,536 | Dao-AILab-FHT | 0.164700 | 0.188140 | 0.164321 | 1.145x | 1.002x | parity | stable / stable / stable |
| fwht | fp32 | 15 | 256 | Dao-AILab-FHT | 0.116132 | 0.699310 | 0.106998 | 6.536x | 1.085x | faster | stable / stable / stable-with-outlier |
| fwht | fp32 | 15 | 512 | Dao-AILab-FHT | 0.207749 | 1.380014 | 0.193034 | 7.149x | 1.076x | faster | stable / stable / stable |
| ntt | uint64 | 16 | 4 | GPU-NTT-Merge-natural | 0.032881 | 0.031529 | 0.029420 | 1.072x | 1.118x | faster | stable-with-outlier / stable / stable |
| ntt | uint64 | 16 | 64 | GPU-NTT-Merge-natural | 0.386632 | 0.301097 | 0.274237 | 1.098x | 1.410x | faster | stable / stable / stable |
| ntt | uint64 | 16 | 256 | GPU-NTT-Merge-natural | 1.522080 | 1.163039 | 1.058611 | 1.099x | 1.438x | faster | stable / stable / stable |

## Aggregate

| Operator | Shapes | Search/base geomean | Search/library geomean | Faster | Parity | Slower |
|:--|--:|--:|--:|--:|--:|--:|
| fft | 4 | 2.931x | 1.015x | 1 | 3 | 0 |
| fwht | 3 | 3.768x | 1.054x | 2 | 1 | 0 |
| ntt | 3 | 1.089x | 1.313x | 3 | 0 | 0 |

## Extended Internal Coverage

These full-protocol adaptive shapes quantify search gain where no exact external-library row is joined.
They are not external superiority claims.

| Operator | Numeric | Accumulation | Shapes | Search/base geomean | Median | Maximum |
|:--|:--|:--|--:|--:|--:|--:|
| fft | bf16 | fp32 | 4 | 1.000x | 1.000x | 1.000x |
| fft | bf16 | native | 9 | 1.025x | 1.000x | 1.123x |
| fft | fp16 | fp32 | 7 | 1.085x | 1.116x | 1.136x |
| fft | fp16 | native | 8 | 1.124x | 1.133x | 1.180x |
| fft | fp32 | native | 8 | 1.097x | 1.091x | 1.150x |
| fft | fp64 | native | 5 | 1.110x | 1.131x | 1.146x |
| fwht | bf16 | fp32 | 6 | 1.083x | 1.059x | 1.285x |
| fwht | bf16 | native | 7 | 1.052x | 1.044x | 1.121x |
| fwht | fp16 | fp32 | 4 | 1.148x | 1.138x | 1.208x |
| fwht | fp16 | native | 5 | 1.080x | 1.090x | 1.113x |
| fwht | fp32 | native | 7 | 1.089x | 1.089x | 1.143x |
| fwht | fp64 | native | 4 | 1.106x | 1.108x | 1.194x |
| ntt | uint64 | native | 2 | 1.088x | 1.088x | 1.089x |
| structured-2x2 | bf16 | fp32 | 5 | 1.135x | 1.123x | 1.217x |
| structured-2x2 | bf16 | native | 2 | 1.000x | 1.000x | 1.000x |
| structured-2x2 | fp16 | fp32 | 2 | 1.137x | 1.140x | 1.229x |
| structured-2x2 | fp16 | native | 3 | 1.161x | 1.139x | 1.212x |
| structured-2x2 | fp32 | native | 2 | 1.037x | 1.037x | 1.047x |
| structured-2x2 | fp64 | native | 7 | 1.097x | 1.057x | 1.292x |
| subset-zeta | uint32 | native | 2 | 1.139x | 1.139x | 1.168x |
