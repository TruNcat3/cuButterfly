# V100 Three-Way Library Comparison

All local rows use 1000 warmups, 100 repetitions, five randomized process trials, and correctness checks.
GPU-NTT rows are imported from the archived matching-protocol run on the same V100; they were not interleaved with this refresh.
Ratios above one mean the searched cuButterfly/cuNTT configuration has higher throughput.

| Operator | Numeric | logN | Batch | High-performance library | Library ms | Base ms | Searched ms | Search/base | Search/library | Result | Stability |
|:--|:--|--:|--:|:--|--:|--:|--:|--:|--:|:--|:--|
| fft | fp32 | 8 | 16,384 | cuFFT | 0.083794 | 0.112138 | 0.084306 | 1.330x | 0.994x | parity | stable / stable-with-outlier / stable |
| fft | fp32 | 14 | 256 | cuFFT | 0.124426 | 0.722483 | 0.122358 | 5.905x | 1.017x | parity | stable / stable / stable |
| fft | fp32 | 18 | 16 | cuFFT | 0.213156 | 0.510863 | 0.196731 | 2.597x | 1.083x | faster | stable / stable / stable |
| fft | fp32 | 20 | 4 | cuFFT | 0.213975 | 0.768051 | 0.217283 | 3.535x | 0.985x | parity | stable / stable / stable |
| fwht | fp32 | 8 | 65,536 | Dao-AILab-FHT | 0.165396 | 0.187965 | 0.165151 | 1.138x | 1.001x | parity | stable / stable / stable |
| fwht | fp32 | 15 | 256 | Dao-AILab-FHT | 0.117432 | 0.703703 | 0.108636 | 6.478x | 1.081x | faster | stable / stable / stable-with-outlier |
| fwht | fp32 | 15 | 512 | Dao-AILab-FHT | 0.210289 | 1.387151 | 0.195328 | 7.102x | 1.077x | faster | stable / stable / stable |
| ntt | uint64 | 16 | 4 | GPU-NTT-Merge-natural | 0.032881 | 0.031570 | 0.029420 | 1.073x | 1.118x | faster | stable-with-outlier / stable / stable |
| ntt | uint64 | 16 | 64 | GPU-NTT-Merge-natural | 0.386632 | 0.301302 | 0.274883 | 1.096x | 1.407x | faster | stable / stable / stable |
| ntt | uint64 | 16 | 256 | GPU-NTT-Merge-natural | 1.522080 | 1.163162 | 1.054403 | 1.103x | 1.444x | faster | stable / stable / stable |

## Aggregate

| Operator | Shapes | Search/base geomean | Search/library geomean | Faster | Parity | Slower |
|:--|--:|--:|--:|--:|--:|--:|
| fft | 4 | 2.914x | 1.019x | 1 | 3 | 0 |
| fwht | 3 | 3.741x | 1.052x | 2 | 1 | 0 |
| ntt | 3 | 1.091x | 1.314x | 3 | 0 | 0 |

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
