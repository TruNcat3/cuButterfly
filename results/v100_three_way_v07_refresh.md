# V100 Three-Way Library Comparison

All local rows use 1000 warmups, 100 repetitions, five randomized process trials, and correctness checks.
GPU-NTT rows are imported from the archived matching-protocol run on the same V100; they were not interleaved with this refresh.
Ratios above one mean the searched cuButterfly/cuNTT configuration has higher throughput.

| Operator | Numeric | logN | Batch | High-performance library | Library ms | Base ms | Searched ms | Search/base | Search/library | Result | Stability |
|:--|:--|--:|--:|:--|--:|--:|--:|--:|--:|:--|:--|
| fft | fp32 | 8 | 16,384 | cuFFT | 0.083814 | 0.112159 | 0.084244 | 1.331x | 0.995x | parity | stable / stable / stable |
| fft | fp32 | 14 | 256 | cuFFT | 0.124518 | 0.722432 | 0.122470 | 5.899x | 1.017x | parity | stable-with-outlier / stable / stable |
| fft | fp32 | 18 | 16 | cuFFT | 0.213402 | 0.510546 | 0.196721 | 2.595x | 1.085x | faster | stable / stable / stable |
| fft | fp32 | 20 | 4 | cuFFT | 0.214200 | 0.767498 | 0.217283 | 3.532x | 0.986x | parity | stable / stable / stable |
| fwht | fp32 | 8 | 65,536 | Dao-AILab-FHT | 0.165376 | 0.188211 | 0.165151 | 1.140x | 1.001x | parity | stable / stable / stable |
| fwht | fp32 | 15 | 256 | Dao-AILab-FHT | 0.117985 | 0.703836 | 0.133100 | 5.288x | 0.886x | slower | unstable / stable / stable |
| fwht | fp32 | 15 | 512 | Dao-AILab-FHT | 0.210401 | 1.387203 | 0.242688 | 5.716x | 0.867x | slower | stable / stable / stable |
| ntt | uint64 | 16 | 4 | GPU-NTT-Merge-natural | 0.032881 | 0.031498 | 0.029409 | 1.071x | 1.118x | faster | stable-with-outlier / stable / stable |
| ntt | uint64 | 16 | 64 | GPU-NTT-Merge-natural | 0.386632 | 0.299162 | 0.275374 | 1.086x | 1.404x | faster | stable / stable / stable |
| ntt | uint64 | 16 | 256 | GPU-NTT-Merge-natural | 1.522080 | 1.163397 | 1.058652 | 1.099x | 1.438x | faster | stable / stable / stable |

## Aggregate

| Operator | Shapes | Search/base geomean | Search/library geomean | Faster | Parity | Slower |
|:--|--:|--:|--:|--:|--:|--:|
| fft | 4 | 2.913x | 1.020x | 1 | 3 | 0 |
| fwht | 3 | 3.254x | 0.916x | 0 | 1 | 2 |
| ntt | 3 | 1.085x | 1.312x | 3 | 0 | 0 |

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
