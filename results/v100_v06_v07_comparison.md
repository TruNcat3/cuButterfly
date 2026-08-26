# V100 v0.6/v0.7 NTT Comparison

All paths implement the same forward, natural-order NTT contract. Each point uses 30 warmups,
500 timed repetitions, three randomized process trials, CUDA-event kernel time, and correctness checks.
Ratios above one favor the searched configuration named in the numerator.

`v0.7-base` is the first generated HybridDataflow point (`Ur=1,Td=4`); `v0.7-search` uses
the generated V100 static table. HybridDataflow is currently an on-chip whole-transform backend
for `logN=10/12`, so this table does not directly join the archived GPU-NTT `logN=16/20` rows.
At `logN=10`, the v0.6 pair is generic baseline/Tile256; at `logN=12`, it is Hybrid2D radix-2/radix-4.

| Numeric | logN | Batch | v0.6 ms | v0.6 search ms | v0.7 ms | v0.7 search ms | v0.6 search/base | v0.7 search/base | v0.7 search / v0.6 search throughput | Fastest | Stability |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|:--|:--|
| uint64 | 10 | 1 | 0.026718 | 0.011024 | 0.041708 | 0.030841 | 2.424x | 1.352x | 0.357x | v06_search | unstable/stable/unstable/unstable |
| uint64 | 10 | 16 | 0.026931 | 0.012374 | 0.043454 | 0.036325 | 2.176x | 1.196x | 0.341x | v06_search | stable/unstable/unstable/stable |
| uint64 | 10 | 160 | 0.044626 | 0.023112 | 0.045902 | 0.039707 | 1.931x | 1.156x | 0.582x | v06_search | unstable/unstable/unstable/unstable |
| uint64 | 10 | 256 | 0.056480 | 0.030566 | 0.062341 | 0.041882 | 1.848x | 1.488x | 0.730x | v06_search | unstable/unstable/unstable/unstable |
| uint64 | 10 | 512 | 0.091507 | 0.052978 | 0.119906 | 0.097669 | 1.727x | 1.228x | 0.542x | v06_search | unstable/unstable/unstable/unstable |
| uint64 | 10 | 80 | 0.041165 | 0.017228 | 0.042052 | 0.031224 | 2.389x | 1.347x | 0.552x | v06_search | unstable/unstable/unstable/unstable |
| uint32 | 12 | 1 | 0.006869 | 0.005978 | 0.065548 | 0.068137 | 1.149x | 0.962x | 0.088x | v06_search | unstable/stable/stable/stable |
| uint32 | 12 | 16 | 0.008491 | 0.007643 | 0.065925 | 0.068178 | 1.111x | 0.967x | 0.112x | v06_search | stable/stable/stable/stable |
| uint32 | 12 | 160 | 0.030839 | 0.031238 | 0.077525 | 0.076937 | 0.987x | 1.008x | 0.406x | v06 | stable/unstable/stable/stable |
| uint32 | 12 | 256 | 0.054782 | 0.043022 | 0.167471 | 0.124766 | 1.273x | 1.342x | 0.345x | v06_search | unstable/stable/stable/stable |
| uint32 | 12 | 512 | 0.085275 | 0.074883 | 0.269853 | 0.224819 | 1.139x | 1.200x | 0.333x | v06_search | stable/stable/stable/stable |
| uint32 | 12 | 80 | 0.017928 | 0.016089 | 0.066142 | 0.068350 | 1.114x | 0.968x | 0.235x | v06_search | stable/unstable/stable/stable |
| uint64 | 12 | 1 | 0.007678 | 0.007776 | 0.113754 | 0.096551 | 0.987x | 1.178x | 0.081x | v06 | unstable/unstable/stable/stable |
| uint64 | 12 | 16 | 0.009497 | 0.009099 | 0.113904 | 0.097059 | 1.044x | 1.174x | 0.094x | v06_search | stable/stable/stable/stable |
| uint64 | 12 | 160 | 0.046920 | 0.052124 | 0.233841 | 0.136632 | 0.900x | 1.711x | 0.381x | v06 | stable/unstable/stable/unstable |
| uint64 | 12 | 256 | 0.069100 | 0.068887 | 0.456339 | 0.256881 | 1.003x | 1.776x | 0.268x | v06_search | stable/unstable/stable/stable |
| uint64 | 12 | 512 | 0.126587 | 0.126212 | 0.802269 | 0.512788 | 1.003x | 1.565x | 0.246x | v06_search | stable/unstable/stable/stable |
| uint64 | 12 | 80 | 0.025407 | 0.024222 | 0.114659 | 0.101284 | 1.049x | 1.132x | 0.239x | v06_search | stable/stable/stable/stable |

## Aggregate

| Numeric | Shapes | v0.6 search/base | v0.7 search/base | v0.7 search / v0.6 search | v0.7 wins | Parity | v0.6 wins |
|:--|--:|--:|--:|--:|--:|--:|--:|
| overall | 18 | 1.323x | 1.243x | 0.274x | 0 | 0 | 18 |
| uint32 | 6 | 1.126x | 1.065x | 0.218x | 0 | 0 | 6 |
| uint64 | 12 | 1.435x | 1.342x | 0.307x | 0 | 0 | 12 |
| uint32/logN12 | 6 | 1.126x | 1.065x | 0.218x | 0 | 0 | 6 |
| uint64/logN10 | 6 | 2.066x | 1.290x | 0.499x | 0 | 0 | 6 |
| uint64/logN12 | 6 | 0.996x | 1.397x | 0.189x | 0 | 0 | 6 |
