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
| uint64 | 10 | 1 | 0.024799 | 0.011026 | 0.041701 | 0.006109 | 2.249x | 6.826x | 1.805x | v07_search | unstable/stable/stable/stable |
| uint64 | 10 | 16 | 0.026071 | 0.012337 | 0.041796 | 0.006199 | 2.113x | 6.742x | 1.990x | v07_search | stable/unstable/stable/unstable |
| uint64 | 10 | 160 | 0.044626 | 0.022807 | 0.045912 | 0.009112 | 1.957x | 5.039x | 2.503x | v07_search | stable/stable/stable/stable |
| uint64 | 10 | 256 | 0.056478 | 0.030491 | 0.062212 | 0.014803 | 1.852x | 4.203x | 2.060x | v07_search | unstable/stable/stable/unstable |
| uint64 | 10 | 512 | 0.088406 | 0.052974 | 0.115397 | 0.025747 | 1.669x | 4.482x | 2.057x | v07_search | unstable/stable/stable/unstable |
| uint64 | 10 | 80 | 0.035334 | 0.017152 | 0.042039 | 0.007049 | 2.060x | 5.964x | 2.433x | v07_search | stable/stable/stable/stable |
| uint32 | 12 | 1 | 0.006867 | 0.005982 | 0.076644 | 0.013097 | 1.148x | 5.852x | 0.457x | v06_search | stable/unstable/unstable/stable |
| uint32 | 12 | 16 | 0.008468 | 0.007645 | 0.065939 | 0.013482 | 1.108x | 4.891x | 0.567x | v06_search | stable/stable/unstable/stable |
| uint32 | 12 | 160 | 0.030843 | 0.027054 | 0.077517 | 0.023716 | 1.140x | 3.269x | 1.141x | v07_search | unstable/stable/stable/stable |
| uint32 | 12 | 256 | 0.048208 | 0.042994 | 0.167541 | 0.049975 | 1.121x | 3.352x | 0.860x | v06_search | stable/stable/unstable/stable |
| uint32 | 12 | 512 | 0.085262 | 0.074904 | 0.269652 | 0.086041 | 1.138x | 3.134x | 0.871x | v06_search | stable/unstable/unstable/stable |
| uint32 | 12 | 80 | 0.017932 | 0.016079 | 0.066132 | 0.015256 | 1.115x | 4.335x | 1.054x | v07_search | stable/stable/stable/stable |
| uint64 | 12 | 1 | 0.007676 | 0.007686 | 0.113795 | 0.019556 | 0.999x | 5.819x | 0.393x | v06 | stable/stable/unstable/stable |
| uint64 | 12 | 16 | 0.009492 | 0.009097 | 0.113902 | 0.019663 | 1.043x | 5.793x | 0.463x | v06_search | stable/stable/stable/stable |
| uint64 | 12 | 160 | 0.048859 | 0.046750 | 0.233808 | 0.048869 | 1.045x | 4.784x | 0.957x | v06_search | unstable/stable/stable/stable |
| uint64 | 12 | 256 | 0.069085 | 0.068852 | 0.456323 | 0.119271 | 1.003x | 3.826x | 0.577x | v06_search | stable/stable/stable/stable |
| uint64 | 12 | 512 | 0.126538 | 0.125776 | 0.802271 | 0.170031 | 1.006x | 4.718x | 0.740x | v06_search | stable/stable/stable/stable |
| uint64 | 12 | 80 | 0.025405 | 0.024199 | 0.114526 | 0.023558 | 1.050x | 4.861x | 1.027x | v07_search | stable/stable/stable/unstable |

## Aggregate

| Numeric | Shapes | v0.6 search/base | v0.7 search/base | v0.7 search / v0.6 search | v0.7 wins | Parity | v0.6 wins |
|:--|--:|--:|--:|--:|--:|--:|--:|
| overall | 18 | 1.316x | 4.760x | 1.028x | 8 | 1 | 9 |
| uint32 | 6 | 1.128x | 4.026x | 0.785x | 2 | 0 | 4 |
| uint64 | 12 | 1.422x | 5.175x | 1.177x | 6 | 1 | 5 |
| uint32/logN12 | 6 | 1.128x | 4.026x | 0.785x | 2 | 0 | 4 |
| uint64/logN10 | 6 | 1.974x | 5.445x | 2.127x | 6 | 0 | 0 |
| uint64/logN12 | 6 | 1.024x | 4.918x | 0.651x | 0 | 1 | 5 |
