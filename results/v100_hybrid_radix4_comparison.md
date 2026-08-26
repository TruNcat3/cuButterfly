# V100 HybridDataflow Resident Radix-4 Comparison

All rows use 30 warmups, 500 timed iterations, three randomized process trials, and correctness checks.
The v0.6 column is the measured minimum of the legal mature cores in that group.

| Numeric | logN | Batch | v0.6 best | v0.6 ms | v0.7 pipeline ms | v0.7 resident R4 ms | R4/pipeline | R4/v0.6 throughput | Winner |
|:--|--:|--:|:--|--:|--:|--:|--:|--:|:--|
| uint64 | 10 | 1 | v0.6-tile256 | 0.011057 | 0.036110 | 0.006103 | 5.917x | 1.812x | v0.7-resident-radix4 |
| uint64 | 10 | 16 | v0.6-tile256 | 0.012466 | 0.031060 | 0.006197 | 5.012x | 2.012x | v0.7-resident-radix4 |
| uint64 | 10 | 160 | v0.6-tile256 | 0.022813 | 0.033976 | 0.009124 | 3.724x | 2.500x | v0.7-resident-radix4 |
| uint64 | 10 | 256 | v0.6-tile256 | 0.030497 | 0.040401 | 0.014801 | 2.730x | 2.060x | v0.7-resident-radix4 |
| uint64 | 10 | 512 | v0.6-tile256 | 0.052978 | 0.085606 | 0.025723 | 3.328x | 2.060x | v0.7-resident-radix4 |
| uint64 | 10 | 80 | v0.6-tile256 | 0.017175 | 0.031238 | 0.007053 | 4.429x | 2.435x | v0.7-resident-radix4 |
| uint32 | 12 | 1 | v0.6-hybrid2d-radix4 | 0.005962 | 0.068157 | 0.013101 | 5.202x | 0.455x | v0.6-hybrid2d-radix4 |
| uint32 | 12 | 16 | v0.6-hybrid2d-radix2 | 0.008479 | 0.068176 | 0.013482 | 5.057x | 0.629x | v0.6-hybrid2d-radix2 |
| uint32 | 12 | 160 | v0.6-hybrid2d-radix4 | 0.027066 | 0.090034 | 0.023699 | 3.799x | 1.142x | v0.7-resident-radix4 |
| uint32 | 12 | 256 | v0.6-hybrid2d-radix4 | 0.045158 | 0.124750 | 0.050289 | 2.481x | 0.898x | v0.6-hybrid2d-radix4 |
| uint32 | 12 | 512 | v0.6-hybrid2d-radix4 | 0.074924 | 0.224899 | 0.085965 | 2.616x | 0.872x | v0.6-hybrid2d-radix4 |
| uint32 | 12 | 80 | v0.6-hybrid2d-radix4 | 0.016087 | 0.068352 | 0.015268 | 4.477x | 1.054x | v0.7-resident-radix4 |
| uint64 | 12 | 1 | v0.6-hybrid2d-radix4 | 0.007692 | 0.096545 | 0.019569 | 4.934x | 0.393x | v0.6-hybrid2d-radix4 |
| uint64 | 12 | 16 | v0.6-hybrid2d-radix4 | 0.009107 | 0.097083 | 0.019683 | 4.932x | 0.463x | v0.6-hybrid2d-radix4 |
| uint64 | 12 | 160 | v0.6-hybrid2d-radix2 | 0.046918 | 0.137511 | 0.049011 | 2.806x | 0.957x | v0.6-hybrid2d-radix2 |
| uint64 | 12 | 256 | v0.6-hybrid2d-radix4 | 0.068846 | 0.256842 | 0.119343 | 2.152x | 0.577x | v0.6-hybrid2d-radix4 |
| uint64 | 12 | 512 | v0.6-hybrid2d-radix4 | 0.125809 | 0.512705 | 0.177195 | 2.893x | 0.710x | v0.6-hybrid2d-radix4 |
| uint64 | 12 | 80 | v0.6-hybrid2d-radix4 | 0.024224 | 0.101263 | 0.023560 | 4.298x | 1.028x | v0.7-resident-radix4 |

## Aggregate

| Numeric | Shapes | R4/pipeline | R4/v0.6 throughput | R4 wins | Parity | v0.6 wins |
|:--|--:|--:|--:|--:|--:|--:|
| overall | 18 | 3.769x | 1.035x | 8 | 1 | 9 |
| uint32/logN12 | 6 | 3.777x | 0.804x | 2 | 0 | 4 |
| uint64/logN10 | 6 | 4.055x | 2.133x | 6 | 0 | 0 |
| uint64/logN12 | 6 | 3.497x | 0.647x | 0 | 1 | 5 |
