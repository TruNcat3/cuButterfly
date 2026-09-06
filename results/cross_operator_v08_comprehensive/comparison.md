# Cross-Operator v0.8 Comparison

Hardware: `v100-sm70`. Ratios above 1 mean v0.8 has lower kernel time.
The `status` column is part of the claim: compatibility-lowered points are not presented as native resident subgraphs.

| operator | precision | logN | batch | v0.6 ms | v0.8 search ms | search/v0.6 | model ms | model/v0.6 | model/search | library/external | search/library | model/library | implementation | status |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|:--|--:|--:|:--|:--|
| fft | fp32 | 14 | 1 | 0.020736 | 0.020736 | 1.000x | n/a | n/a | n/a | cuFFT (0.012186 ms) | 0.588x | n/a | v08-inherited-cufftdx-direct | v08-inherited-v06 |
| fft | fp32 | 14 | 4 | 0.020838 | 0.020838 | 1.000x | n/a | n/a | n/a | cuFFT (0.012186 ms) | 0.585x | n/a | v08-inherited-cufftdx-direct | v08-inherited-v06 |
| fft | fp32 | 14 | 16 | 0.020941 | 0.020941 | 1.000x | n/a | n/a | n/a | cuFFT (0.013722 ms) | 0.655x | n/a | v08-inherited-cufftdx-direct | v08-inherited-v06 |
| fft | fp32 | 14 | 64 | 0.032410 | 0.032410 | 1.000x | n/a | n/a | n/a | cuFFT (0.027238 ms) | 0.840x | n/a | v08-inherited-cufftdx-direct | v08-inherited-v06 |
| fwht | fp32 | 15 | 1 | 0.016691 | 0.007885 | 2.117x | n/a | n/a | n/a | none-matched | n/a | n/a | online-radix4-8x7 | compatibility-lowered-two-segment |
| fwht | fp32 | 15 | 4 | 0.020275 | 0.015616 | 1.298x | n/a | n/a | n/a | none-matched | n/a | n/a | online-radix4-8x7 | compatibility-lowered-two-segment |
| fwht | fp32 | 15 | 16 | 0.032461 | 0.032461 | 1.000x | n/a | n/a | n/a | none-matched | n/a | n/a | v08-inherited-hierarchical-radix4 | v08-inherited-v06 |
| fwht | fp32 | 15 | 64 | 0.185139 | 0.161843 | 1.144x | n/a | n/a | n/a | none-matched | n/a | n/a | online-radix4-8x7 | compatibility-lowered-two-segment |
| ntt | uint32 | 20 | 1 | 0.085965 | 0.085965 | 1.000x | n/a | n/a | n/a | none-matched | n/a | n/a | v08-inherited-hybrid2d-radix4-10x10 | v08-inherited-v06 |
| ntt | uint32 | 20 | 4 | 0.280576 | 0.280576 | 1.000x | n/a | n/a | n/a | none-matched | n/a | n/a | v08-inherited-hybrid2d-radix4-10x10 | v08-inherited-v06 |
| ntt | uint32 | 20 | 16 | 1.052006 | 1.014835 | 1.037x | n/a | n/a | n/a | none-matched | n/a | n/a | resident-queue-radix4-10x10 | native-resident-subgraph |
| ntt | uint32 | 20 | 64 | 4.368179 | 4.368179 | 1.000x | n/a | n/a | n/a | none-matched | n/a | n/a | v08-inherited-hybrid2d-radix4-10x10 | v08-inherited-v06 |
| structured-2x2 | fp32 | 15 | 1 | 0.017101 | 0.008960 | 1.909x | n/a | n/a | n/a | none-matched | n/a | n/a | online-radix4-8x7 | compatibility-lowered-two-segment |
| structured-2x2 | fp32 | 15 | 4 | 0.020992 | 0.016691 | 1.258x | n/a | n/a | n/a | none-matched | n/a | n/a | online-radix4-8x7 | compatibility-lowered-two-segment |
| structured-2x2 | fp32 | 15 | 16 | 0.035021 | 0.035021 | 1.000x | n/a | n/a | n/a | none-matched | n/a | n/a | v08-inherited-hierarchical-radix4 | v08-inherited-v06 |
| structured-2x2 | fp32 | 15 | 64 | 0.188160 | 0.167885 | 1.121x | n/a | n/a | n/a | none-matched | n/a | n/a | online-radix4-8x7 | compatibility-lowered-two-segment |
| subset-zeta | uint32 | 15 | 1 | 0.020429 | 0.007936 | 2.574x | n/a | n/a | n/a | none-matched | n/a | n/a | online-radix4-8x7 | compatibility-lowered-two-segment |
| subset-zeta | uint32 | 15 | 4 | 0.020122 | 0.015514 | 1.297x | n/a | n/a | n/a | none-matched | n/a | n/a | online-radix4-8x7 | compatibility-lowered-two-segment |
| subset-zeta | uint32 | 15 | 16 | 0.031744 | 0.031744 | 1.000x | n/a | n/a | n/a | none-matched | n/a | n/a | v08-inherited-hierarchical-radix4 | v08-inherited-v06 |
| subset-zeta | uint32 | 15 | 64 | 0.185088 | 0.161690 | 1.145x | n/a | n/a | n/a | none-matched | n/a | n/a | online-radix4-8x7 | compatibility-lowered-two-segment |
| superset-zeta | uint32 | 15 | 1 | 0.016589 | 0.007987 | 2.077x | n/a | n/a | n/a | none-matched | n/a | n/a | online-radix4-8x7 | compatibility-lowered-two-segment |
| superset-zeta | uint32 | 15 | 4 | 0.020275 | 0.015514 | 1.307x | n/a | n/a | n/a | none-matched | n/a | n/a | online-radix4-8x7 | compatibility-lowered-two-segment |
| superset-zeta | uint32 | 15 | 16 | 0.031846 | 0.031846 | 1.000x | n/a | n/a | n/a | none-matched | n/a | n/a | v08-inherited-hierarchical-radix4 | v08-inherited-v06 |
| superset-zeta | uint32 | 15 | 64 | 0.184934 | 0.161690 | 1.144x | n/a | n/a | n/a | none-matched | n/a | n/a | online-radix4-8x7 | compatibility-lowered-two-segment |
| xor-zeta | uint32 | 15 | 1 | 0.016691 | 0.007936 | 2.103x | n/a | n/a | n/a | none-matched | n/a | n/a | online-radix4-8x7 | compatibility-lowered-two-segment |
| xor-zeta | uint32 | 15 | 4 | 0.020275 | 0.015462 | 1.311x | n/a | n/a | n/a | none-matched | n/a | n/a | online-radix4-8x7 | compatibility-lowered-two-segment |
| xor-zeta | uint32 | 15 | 16 | 0.031693 | 0.031693 | 1.000x | n/a | n/a | n/a | none-matched | n/a | n/a | v08-inherited-hierarchical-radix4 | v08-inherited-v06 |
| xor-zeta | uint32 | 15 | 64 | 0.185344 | 0.161638 | 1.147x | n/a | n/a | n/a | none-matched | n/a | n/a | online-radix4-8x7 | compatibility-lowered-two-segment |

## Interpretation

FFT and NTT can use compiled specialized physical cores in this matrix. FWHT, subset/superset zeta, legacy xor-zeta, and structured-2x2 are covered by the same mapping/profile protocol, while their v0.8 rows remain explicitly marked compatibility-lowered until a native resident operator-templated chain is compiled.

The model columns are measured executions of the runtime-selected point; their `SelectionInfo` prediction is retained in the raw benchmark CSV. No external specialized baseline is fabricated for zeta or structured-2x2; the matching Dao-AILab FHT and GPU-NTT rows are joined by the separate external-baseline protocol when available.

## Aggregate View

Geometric means summarize ratios across batch/length cells; they do not replace the per-cell table.

| operator | cells | search/v0.6 | model/v0.6 | model/search | search/library | model/library | search wins vs v0.6 |
|:--|--:|--:|--:|--:|--:|--:|--:|
| fft | 4 | 1.000x | n/a | n/a | 0.660x | n/a | 0/4 |
| fwht | 4 | 1.332x | n/a | n/a | n/a | n/a | 3/4 |
| ntt | 4 | 1.009x | n/a | n/a | n/a | n/a | 1/4 |
| structured-2x2 | 4 | 1.281x | n/a | n/a | n/a | n/a | 3/4 |
| subset-zeta | 4 | 1.398x | n/a | n/a | n/a | n/a | 3/4 |
| superset-zeta | 4 | 1.327x | n/a | n/a | n/a | n/a | 3/4 |
| xor-zeta | 4 | 1.334x | n/a | n/a | n/a | n/a | 3/4 |
