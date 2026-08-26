# v0.7 Dynamic Instruction Attribution

Counts are dynamic warp instructions from NCU SourceCounters. SASS PCs are deduplicated across inline source-stack views.

| category | v0.6 (M) | v0.7 d6 (M) | delta (M) | candidate/baseline |
|:--|--:|--:|--:|--:|
| control | 9.621 | 32.681 | 23.060 | 3.397x |
| shuffle | 1.017 | 22.507 | 21.489 | 22.123x |
| integer/arithmetic | 150.393 | 170.359 | 19.966 | 1.133x |
| local memory | 0.008 | 0.806 | 0.798 | 98.354x |
| synchronization/cache | 0.666 | 1.457 | 0.790 | 2.186x |
| global memory | 9.967 | 9.704 | -0.262 | 0.974x |
| scheduling | 1.688 | 1.308 | -0.380 | 0.775x |
| shared memory | 12.616 | 7.602 | -5.014 | 0.603x |
| **total** | **185.976** | **246.424** | **60.447** | **1.325x** |

## Largest Opcode Deltas

| opcode | v0.6 (M) | v0.7 d6 (M) | delta (M) |
|:--|--:|--:|--:|
| SHFL | 1.017 | 22.507 | 21.489 |
| SEL | 15.734 | 28.937 | 13.203 |
| ISETP | 20.951 | 30.201 | 9.250 |
| BSYNC | 0.669 | 9.615 | 8.946 |
| BRA | 5.859 | 10.682 | 4.822 |
| BSSY | 0.628 | 4.953 | 4.325 |
| BMOV | 0.628 | 4.953 | 4.325 |
| IMAD | 67.749 | 71.874 | 4.125 |
| MOV | 0.058 | 1.088 | 1.030 |
| LDL | 0.000 | 0.798 | 0.798 |
| CCTL | 0.063 | 0.802 | 0.739 |
| LOP3 | 4.469 | 5.182 | 0.713 |

## Interpretation

The largest positive category is **control**, contributing 38.1% of the net 60.447M-instruction gap. Optimize that physical-codelet path before further CTA scheduling scans.
Source coverage: v0.6=621 PCs, v0.7 d6=3514 PCs.
