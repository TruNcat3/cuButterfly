# HybridDataflow benchmark

| case | bits | logN | batch | backend | Us | Ud | Td | kernel ms | correct |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|
| dataflow_default_w32_n6_b1 | 32 | 6 | 1 | hybrid-dataflow | 6 | 32 | 4 | 0.006656 | 1 |
| dataflow_search_w32_n6_b1_us2 | 32 | 6 | 1 | hybrid-dataflow | 2 | 32 | 4 | 0.027290 | 1 |
| dataflow_search_w32_n6_b1_us4 | 32 | 6 | 1 | hybrid-dataflow | 4 | 32 | 4 | 0.017152 | 1 |
| dataflow_default_w32_n6_b16 | 32 | 6 | 16 | hybrid-dataflow | 6 | 32 | 4 | 0.006758 | 1 |
| dataflow_search_w32_n6_b16_us2 | 32 | 6 | 16 | hybrid-dataflow | 2 | 32 | 4 | 0.027392 | 1 |
| dataflow_search_w32_n6_b16_us4 | 32 | 6 | 16 | hybrid-dataflow | 4 | 32 | 4 | 0.017254 | 1 |
| dataflow_default_w32_n8_b1 | 32 | 8 | 1 | hybrid-dataflow | 8 | 32 | 4 | 0.013363 | 1 |
| dataflow_search_w32_n8_b1_us2 | 32 | 8 | 1 | hybrid-dataflow | 2 | 32 | 4 | 0.115149 | 1 |
| dataflow_search_w32_n8_b1_us4 | 32 | 8 | 1 | hybrid-dataflow | 4 | 32 | 4 | 0.026163 | 1 |
| dataflow_default_w32_n8_b16 | 32 | 8 | 16 | hybrid-dataflow | 8 | 32 | 4 | 0.013363 | 1 |
| dataflow_search_w32_n8_b16_us2 | 32 | 8 | 16 | hybrid-dataflow | 2 | 32 | 4 | 0.115302 | 1 |
| dataflow_search_w32_n8_b16_us4 | 32 | 8 | 16 | hybrid-dataflow | 4 | 32 | 4 | 0.026112 | 1 |
| dataflow_default_w32_n10_b1 | 32 | 10 | 1 | hybrid-dataflow | 5 | 32 | 4 | 0.045568 | 1 |
| dataflow_search_w32_n10_b1_us2 | 32 | 10 | 1 | hybrid-dataflow | 2 | 32 | 4 | 0.539546 | 1 |
| dataflow_search_w32_n10_b1_us4 | 32 | 10 | 1 | hybrid-dataflow | 4 | 32 | 4 | 0.183552 | 1 |
| dataflow_default_w32_n10_b16 | 32 | 10 | 16 | hybrid-dataflow | 5 | 32 | 4 | 0.045773 | 1 |
| dataflow_search_w32_n10_b16_us2 | 32 | 10 | 16 | hybrid-dataflow | 2 | 32 | 4 | 0.539597 | 1 |
| dataflow_search_w32_n10_b16_us4 | 32 | 10 | 16 | hybrid-dataflow | 4 | 32 | 4 | 0.183603 | 1 |
| dataflow_default_w32_n12_b1 | 32 | 12 | 1 | hybrid-dataflow | 6 | 32 | 4 | 0.084992 | 1 |
| dataflow_search_w32_n12_b1_us2 | 32 | 12 | 1 | hybrid-dataflow | 2 | 32 | 4 | 2.540800 | 1 |
| dataflow_search_w32_n12_b1_us4 | 32 | 12 | 1 | hybrid-dataflow | 4 | 32 | 4 | 0.336845 | 1 |
| hybrid2d_w32_n12_b1 | 32 | 12 | 1 | hybrid2d | 0 | 0 | 0 | 0.008090 | 1 |
| dataflow_default_w32_n12_b16 | 32 | 12 | 16 | hybrid-dataflow | 6 | 32 | 4 | 0.077414 | 1 |
| dataflow_search_w32_n12_b16_us2 | 32 | 12 | 16 | hybrid-dataflow | 2 | 32 | 4 | 2.303488 | 1 |
| dataflow_search_w32_n12_b16_us4 | 32 | 12 | 16 | hybrid-dataflow | 4 | 32 | 4 | 0.312576 | 1 |
| hybrid2d_w32_n12_b16 | 32 | 12 | 16 | hybrid2d | 0 | 0 | 0 | 0.009114 | 1 |
| dataflow_default_w64_n6_b1 | 64 | 6 | 1 | hybrid-dataflow | 6 | 16 | 4 | 0.007680 | 1 |
| dataflow_search_w64_n6_b1_us2 | 64 | 6 | 1 | hybrid-dataflow | 2 | 16 | 4 | 0.028262 | 1 |
| dataflow_search_w64_n6_b1_us4 | 64 | 6 | 1 | hybrid-dataflow | 4 | 16 | 4 | 0.017254 | 1 |
| tile256_w64_n6_b1 | 64 | 6 | 1 | tile256 | 0 | 0 | 0 | 0.005427 | 1 |
| dataflow_default_w64_n6_b16 | 64 | 6 | 16 | hybrid-dataflow | 6 | 16 | 4 | 0.007834 | 1 |
| dataflow_search_w64_n6_b16_us2 | 64 | 6 | 16 | hybrid-dataflow | 2 | 16 | 4 | 0.028467 | 1 |
| dataflow_search_w64_n6_b16_us4 | 64 | 6 | 16 | hybrid-dataflow | 4 | 16 | 4 | 0.017459 | 1 |
| tile256_w64_n6_b16 | 64 | 6 | 16 | tile256 | 0 | 0 | 0 | 0.005632 | 1 |
| dataflow_default_w64_n8_b1 | 64 | 8 | 1 | hybrid-dataflow | 8 | 16 | 2 | 0.021555 | 1 |
| dataflow_search_w64_n8_b1_us2 | 64 | 8 | 1 | hybrid-dataflow | 2 | 16 | 4 | 0.120832 | 1 |
| dataflow_search_w64_n8_b1_us4 | 64 | 8 | 1 | hybrid-dataflow | 4 | 16 | 4 | 0.026470 | 1 |
| stage_pipeline_w64_n8_b1 | 64 | 8 | 1 | stage-pipeline | 8 | 0 | 0 | 0.009882 | 1 |
| tile256_w64_n8_b1 | 64 | 8 | 1 | tile256 | 0 | 0 | 0 | 0.006656 | 1 |
| dataflow_default_w64_n8_b16 | 64 | 8 | 16 | hybrid-dataflow | 8 | 16 | 2 | 0.021555 | 1 |
| dataflow_search_w64_n8_b16_us2 | 64 | 8 | 16 | hybrid-dataflow | 2 | 16 | 4 | 0.120832 | 1 |
| dataflow_search_w64_n8_b16_us4 | 64 | 8 | 16 | hybrid-dataflow | 4 | 16 | 4 | 0.026470 | 1 |
| stage_pipeline_w64_n8_b16 | 64 | 8 | 16 | stage-pipeline | 8 | 0 | 0 | 0.010240 | 1 |
| tile256_w64_n8_b16 | 64 | 8 | 16 | tile256 | 0 | 0 | 0 | 0.006707 | 1 |
| dataflow_default_w64_n10_b1 | 64 | 10 | 1 | hybrid-dataflow | 5 | 16 | 4 | 0.045414 | 1 |
| dataflow_search_w64_n10_b1_us2 | 64 | 10 | 1 | hybrid-dataflow | 2 | 16 | 4 | 0.567091 | 1 |
| dataflow_search_w64_n10_b1_us4 | 64 | 10 | 1 | hybrid-dataflow | 4 | 16 | 4 | 0.180787 | 1 |
| tile256_w64_n10_b1 | 64 | 10 | 1 | tile256 | 0 | 0 | 0 | 0.011264 | 1 |
| dataflow_default_w64_n10_b16 | 64 | 10 | 16 | hybrid-dataflow | 5 | 16 | 4 | 0.045466 | 1 |
| dataflow_search_w64_n10_b16_us2 | 64 | 10 | 16 | hybrid-dataflow | 2 | 16 | 4 | 0.565862 | 1 |
| dataflow_search_w64_n10_b16_us4 | 64 | 10 | 16 | hybrid-dataflow | 4 | 16 | 4 | 0.180838 | 1 |
| tile256_w64_n10_b16 | 64 | 10 | 16 | tile256 | 0 | 0 | 0 | 0.012698 | 1 |
| dataflow_default_w64_n12_b1 | 64 | 12 | 1 | hybrid-dataflow | 6 | 16 | 4 | 0.125491 | 1 |
| dataflow_search_w64_n12_b1_us2 | 64 | 12 | 1 | hybrid-dataflow | 2 | 16 | 4 | 2.681344 | 1 |
| dataflow_search_w64_n12_b1_us4 | 64 | 12 | 1 | hybrid-dataflow | 4 | 16 | 4 | 0.354867 | 1 |
| tile256_w64_n12_b1 | 64 | 12 | 1 | tile256 | 0 | 0 | 0 | 0.016077 | 1 |
| dataflow_default_w64_n12_b16 | 64 | 12 | 16 | hybrid-dataflow | 6 | 16 | 4 | 0.124416 | 1 |
| dataflow_search_w64_n12_b16_us2 | 64 | 12 | 16 | hybrid-dataflow | 2 | 16 | 4 | 2.649600 | 1 |
| dataflow_search_w64_n12_b16_us4 | 64 | 12 | 16 | hybrid-dataflow | 4 | 16 | 4 | 0.352307 | 1 |
| tile256_w64_n12_b16 | 64 | 12 | 16 | tile256 | 0 | 0 | 0 | 0.022170 | 1 |

## Best generated point versus baseline

| bits | logN | batch | baseline | baseline ms | best Us | best Td | dataflow ms | baseline/dataflow |
|---:|---:|---:|---|---:|---:|---:|---:|---:|
| 32 | 12 | 1 | hybrid2d | 0.008090 | 6 | 4 | 0.084992 | 0.095x |
| 32 | 12 | 16 | hybrid2d | 0.009114 | 6 | 4 | 0.077414 | 0.118x |
| 64 | 6 | 1 | tile256 | 0.005427 | 6 | 4 | 0.007680 | 0.707x |
| 64 | 6 | 16 | tile256 | 0.005632 | 6 | 4 | 0.007834 | 0.719x |
| 64 | 8 | 1 | tile256 | 0.006656 | 8 | 2 | 0.021555 | 0.309x |
| 64 | 8 | 16 | tile256 | 0.006707 | 8 | 2 | 0.021555 | 0.311x |
| 64 | 10 | 1 | tile256 | 0.011264 | 5 | 4 | 0.045414 | 0.248x |
| 64 | 10 | 16 | tile256 | 0.012698 | 5 | 4 | 0.045466 | 0.279x |
| 64 | 12 | 1 | tile256 | 0.016077 | 6 | 4 | 0.125491 | 0.128x |
| 64 | 12 | 16 | tile256 | 0.022170 | 6 | 4 | 0.124416 | 0.178x |
