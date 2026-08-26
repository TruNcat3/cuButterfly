# HybridDataflow benchmark

| case | bits | logN | batch | backend | Us | Ur | Rb | Ud | Td | kernel ms | correct |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| dataflow_w32_n10_b1_ur1_td4 | 32 | 10 | 1 | hybrid-dataflow | 5 | 1 | 1 | 32 | 4 | 0.040940 | 1 |
| dataflow_w32_n10_b1_ur2_td4 | 32 | 10 | 1 | hybrid-dataflow | 5 | 2 | 1 | 32 | 4 | 0.036393 | 1 |
| dataflow_w32_n10_b1_ur5_td4 | 32 | 10 | 1 | hybrid-dataflow | 5 | 5 | 1 | 32 | 4 | 0.029921 | 1 |
| dataflow_w32_n10_b16_ur1_td4 | 32 | 10 | 16 | hybrid-dataflow | 5 | 1 | 1 | 32 | 4 | 0.041103 | 1 |
| dataflow_w32_n10_b16_ur2_td4 | 32 | 10 | 16 | hybrid-dataflow | 5 | 2 | 1 | 32 | 4 | 0.036577 | 1 |
| dataflow_w32_n10_b16_ur5_td4 | 32 | 10 | 16 | hybrid-dataflow | 5 | 5 | 1 | 32 | 4 | 0.030167 | 1 |
| dataflow_w32_n10_b256_ur1_td4 | 32 | 10 | 256 | hybrid-dataflow | 5 | 1 | 1 | 32 | 4 | 0.060846 | 1 |
| dataflow_w32_n10_b256_ur2_td4 | 32 | 10 | 256 | hybrid-dataflow | 5 | 2 | 1 | 32 | 4 | 0.049070 | 1 |
| dataflow_w32_n10_b256_ur5_td4 | 32 | 10 | 256 | hybrid-dataflow | 5 | 5 | 1 | 32 | 4 | 0.039322 | 1 |
| dataflow_w32_n12_b1_ur1_td4 | 32 | 12 | 1 | hybrid-dataflow | 6 | 1 | 1 | 32 | 4 | 0.077210 | 1 |
| dataflow_w32_n12_b1_ur1_td8 | 32 | 12 | 1 | hybrid-dataflow | 6 | 1 | 1 | 32 | 8 | 0.088105 | 1 |
| dataflow_w32_n12_b1_ur2_td4 | 32 | 12 | 1 | hybrid-dataflow | 6 | 2 | 1 | 32 | 4 | 0.114217 | 1 |
| dataflow_w32_n12_b1_ur2_td8 | 32 | 12 | 1 | hybrid-dataflow | 6 | 2 | 1 | 32 | 8 | 0.111862 | 1 |
| dataflow_w32_n12_b1_ur3_td4 | 32 | 12 | 1 | hybrid-dataflow | 6 | 3 | 1 | 32 | 4 | 0.133427 | 1 |
| dataflow_w32_n12_b1_ur3_td8 | 32 | 12 | 1 | hybrid-dataflow | 6 | 3 | 1 | 32 | 8 | 0.103404 | 1 |
| dataflow_w32_n12_b1_ur6_td4 | 32 | 12 | 1 | hybrid-dataflow | 6 | 6 | 1 | 32 | 4 | 0.118149 | 1 |
| dataflow_w32_n12_b1_ur6_td8 | 32 | 12 | 1 | hybrid-dataflow | 6 | 6 | 1 | 32 | 8 | 0.109261 | 1 |
| hybrid2d_w32_n12_b1 | 32 | 12 | 1 | hybrid2d | 0 | 0 | 0 | 0 | 0 | 0.008417 | 1 |
| dataflow_w32_n12_b16_ur1_td4 | 32 | 12 | 16 | hybrid-dataflow | 6 | 1 | 1 | 32 | 4 | 0.077353 | 1 |
| dataflow_w32_n12_b16_ur1_td8 | 32 | 12 | 16 | hybrid-dataflow | 6 | 1 | 1 | 32 | 8 | 0.087921 | 1 |
| dataflow_w32_n12_b16_ur2_td4 | 32 | 12 | 16 | hybrid-dataflow | 6 | 2 | 1 | 32 | 4 | 0.114872 | 1 |
| dataflow_w32_n12_b16_ur2_td8 | 32 | 12 | 16 | hybrid-dataflow | 6 | 2 | 1 | 32 | 8 | 0.112128 | 1 |
| dataflow_w32_n12_b16_ur3_td4 | 32 | 12 | 16 | hybrid-dataflow | 6 | 3 | 1 | 32 | 4 | 0.133591 | 1 |
| dataflow_w32_n12_b16_ur3_td8 | 32 | 12 | 16 | hybrid-dataflow | 6 | 3 | 1 | 32 | 8 | 0.103485 | 1 |
| dataflow_w32_n12_b16_ur6_td4 | 32 | 12 | 16 | hybrid-dataflow | 6 | 6 | 1 | 32 | 4 | 0.118108 | 1 |
| dataflow_w32_n12_b16_ur6_td8 | 32 | 12 | 16 | hybrid-dataflow | 6 | 6 | 1 | 32 | 8 | 0.109404 | 1 |
| hybrid2d_w32_n12_b16 | 32 | 12 | 16 | hybrid2d | 0 | 0 | 0 | 0 | 0 | 0.010035 | 1 |
| dataflow_w32_n12_b256_ur1_td4 | 32 | 12 | 256 | hybrid-dataflow | 6 | 1 | 1 | 32 | 4 | 0.194048 | 1 |
| dataflow_w32_n12_b256_ur1_td8 | 32 | 12 | 256 | hybrid-dataflow | 6 | 1 | 1 | 32 | 8 | 0.192963 | 1 |
| dataflow_w32_n12_b256_ur2_td4 | 32 | 12 | 256 | hybrid-dataflow | 6 | 2 | 1 | 32 | 4 | 0.280576 | 1 |
| dataflow_w32_n12_b256_ur2_td8 | 32 | 12 | 256 | hybrid-dataflow | 6 | 2 | 1 | 32 | 8 | 0.294707 | 1 |
| dataflow_w32_n12_b256_ur3_td4 | 32 | 12 | 256 | hybrid-dataflow | 6 | 3 | 1 | 32 | 4 | 0.300175 | 1 |
| dataflow_w32_n12_b256_ur3_td8 | 32 | 12 | 256 | hybrid-dataflow | 6 | 3 | 1 | 32 | 8 | 0.251146 | 1 |
| dataflow_w32_n12_b256_ur6_td4 | 32 | 12 | 256 | hybrid-dataflow | 6 | 6 | 1 | 32 | 4 | 0.264806 | 1 |
| dataflow_w32_n12_b256_ur6_td8 | 32 | 12 | 256 | hybrid-dataflow | 6 | 6 | 1 | 32 | 8 | 0.230236 | 1 |
| hybrid2d_w32_n12_b256 | 32 | 12 | 256 | hybrid2d | 0 | 0 | 0 | 0 | 0 | 0.054989 | 1 |
| dataflow_w64_n10_b1_ur1_td4 | 64 | 10 | 1 | hybrid-dataflow | 5 | 1 | 1 | 16 | 4 | 0.049193 | 1 |
| dataflow_w64_n10_b1_ur2_td4 | 64 | 10 | 1 | hybrid-dataflow | 5 | 2 | 1 | 16 | 4 | 0.052531 | 1 |
| dataflow_w64_n10_b1_ur5_td4 | 64 | 10 | 1 | hybrid-dataflow | 5 | 5 | 1 | 16 | 4 | 0.047043 | 1 |
| tile256_w64_n10_b1 | 64 | 10 | 1 | tile256 | 0 | 0 | 0 | 0 | 0 | 0.012964 | 1 |
| dataflow_w64_n10_b16_ur1_td4 | 64 | 10 | 16 | hybrid-dataflow | 5 | 1 | 1 | 16 | 4 | 0.049357 | 1 |
| dataflow_w64_n10_b16_ur2_td4 | 64 | 10 | 16 | hybrid-dataflow | 5 | 2 | 1 | 16 | 4 | 0.052511 | 1 |
| dataflow_w64_n10_b16_ur5_td4 | 64 | 10 | 16 | hybrid-dataflow | 5 | 5 | 1 | 16 | 4 | 0.047391 | 1 |
| tile256_w64_n10_b16 | 64 | 10 | 16 | tile256 | 0 | 0 | 0 | 0 | 0 | 0.014561 | 1 |
| dataflow_w64_n10_b256_ur1_td4 | 64 | 10 | 256 | hybrid-dataflow | 5 | 1 | 1 | 16 | 4 | 0.072786 | 1 |
| dataflow_w64_n10_b256_ur2_td4 | 64 | 10 | 256 | hybrid-dataflow | 5 | 2 | 1 | 16 | 4 | 0.069059 | 1 |
| dataflow_w64_n10_b256_ur5_td4 | 64 | 10 | 256 | hybrid-dataflow | 5 | 5 | 1 | 16 | 4 | 0.056545 | 1 |
| tile256_w64_n10_b256 | 64 | 10 | 256 | tile256 | 0 | 0 | 0 | 0 | 0 | 0.035738 | 1 |
| dataflow_w64_n12_b1_ur1_td4 | 64 | 12 | 1 | hybrid-dataflow | 6 | 1 | 1 | 16 | 4 | 0.133857 | 1 |
| dataflow_w64_n12_b1_ur1_td8 | 64 | 12 | 1 | hybrid-dataflow | 6 | 1 | 1 | 16 | 8 | 0.159068 | 1 |
| dataflow_w64_n12_b1_ur2_td4 | 64 | 12 | 1 | hybrid-dataflow | 6 | 2 | 1 | 16 | 4 | 0.157123 | 1 |
| dataflow_w64_n12_b1_ur2_td8 | 64 | 12 | 1 | hybrid-dataflow | 6 | 2 | 1 | 16 | 8 | 0.158106 | 1 |
| dataflow_w64_n12_b1_ur3_td4 | 64 | 12 | 1 | hybrid-dataflow | 6 | 3 | 1 | 16 | 4 | 0.200212 | 1 |
| dataflow_w64_n12_b1_ur3_td8 | 64 | 12 | 1 | hybrid-dataflow | 6 | 3 | 1 | 16 | 8 | 0.156836 | 1 |
| dataflow_w64_n12_b1_ur6_td4 | 64 | 12 | 1 | hybrid-dataflow | 6 | 6 | 1 | 16 | 4 | 0.173138 | 1 |
| dataflow_w64_n12_b1_ur6_td8 | 64 | 12 | 1 | hybrid-dataflow | 6 | 6 | 1 | 16 | 8 | 0.150753 | 1 |
| dataflow_w64_n12_b1_ur6_td8_rb3 | 64 | 12 | 1 | hybrid-dataflow | 6 | 6 | 3 | 16 | 8 | 0.153559 | 1 |
| tile256_w64_n12_b1 | 64 | 12 | 1 | tile256 | 0 | 0 | 0 | 0 | 0 | 0.022651 | 1 |
| dataflow_w64_n12_b16_ur1_td4 | 64 | 12 | 16 | hybrid-dataflow | 6 | 1 | 1 | 16 | 4 | 0.133509 | 1 |
| dataflow_w64_n12_b16_ur1_td8 | 64 | 12 | 16 | hybrid-dataflow | 6 | 1 | 1 | 16 | 8 | 0.159027 | 1 |
| dataflow_w64_n12_b16_ur2_td4 | 64 | 12 | 16 | hybrid-dataflow | 6 | 2 | 1 | 16 | 4 | 0.158228 | 1 |
| dataflow_w64_n12_b16_ur2_td8 | 64 | 12 | 16 | hybrid-dataflow | 6 | 2 | 1 | 16 | 8 | 0.158433 | 1 |
| dataflow_w64_n12_b16_ur3_td4 | 64 | 12 | 16 | hybrid-dataflow | 6 | 3 | 1 | 16 | 4 | 0.201503 | 1 |
| dataflow_w64_n12_b16_ur3_td8 | 64 | 12 | 16 | hybrid-dataflow | 6 | 3 | 1 | 16 | 8 | 0.157716 | 1 |
| dataflow_w64_n12_b16_ur6_td4 | 64 | 12 | 16 | hybrid-dataflow | 6 | 6 | 1 | 16 | 4 | 0.173650 | 1 |
| dataflow_w64_n12_b16_ur6_td8 | 64 | 12 | 16 | hybrid-dataflow | 6 | 6 | 1 | 16 | 8 | 0.151122 | 1 |
| dataflow_w64_n12_b16_ur6_td8_rb3 | 64 | 12 | 16 | hybrid-dataflow | 6 | 6 | 3 | 16 | 8 | 0.154665 | 1 |
| tile256_w64_n12_b16 | 64 | 12 | 16 | tile256 | 0 | 0 | 0 | 0 | 0 | 0.026051 | 1 |
| dataflow_w64_n12_b256_ur1_td4 | 64 | 12 | 256 | hybrid-dataflow | 6 | 1 | 1 | 16 | 4 | 0.528773 | 1 |
| dataflow_w64_n12_b256_ur1_td8 | 64 | 12 | 256 | hybrid-dataflow | 6 | 1 | 1 | 16 | 8 | 0.595948 | 1 |
| dataflow_w64_n12_b256_ur2_td4 | 64 | 12 | 256 | hybrid-dataflow | 6 | 2 | 1 | 16 | 4 | 0.404951 | 1 |
| dataflow_w64_n12_b256_ur2_td8 | 64 | 12 | 256 | hybrid-dataflow | 6 | 2 | 1 | 16 | 8 | 0.630804 | 1 |
| dataflow_w64_n12_b256_ur3_td4 | 64 | 12 | 256 | hybrid-dataflow | 6 | 3 | 1 | 16 | 4 | 0.471654 | 1 |
| dataflow_w64_n12_b256_ur3_td8 | 64 | 12 | 256 | hybrid-dataflow | 6 | 3 | 1 | 16 | 8 | 0.415990 | 1 |
| dataflow_w64_n12_b256_ur6_td4 | 64 | 12 | 256 | hybrid-dataflow | 6 | 6 | 1 | 16 | 4 | 0.400609 | 1 |
| dataflow_w64_n12_b256_ur6_td8 | 64 | 12 | 256 | hybrid-dataflow | 6 | 6 | 1 | 16 | 8 | 0.362025 | 1 |
| dataflow_w64_n12_b256_ur6_td8_rb3 | 64 | 12 | 256 | hybrid-dataflow | 6 | 6 | 3 | 16 | 8 | 0.456602 | 1 |
| tile256_w64_n12_b256 | 64 | 12 | 256 | tile256 | 0 | 0 | 0 | 0 | 0 | 0.181309 | 1 |

## Best generated point versus baseline

| bits | logN | batch | baseline | baseline ms | best Us | best Ur | best Rb | best Td | dataflow ms | baseline/dataflow |
|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 32 | 12 | 1 | hybrid2d | 0.008417 | 6 | 1 | 1 | 4 | 0.077210 | 0.109x |
| 32 | 12 | 16 | hybrid2d | 0.010035 | 6 | 1 | 1 | 4 | 0.077353 | 0.130x |
| 32 | 12 | 256 | hybrid2d | 0.054989 | 6 | 1 | 1 | 8 | 0.192963 | 0.285x |
| 64 | 10 | 1 | tile256 | 0.012964 | 5 | 5 | 1 | 4 | 0.047043 | 0.276x |
| 64 | 10 | 16 | tile256 | 0.014561 | 5 | 5 | 1 | 4 | 0.047391 | 0.307x |
| 64 | 10 | 256 | tile256 | 0.035738 | 5 | 5 | 1 | 4 | 0.056545 | 0.632x |
| 64 | 12 | 1 | tile256 | 0.022651 | 6 | 1 | 1 | 4 | 0.133857 | 0.169x |
| 64 | 12 | 16 | tile256 | 0.026051 | 6 | 1 | 1 | 4 | 0.133509 | 0.195x |
| 64 | 12 | 256 | tile256 | 0.181309 | 6 | 6 | 1 | 8 | 0.362025 | 0.501x |
