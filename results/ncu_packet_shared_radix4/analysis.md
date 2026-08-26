# Packet-Shared Radix-4 Fixed-Clock Confirmation

| metric | v0.6 | d7 | lane32 | packet128 |
|:--|--:|--:|--:|--:|
| time (us) | 1259.520 | 1356.960 | 1342.144 | 1061.184 |
| warp instructions | 181,973,064 | 237,176,800 | 237,731,061 | 172,467,507 |
| ADU pipe instructions | 498,400 | 190,130 | 194,148 | 133,344 |
| ALU pipe instructions | 78,567,352 | 92,764,983 | 93,174,847 | 82,873,115 |
| CBU pipe instructions | 8,562,069 | 31,497,430 | 32,167,217 | 3,616,559 |
| LSU pipe instructions | 24,123,416 | 40,882,338 | 40,366,198 | 22,461,951 |
| XU pipe instructions | 1,057,728 | 1,052,896 | 1,052,896 | 1,053,376 |
| global-memory instructions | 10,054,358 | 9,152,895 | 8,667,583 | 10,085,690 |
| shared-memory instructions | 13,296,944 | 7,602,176 | 7,602,176 | 12,582,912 |
| global-load sectors | 29,815,000 | 31,436,554 | 26,725,546 | 29,060,067 |
| active warps (%) | 47.890 | 24.730 | 24.650 | 23.990 |
| barrier stall (%) | 5.770 | 7.180 | 7.910 | 10.230 |
| long scoreboard stall (%) | 40.760 | 17.440 | 16.780 | 27.320 |
| MIO throttle stall (%) | 5.400 | 3.370 | 3.050 | 4.170 |
| registers/thread | 40.000 | 105.000 | 109.000 | 80.000 |
| shared bytes/CTA | 16416.000 | 41184.000 | 41184.000 | 16496.000 |

## Decision

Packet128 throughput/v0.6=1.187x, throughput/lane32=1.265x. Its dynamic instruction count is 0.948x v0.6 and 0.725x lane32.
The packet core closes the physical-codelet instruction gap and beats the fixed-clock v0.6 control. It is eligible for broader batch/length screening.
