# Cross-Operator v0.8 Comparison

Hardware: `v100-sm70`. Ratios above 1 mean v0.8 has lower kernel time.
The `status` column is part of the claim: compatibility-lowered points are not presented as native resident subgraphs.

| operator | precision | logN | batch | v0.6 ms | v0.8 search ms | search/v0.6 | model ms | model/v0.6 | model/search | library/external | search/library | model/library | implementation | status |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|:--|--:|--:|:--|:--|
| fft | fp32 | 10 | 1 | 0.008653 | 0.006605 | 1.310x | 0.006656 | 1.300x | 0.992x | cuFFT (0.004608 ms) | 0.698x | 0.692x | hybrid-dataflow-radix4 | native-resident-subgraph |
| fft | fp32 | 10 | 4 | 0.008755 | 0.006707 | 1.305x | 0.006707 | 1.305x | 1.000x | cuFFT (0.004608 ms) | 0.687x | 0.687x | hybrid-dataflow-radix4 | native-resident-subgraph |
| fft | fp32 | 10 | 16 | 0.009984 | 0.006758 | 1.477x | 0.006758 | 1.477x | 1.000x | cuFFT (0.004659 ms) | 0.689x | 0.689x | hybrid-dataflow-radix4 | native-resident-subgraph |
| fft | fp32 | 10 | 64 | 0.012698 | 0.007424 | 1.710x | 0.007424 | 1.710x | 1.000x | cuFFT (0.005069 ms) | 0.683x | 0.683x | hybrid-dataflow-radix4 | native-resident-subgraph |
| fft | fp32 | 12 | 1 | 0.013670 | 0.013670 | 1.000x | 0.017869 | 0.765x | 0.765x | cuFFT (0.006861 ms) | 0.502x | 0.384x | v08-inherited-hierarchical-radix4 | v08-inherited-v06 |
| fft | fp32 | 12 | 4 | 0.014643 | 0.014643 | 1.000x | 0.018125 | 0.808x | 0.808x | cuFFT (0.006810 ms) | 0.465x | 0.376x | v08-inherited-hierarchical-radix4 | v08-inherited-v06 |
| fft | fp32 | 12 | 16 | 0.018534 | 0.018074 | 1.025x | 0.018074 | 1.025x | 1.000x | cuFFT (0.006861 ms) | 0.380x | 0.380x | hybrid-dataflow-radix4 | native-resident-subgraph |
| fft | fp32 | 12 | 64 | 0.030515 | 0.021248 | 1.436x | 0.021299 | 1.433x | 0.998x | cuFFT (0.008090 ms) | 0.381x | 0.380x | hybrid-dataflow-radix4 | native-resident-subgraph |
| fwht | fp32 | 10 | 1 | 0.007629 | 0.004659 | 1.637x | 0.004659 | 1.637x | 1.000x | none-matched | n/a | n/a | hybrid-dataflow-radix4 | native-resident-subgraph |
| fwht | fp32 | 10 | 4 | 0.007731 | 0.004710 | 1.641x | 0.004659 | 1.659x | 1.011x | none-matched | n/a | n/a | hybrid-dataflow-radix4 | native-resident-subgraph |
| fwht | fp32 | 10 | 16 | 0.007578 | 0.004710 | 1.609x | 0.004710 | 1.609x | 1.000x | none-matched | n/a | n/a | hybrid-dataflow-radix4 | native-resident-subgraph |
| fwht | fp32 | 10 | 64 | 0.008448 | 0.004813 | 1.755x | 0.004813 | 1.755x | 1.000x | none-matched | n/a | n/a | hybrid-dataflow-radix4 | native-resident-subgraph |
| fwht | fp32 | 12 | 1 | 0.012032 | 0.010650 | 1.130x | 0.010650 | 1.130x | 1.000x | none-matched | n/a | n/a | hybrid-dataflow-radix4 | native-resident-subgraph |
| fwht | fp32 | 12 | 4 | 0.012493 | 0.010701 | 1.167x | 0.010701 | 1.167x | 1.000x | none-matched | n/a | n/a | hybrid-dataflow-radix4 | native-resident-subgraph |
| fwht | fp32 | 12 | 16 | 0.013312 | 0.010752 | 1.238x | 0.010803 | 1.232x | 0.995x | none-matched | n/a | n/a | hybrid-dataflow-radix4 | native-resident-subgraph |
| fwht | fp32 | 12 | 64 | 0.018739 | 0.010906 | 1.718x | 0.010854 | 1.726x | 1.005x | none-matched | n/a | n/a | hybrid-dataflow-radix4 | native-resident-subgraph |
| fwht | fp32 | 14 | 1 | 0.021555 | 0.021555 | 1.000x | 0.035379 | 0.609x | 0.609x | none-matched | n/a | n/a | v08-inherited-hierarchical-radix4 | v08-inherited-v06 |
| fwht | fp32 | 14 | 4 | 0.018278 | 0.018278 | 1.000x | 0.035482 | 0.515x | 0.515x | none-matched | n/a | n/a | v08-inherited-hierarchical-radix4 | v08-inherited-v06 |
| fwht | fp32 | 14 | 16 | 0.025293 | 0.025293 | 1.000x | 0.035533 | 0.712x | 0.712x | none-matched | n/a | n/a | v08-inherited-hierarchical-radix4 | v08-inherited-v06 |
| fwht | fp32 | 14 | 64 | 0.067942 | 0.050432 | 1.347x | 0.050278 | 1.351x | 1.003x | none-matched | n/a | n/a | hybrid-dataflow-radix4 | native-resident-subgraph |
| structured-2x2 | fp32 | 10 | 1 | 0.007885 | 0.005325 | 1.481x | 0.005325 | 1.481x | 1.000x | none-matched | n/a | n/a | hybrid-dataflow-radix4 | native-resident-subgraph |
| structured-2x2 | fp32 | 10 | 4 | 0.008090 | 0.005427 | 1.491x | 0.005376 | 1.505x | 1.009x | none-matched | n/a | n/a | hybrid-dataflow-radix4 | native-resident-subgraph |
| structured-2x2 | fp32 | 10 | 16 | 0.008090 | 0.005376 | 1.505x | 0.005427 | 1.491x | 0.991x | none-matched | n/a | n/a | hybrid-dataflow-radix4 | native-resident-subgraph |
| structured-2x2 | fp32 | 10 | 64 | 0.009011 | 0.005530 | 1.629x | 0.005530 | 1.629x | 1.000x | none-matched | n/a | n/a | hybrid-dataflow-radix4 | native-resident-subgraph |
| structured-2x2 | fp32 | 12 | 1 | 0.012544 | 0.009882 | 1.269x | 0.011469 | 1.094x | 0.862x | none-matched | n/a | n/a | hybrid-dataflow-radix4 | native-resident-subgraph |
| structured-2x2 | fp32 | 12 | 4 | 0.012800 | 0.011571 | 1.106x | 0.011520 | 1.111x | 1.004x | none-matched | n/a | n/a | hybrid-dataflow-radix4 | native-resident-subgraph |
| structured-2x2 | fp32 | 12 | 16 | 0.014029 | 0.011571 | 1.212x | 0.011622 | 1.207x | 0.996x | none-matched | n/a | n/a | hybrid-dataflow-radix4 | native-resident-subgraph |
| structured-2x2 | fp32 | 12 | 64 | 0.019661 | 0.011878 | 1.655x | 0.011776 | 1.670x | 1.009x | none-matched | n/a | n/a | hybrid-dataflow-radix4 | native-resident-subgraph |
| subset-zeta | uint32 | 10 | 1 | 0.007526 | 0.004506 | 1.670x | 0.004506 | 1.670x | 1.000x | none-matched | n/a | n/a | hybrid-dataflow-radix4 | native-resident-subgraph |
| subset-zeta | uint32 | 10 | 4 | 0.007578 | 0.004608 | 1.645x | 0.004608 | 1.645x | 1.000x | none-matched | n/a | n/a | hybrid-dataflow-radix4 | native-resident-subgraph |
| subset-zeta | uint32 | 10 | 16 | 0.008550 | 0.004557 | 1.876x | 0.004608 | 1.855x | 0.989x | none-matched | n/a | n/a | hybrid-dataflow-radix4 | native-resident-subgraph |
| subset-zeta | uint32 | 10 | 64 | 0.008346 | 0.004608 | 1.811x | 0.004608 | 1.811x | 1.000x | none-matched | n/a | n/a | hybrid-dataflow-radix4 | native-resident-subgraph |
| subset-zeta | uint32 | 12 | 1 | 0.012595 | 0.009984 | 1.262x | 0.009984 | 1.262x | 1.000x | none-matched | n/a | n/a | hybrid-dataflow-radix4 | native-resident-subgraph |
| subset-zeta | uint32 | 12 | 4 | 0.012032 | 0.010035 | 1.199x | 0.010086 | 1.193x | 0.995x | none-matched | n/a | n/a | hybrid-dataflow-radix4 | native-resident-subgraph |
| subset-zeta | uint32 | 12 | 16 | 0.013261 | 0.010035 | 1.321x | 0.010138 | 1.308x | 0.990x | none-matched | n/a | n/a | hybrid-dataflow-radix4 | native-resident-subgraph |
| subset-zeta | uint32 | 12 | 64 | 0.018483 | 0.010240 | 1.805x | 0.010291 | 1.796x | 0.995x | none-matched | n/a | n/a | hybrid-dataflow-radix4 | native-resident-subgraph |
| subset-zeta | uint32 | 14 | 1 | 0.017459 | 0.017459 | 1.000x | 0.032358 | 0.540x | 0.540x | none-matched | n/a | n/a | v08-inherited-hierarchical-radix4 | v08-inherited-v06 |
| subset-zeta | uint32 | 14 | 4 | 0.018176 | 0.018176 | 1.000x | 0.032563 | 0.558x | 0.558x | none-matched | n/a | n/a | v08-inherited-hierarchical-radix4 | v08-inherited-v06 |
| subset-zeta | uint32 | 14 | 16 | 0.024781 | 0.024781 | 1.000x | 0.032563 | 0.761x | 0.761x | none-matched | n/a | n/a | v08-inherited-hierarchical-radix4 | v08-inherited-v06 |
| subset-zeta | uint32 | 14 | 64 | 0.067686 | 0.047616 | 1.421x | 0.047514 | 1.425x | 1.002x | none-matched | n/a | n/a | hybrid-dataflow-radix4 | native-resident-subgraph |
| superset-zeta | uint32 | 10 | 1 | 0.008038 | 0.004710 | 1.707x | 0.004659 | 1.725x | 1.011x | none-matched | n/a | n/a | hybrid-dataflow-radix4 | native-resident-subgraph |
| superset-zeta | uint32 | 10 | 4 | 0.007680 | 0.004659 | 1.648x | 0.004710 | 1.631x | 0.989x | none-matched | n/a | n/a | hybrid-dataflow-radix4 | native-resident-subgraph |
| superset-zeta | uint32 | 10 | 16 | 0.007680 | 0.004659 | 1.648x | 0.004659 | 1.648x | 1.000x | none-matched | n/a | n/a | hybrid-dataflow-radix4 | native-resident-subgraph |
| superset-zeta | uint32 | 10 | 64 | 0.008397 | 0.004813 | 1.745x | 0.004813 | 1.745x | 1.000x | none-matched | n/a | n/a | hybrid-dataflow-radix4 | native-resident-subgraph |
| superset-zeta | uint32 | 12 | 1 | 0.013312 | 0.010445 | 1.274x | 0.010496 | 1.268x | 0.995x | none-matched | n/a | n/a | hybrid-dataflow-radix4 | native-resident-subgraph |
| superset-zeta | uint32 | 12 | 4 | 0.012288 | 0.010547 | 1.165x | 0.010445 | 1.176x | 1.010x | none-matched | n/a | n/a | hybrid-dataflow-radix4 | native-resident-subgraph |
| superset-zeta | uint32 | 12 | 16 | 0.013158 | 0.010598 | 1.242x | 0.010598 | 1.242x | 1.000x | none-matched | n/a | n/a | hybrid-dataflow-radix4 | native-resident-subgraph |
| superset-zeta | uint32 | 12 | 64 | 0.018483 | 0.010752 | 1.719x | 0.010752 | 1.719x | 1.000x | none-matched | n/a | n/a | hybrid-dataflow-radix4 | native-resident-subgraph |
| superset-zeta | uint32 | 14 | 1 | 0.016947 | 0.016947 | 1.000x | 0.034918 | 0.485x | 0.485x | none-matched | n/a | n/a | v08-inherited-hierarchical-radix4 | v08-inherited-v06 |
| superset-zeta | uint32 | 14 | 4 | 0.018278 | 0.018278 | 1.000x | 0.035072 | 0.521x | 0.521x | none-matched | n/a | n/a | v08-inherited-hierarchical-radix4 | v08-inherited-v06 |
| superset-zeta | uint32 | 14 | 16 | 0.025190 | 0.025190 | 1.000x | 0.035123 | 0.717x | 0.717x | none-matched | n/a | n/a | v08-inherited-hierarchical-radix4 | v08-inherited-v06 |
| superset-zeta | uint32 | 14 | 64 | 0.067226 | 0.049971 | 1.345x | 0.050074 | 1.343x | 0.998x | none-matched | n/a | n/a | hybrid-dataflow-radix4 | native-resident-subgraph |
| xor-zeta | uint32 | 10 | 1 | 0.007731 | 0.004454 | 1.736x | 0.004454 | 1.736x | 1.000x | none-matched | n/a | n/a | hybrid-dataflow-radix4 | native-resident-subgraph |
| xor-zeta | uint32 | 10 | 4 | 0.007782 | 0.004506 | 1.727x | 0.004506 | 1.727x | 1.000x | none-matched | n/a | n/a | hybrid-dataflow-radix4 | native-resident-subgraph |
| xor-zeta | uint32 | 10 | 16 | 0.007629 | 0.004506 | 1.693x | 0.004506 | 1.693x | 1.000x | none-matched | n/a | n/a | hybrid-dataflow-radix4 | native-resident-subgraph |
| xor-zeta | uint32 | 10 | 64 | 0.008448 | 0.004608 | 1.833x | 0.004659 | 1.813x | 0.989x | none-matched | n/a | n/a | hybrid-dataflow-radix4 | native-resident-subgraph |
| xor-zeta | uint32 | 12 | 1 | 0.013261 | 0.009984 | 1.328x | 0.009984 | 1.328x | 1.000x | none-matched | n/a | n/a | hybrid-dataflow-radix4 | native-resident-subgraph |
| xor-zeta | uint32 | 12 | 4 | 0.011981 | 0.009984 | 1.200x | 0.010086 | 1.188x | 0.990x | none-matched | n/a | n/a | hybrid-dataflow-radix4 | native-resident-subgraph |
| xor-zeta | uint32 | 12 | 16 | 0.013261 | 0.010189 | 1.302x | 0.010138 | 1.308x | 1.005x | none-matched | n/a | n/a | hybrid-dataflow-radix4 | native-resident-subgraph |
| xor-zeta | uint32 | 12 | 64 | 0.018330 | 0.010240 | 1.790x | 0.010291 | 1.781x | 0.995x | none-matched | n/a | n/a | hybrid-dataflow-radix4 | native-resident-subgraph |
| xor-zeta | uint32 | 14 | 1 | 0.017562 | 0.017562 | 1.000x | 0.032307 | 0.544x | 0.544x | none-matched | n/a | n/a | v08-inherited-hierarchical-radix4 | v08-inherited-v06 |
| xor-zeta | uint32 | 14 | 4 | 0.018176 | 0.018176 | 1.000x | 0.032666 | 0.556x | 0.556x | none-matched | n/a | n/a | v08-inherited-hierarchical-radix4 | v08-inherited-v06 |
| xor-zeta | uint32 | 14 | 16 | 0.024883 | 0.024883 | 1.000x | 0.032512 | 0.765x | 0.765x | none-matched | n/a | n/a | v08-inherited-hierarchical-radix4 | v08-inherited-v06 |
| xor-zeta | uint32 | 14 | 64 | 0.067635 | 0.047514 | 1.423x | 0.042752 | 1.582x | 1.111x | none-matched | n/a | n/a | hybrid-dataflow-radix4 | native-resident-subgraph |

## Interpretation

FFT and NTT can use compiled specialized physical cores in this matrix. FWHT, subset/superset zeta, legacy xor-zeta, and structured-2x2 are covered by the same mapping/profile protocol, while their v0.8 rows remain explicitly marked compatibility-lowered until a native resident operator-templated chain is compiled.

The model columns are measured executions of the runtime-selected point; their `SelectionInfo` prediction is retained in the raw benchmark CSV. No external specialized baseline is fabricated for zeta or structured-2x2; the matching Dao-AILab FHT and GPU-NTT rows are joined by the separate external-baseline protocol when available.

## Aggregate View

Geometric means summarize ratios across batch/length cells; they do not replace the per-cell table.

| operator | cells | search/v0.6 | model/v0.6 | model/search | search/library | model/library | search wins vs v0.6 |
|:--|--:|--:|--:|--:|--:|--:|--:|
| fft | 8 | 1.260x | 1.185x | 0.940x | 0.543x | 0.511x | 6/8 |
| fwht | 12 | 1.323x | 1.169x | 0.884x | n/a | n/a | 9/12 |
| structured-2x2 | 8 | 1.406x | 1.381x | 0.983x | n/a | n/a | 8/8 |
| subset-zeta | 12 | 1.380x | 1.218x | 0.882x | n/a | n/a | 9/12 |
| superset-zeta | 12 | 1.343x | 1.165x | 0.868x | n/a | n/a | 9/12 |
| xor-zeta | 12 | 1.384x | 1.233x | 0.891x | n/a | n/a | 9/12 |
