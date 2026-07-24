# Generated V100 FFT Pipeline Search

| logN | Batch | Split | Threads | EPT | Twiddle | Median ms | cuFFT ms | Throughput ratio |
|--:|--:|:--|:--|:--|:--|--:|--:|--:|
| 18 | 16 | 9+9 | 256+256 | 8+8 | recurrence | 0.205455 | 0.211681 | 1.030x |
| 18 | 2 | 9+9 | 256+256 | 8+8 | recurrence | 0.025969 | 0.028078 | 1.081x |
| 18 | 64 | 9+9 | 256+256 | 8+8 | recurrence | 0.839352 | 0.706314 | 0.841x |
| 20 | 16 | 10+10 | 256+128 | 16+16 | recurrence | 0.869458 | 0.720978 | 0.829x |
| 20 | 2 | 10+10 | 512+512 | 8+8 | recurrence | 0.128573 | 0.123269 | 0.959x |
| 20 | 8 | 10+10 | 256+128 | 16+16 | recurrence | 0.448369 | 0.381727 | 0.851x |
