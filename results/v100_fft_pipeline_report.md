# Generated V100 FFT Pipeline Search

| logN | Batch | Split | Threads | EPT | Twiddle | Median ms | cuFFT ms | Throughput ratio |
|--:|--:|:--|:--|:--|:--|--:|--:|--:|
| 18 | 16 | 9+9 | 256+256 | 8+8 | recurrence | 0.205722 | 0.216146 | 1.051x |
| 18 | 2 | 9+9 | 256+256 | 8+8 | recurrence | 0.026051 | 0.028099 | 1.079x |
| 18 | 64 | 9+9 | 256+256 | 8+8 | recurrence | 0.842875 | 0.717599 | 0.851x |
| 20 | 16 | 10+10 | 256+128 | 16+16 | recurrence | 0.883835 | 0.720855 | 0.816x |
| 20 | 2 | 10+10 | 512+512 | 8+8 | recurrence | 0.128799 | 0.123556 | 0.959x |
| 20 | 8 | 10+10 | 256+128 | 16+16 | recurrence | 0.450109 | 0.382157 | 0.849x |
