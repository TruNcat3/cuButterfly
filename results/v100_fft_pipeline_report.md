# Generated V100 FFT Pipeline Search

| logN | Batch | Logical stages | Physical groups | Threads/EPT | Residency | Median ms | cuFFT ms | Throughput ratio |
|--:|--:|:--|:--|:--|:--|--:|--:|--:|
| 18 | 2 | 3x6x3x6 | 9x9 | 256x256 / 8x8 | fusedxglobal-scratchxfused | 0.026010 | 0.028099 | 1.080x |
| 18 | 16 | 3x6x3x6 | 9x9 | 256x256 / 8x8 | fusedxglobal-scratchxfused | 0.205496 | 0.216146 | 1.052x |
| 18 | 64 | 9x3x6 | 9x9 | 256x256 / 8x8 | global-scratchxfused | 0.836690 | 0.717599 | 0.858x |
| 20 | 2 | 10x3x3x4 | 10x10 | 512x512 / 8x8 | global-scratchxfusedxfused | 0.128573 | 0.123556 | 0.961x |
| 20 | 8 | 10x3x7 | 10x10 | 256x128 / 16x16 | global-scratchxfused | 0.445338 | 0.382157 | 0.858x |
| 20 | 16 | 3x3x4x10 | 10x10 | 256x128 / 16x16 | fusedxfusedxglobal-scratch | 0.867799 | 0.720855 | 0.831x |
