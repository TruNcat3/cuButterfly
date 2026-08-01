# Generated V100 FFT Pipeline Search

| logN | Batch | Logical stages | Physical groups | Threads/EPT | Residency | Median ms | cuFFT ms | Throughput ratio |
|--:|--:|:--|:--|:--|:--|--:|--:|--:|
| 18 | 2 | 3x3x3x9 | 9x9 | 256x256 / 8x8 | fusedxfusedxglobal-scratch | 0.025866 | 0.028058 | 1.085x |
| 18 | 16 | 3x6x3x6 | 9x9 | 256x256 / 8x8 | fusedxglobal-scratchxfused | 0.195441 | 0.211108 | 1.080x |
| 18 | 64 | 8x10 | 8x10 | 128x128 / 16x16 | global-scratch | 0.767201 | 0.706273 | 0.921x |
| 20 | 2 | 10x10 | 10x10 | 512x512 / 8x8 | global-scratch | 0.119869 | 0.123453 | 1.030x |
| 20 | 8 | 10x3x3x4 | 10x10 | 256x128 / 16x16 | global-scratchxfusedxfused | 0.397332 | 0.382300 | 0.962x |
| 20 | 16 | 3x3x4x10 | 10x10 | 256x128 / 16x16 | fusedxfusedxglobal-scratch | 0.764334 | 0.729416 | 0.954x |
