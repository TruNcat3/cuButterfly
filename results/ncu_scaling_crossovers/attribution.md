# V100 Scaling-Crossover Counter Attribution

NCU replay time is used only for attribution; CUDA-event time remains the performance authority.

| Workload | Batch | Impl | Kernels | Waves/SM | Active warps | DRAM peak | B/point | Barrier stall | Scoreboard stall | Reg/thread | Shared B |
|:--|--:|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| fft logN=14 | 1024 | cufft | 1 | 12.80 | 46.9% | 68.0% | 16.0 | 6.0% | 15.1% | 61 | 67584 |
| fft logN=14 | 16 | cufft | 2 | 0.54 | 12.4% | 15.9% | 15.2 | 0.4% | 18.8% | 64 | 32768 |
| fft logN=14 | 256 | cufft | 1 | 3.20 | 46.5% | 54.0% | 16.0 | 5.1% | 15.9% | 61 | 67584 |
| fft logN=14 | 1024 | direct | 1 | 12.80 | 47.4% | 72.2% | 16.0 | 7.0% | 6.5% | 63 | 65536 |
| fft logN=14 | 16 | direct | 1 | 0.20 | 48.6% | 0.0% | 0.0 | 8.3% | 3.5% | 63 | 65536 |
| fft logN=14 | 256 | direct | 1 | 3.20 | 47.3% | 57.4% | 15.9 | 6.2% | 7.5% | 63 | 65536 |
| fft logN=18 | 16 | cufft | 2 | 12.80 | 47.0% | 65.2% | 32.1 | 8.3% | 26.6% | 39 | 32768 |
| fft logN=18 | 2 | cufft | 2 | 1.60 | 48.1% | 41.2% | 27.9 | 3.5% | 31.6% | 39 | 32768 |
| fft logN=18 | 64 | cufft | 2 | 51.20 | 46.9% | 79.3% | 32.0 | 9.5% | 24.9% | 39 | 32768 |
| fft logN=18 | 16 | online | 2 | 10.24 | 58.7% | 68.3% | 32.4 | 12.3% | 33.3% | 48 | 16384 |
| fft logN=18 | 2 | online | 2 | 1.28 | 37.7% | 46.9% | 33.4 | 7.1% | 38.0% | 48 | 16384 |
| fft logN=18 | 64 | online | 2 | 40.96 | 60.6% | 70.1% | 32.2 | 13.4% | 30.8% | 48 | 16384 |
| fft logN=20 | 16 | cufft | 2 | 51.20 | 24.1% | 77.5% | 32.1 | 4.7% | 21.4% | 64 | 65536 |
| fft logN=20 | 2 | cufft | 2 | 6.40 | 24.2% | 52.2% | 32.7 | 2.8% | 18.2% | 64 | 65536 |
| fft logN=20 | 8 | cufft | 2 | 25.60 | 24.1% | 70.5% | 32.2 | 4.3% | 19.8% | 64 | 65536 |
| fft logN=20 | 16 | online | 2 | 51.20 | 48.6% | 67.6% | 33.2 | 11.9% | 32.8% | 52 | 32768 |
| fft logN=20 | 2 | online | 2 | 6.40 | 47.0% | 56.9% | 34.0 | 11.4% | 35.1% | 52 | 32768 |
| fft logN=20 | 8 | online | 2 | 25.60 | 48.3% | 66.0% | 33.3 | 11.8% | 33.3% | 52 | 32768 |
| fwht logN=15 | 16 | online | 2 | 9.60 | 80.7% | 4.8% | 4.6 | 28.0% | 11.8% | 25 | 1024 |
| fwht logN=15 | 4 | online | 2 | 2.40 | 56.6% | 0.0% | 0.0 | 17.2% | 4.3% | 25 | 1024 |
| fwht logN=15 | 16 | warp | 1 | 0.20 | 12.5% | 5.2% | 3.8 | 1.5% | 4.8% | 255 | 32768 |
| fwht logN=15 | 4 | warp | 1 | 0.05 | 12.5% | 0.0% | 0.0 | 1.3% | 6.4% | 255 | 32768 |

## Measured Trends

- FFT logN=14 direct: waves/SM change 64.00x from batch 16 to 1024; active warps 48.6% -> 47.4%, DRAM peak 0.0% -> 72.2%, the low-batch DRAM count is cache-resident/replay-sensitive, barrier stalls 8.3% -> 7.0%, and long-scoreboard stalls 3.5% -> 6.5%.
- FFT logN=18 online: waves/SM change 32.00x from batch 2 to 64; active warps 37.7% -> 60.6%, DRAM peak 46.9% -> 70.1%, bytes/point changes 0.96x, barrier stalls 7.1% -> 13.4%, and long-scoreboard stalls 38.0% -> 30.8%.
- FFT logN=20 online: waves/SM change 8.00x from batch 2 to 16; active warps 47.0% -> 48.6%, DRAM peak 56.9% -> 67.6%, bytes/point changes 0.98x, barrier stalls 11.4% -> 11.9%, and long-scoreboard stalls 35.1% -> 32.8%.
- FWHT logN=15 online: waves/SM change 4.00x from batch 4 to 16; active warps 56.6% -> 80.7%, DRAM peak 0.0% -> 4.8%, the low-batch DRAM count is cache-resident/replay-sensitive, barrier stalls 17.2% -> 28.0%, and long-scoreboard stalls 4.3% -> 11.8%.
- FWHT logN=15 warp: waves/SM change 4.00x from batch 4 to 16; active warps 12.5% -> 12.5%, DRAM peak 0.0% -> 5.2%, the low-batch DRAM count is cache-resident/replay-sensitive, barrier stalls 1.3% -> 1.5%, and long-scoreboard stalls 6.4% -> 4.8%.

Counts reported as zero for the smallest cases are not zero algorithmic traffic. `--cache-control none`, NCU replay, and a working set that fits in the 6 MiB L2 make those DRAM counters unsuitable for traffic ratios.

## Architecture Attribution

### FFT logN=14: grid concurrency threshold

The direct unit launches one 1024-thread CTA per transform and is limited to one resident CTA per SM by its register and 65536-byte shared allocations. At batch 16 it exposes only 0.20 waves/SM, whereas cuFFT exposes 0.54 waves/SM across two kernels. The direct grid therefore leaves most of the 80 SMs without work even though active CTAs individually report 48.6% active warps. At batch 1024 it reaches 12.8 waves/SM and the grid-level deficit disappears. This attributes the low-batch gap to `Ub`/grid coverage, not to the local cuFFTDx arithmetic core.

### FFT logN=18: useful-work ceiling

At batch 64, online composition has 60.6% active warps versus cuFFT's 46.9%, but executes 1.60x as many warp instructions and 9.50x as many shared-memory bank conflicts. Its barrier and long-scoreboard stalls are 13.4% and 30.8%, and it reaches only 70.1% of peak DRAM versus cuFFT's 79.3%. The early saturation is therefore caused by less useful work per active warp and exchange/synchronization overhead, not a lack of resident warps.

### FFT logN=20: prefix/suffix imbalance

At batch 16, online and cuFFT both expose 51.2 aggregate waves/SM. Online nevertheless executes 2.00x the warp instructions, incurs 9.42x the shared bank conflicts, and has 11.9% barrier plus 32.8% scoreboard stalls. The first pass consumes 61.8% of profiled time and reaches only 56.6% DRAM, while the second reaches 85.3%. The remaining gap is a prefix-core/layout imbalance and synchronization problem; increasing occupancy alone cannot close it.

### FWHT logN=15: mapping crossover

The warp-register unit uses 255 registers/thread and 32768 shared bytes, limiting it to one CTA/SM. It launches only 0.05 waves/SM at batch 4 and 0.20 at batch 16, with nearly fixed NCU time. Online decomposition exposes 2.40 and 9.60 waves/SM, so it wins when four transforms cannot cover the GPU. By batch 16, the register unit's single-kernel path executes only 0.07x the online warp instructions and avoids its shared exchanges and 28.0% barrier stalls. The crossover is the expected tradeoff between `Ud` spatial decomposition at low batch and `Td` register residence once `Ub` supplies enough independent transforms.

## Methodological Result

The counters support separating three quantities in the selector: grid coverage (`Ub` and waves/SM), residency (register/shared limits), and useful work per resident warp (instruction, exchange, and stall costs). Occupancy is not a sufficient objective: both long online FFTs expose at least as many active warps as cuFFT while delivering a lower saturated ceiling.
