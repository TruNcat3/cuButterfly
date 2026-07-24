# V100 Counter Attribution

## Result

The targeted NCU experiment covers 22 FFT/FWHT implementation-shape pairs and
37 kernels. CUDA-event measurements remain the performance authority; NCU
replay is used only to attribute grid coverage, residency, data service, and
stall behavior. The complete counter table and generated analysis are in
[`attribution.md`](../results/ncu_scaling_crossovers/attribution.md).

## Architecture Findings

| Boundary | Counter evidence | Architecture interpretation |
|:--|:--|:--|
| FFT `logN=14`, batch 16 | direct has 0.20 waves/SM and one resident 1024-thread CTA/SM | insufficient `Ub` grid coverage; the local core is not the limiting factor |
| FFT `logN=18`, batch 64 | online has 1.60x warp instructions, 9.50x bank conflicts, and lower DRAM utilization than cuFFT | extra exchange and synchronization reduce useful work per warp |
| FFT `logN=20`, batch 16 | equal aggregate waves, but online has 2.00x warp instructions and 9.42x bank conflicts | occupancy cannot close the gap; the prefix/core/layout composition must improve |
| FWHT `logN=15`, batch 4 to 16 | warp-register has only 0.05 to 0.20 waves/SM but 0.07x the online warp instructions at batch 16 | low batch favors `Ud` decomposition; sufficient `Ub` favors register-resident `Td` reuse |

For FFT `logN=20`, the online first pass consumes 61.8% of profiled time and
reaches only 56.6% of peak DRAM, while its second pass reaches 85.3%. This
identifies prefix/suffix imbalance as the next concrete FFT optimization target.

## Methodological Consequence

The selector must model three quantities independently:

1. **grid coverage**: batch and spatial data unfolding translated to waves/SM;
2. **residency**: register/shared allocations translated to resident CTAs;
3. **useful work per resident warp**: instructions, exchange conflicts, barrier
   stalls, and scoreboard stalls.

Maximizing occupancy alone is incorrect. Both long online FFT mappings expose
at least as many active warps as cuFFT while reaching a lower saturated
throughput ceiling.

Small cases may report zero DRAM traffic because the collection uses
`--cache-control none`, NCU replay, and working sets that fit in V100's L2. The
report therefore does not interpret those zeros as zero algorithmic traffic.
