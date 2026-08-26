# APPT Writer-Final NCU Attribution

V100 base-clock kernel replay. Ratios use the corrected grouped producer with its independently selected mapping.

| bits | batch | variant | time us | load/grouped | store/grouped | DRAM read MiB | DRAM write MiB | L2 hit | warp-inst/grouped | integer/grouped | active warps | barrier | scoreboard | MIO | regs | waves/SM |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 1 | `v06` | 143.8 | 0.517x | 1.323x | 13.4 | 8.7 | 75.4% | 1.196x | 1.069x | 41.7% | 4.2% | 20.8% | 3.6% | 40 | 0.80 |
| 32 | 1 | `grouped` | 198.1 | 1.000x | 1.000x | 23.5 | 13.1 | 86.6% | 1.000x | 1.000x | 27.7% | 23.5% | 36.6% | 3.7% | 48 | 1.00 |
| 32 | 1 | `writer_final` | 176.7 | 0.782x | 0.988x | 21.7 | 13.1 | 83.3% | 1.004x | 0.988x | 29.5% | 25.4% | 34.4% | 4.0% | 80 | 1.00 |
| 32 | 4 | `v06` | 342.4 | 0.512x | 1.319x | 67.0 | 36.5 | 68.4% | 1.094x | 1.054x | 46.9% | 5.0% | 33.3% | 5.1% | 40 | 0.80 |
| 32 | 4 | `grouped` | 624.6 | 1.000x | 1.000x | 116.8 | 49.2 | 81.8% | 1.000x | 1.000x | 32.4% | 19.9% | 38.4% | 6.6% | 48 | 1.00 |
| 32 | 4 | `writer_final` | 506.1 | 0.784x | 0.963x | 108.1 | 49.1 | 78.6% | 1.001x | 0.986x | 33.5% | 17.6% | 39.6% | 6.7% | 80 | 1.00 |
| 32 | 16 | `v06` | 1077.2 | 0.512x | 1.333x | 262.7 | 141.1 | 66.5% | 1.045x | 1.053x | 48.6% | 5.7% | 40.6% | 5.7% | 40 | 0.80 |
| 32 | 16 | `grouped` | 2158.8 | 1.000x | 1.000x | 515.8 | 192.4 | 80.7% | 1.000x | 1.000x | 34.9% | 24.3% | 37.9% | 8.1% | 48 | 1.00 |
| 32 | 16 | `writer_final` | 1838.1 | 0.766x | 0.991x | 481.8 | 193.4 | 75.2% | 0.999x | 0.985x | 34.1% | 15.8% | 41.8% | 8.3% | 80 | 1.00 |
| 64 | 1 | `v06` | 232.2 | 0.724x | 0.674x | 30.9 | 16.7 | 52.6% | 1.170x | 1.042x | 23.4% | 2.9% | 21.3% | 0.4% | 48 | 1.00 |
| 64 | 1 | `grouped` | 318.5 | 1.000x | 1.000x | 52.0 | 24.6 | 73.7% | 1.000x | 1.000x | 17.4% | 15.0% | 43.8% | 0.6% | 128 | 1.00 |
| 64 | 1 | `writer_final` | 319.7 | 0.836x | 1.000x | 55.1 | 25.4 | 66.8% | 0.982x | 0.983x | 18.9% | 15.4% | 42.5% | 0.8% | 127 | 1.00 |
| 64 | 4 | `v06` | 612.1 | 0.719x | 0.667x | 127.8 | 65.2 | 49.1% | 1.073x | 1.036x | 24.3% | 3.5% | 32.4% | 0.6% | 48 | 1.00 |
| 64 | 4 | `grouped` | 969.3 | 1.000x | 1.000x | 229.3 | 96.9 | 70.7% | 1.000x | 1.000x | 20.7% | 9.7% | 48.7% | 2.0% | 128 | 1.00 |
| 64 | 4 | `writer_final` | 948.2 | 0.835x | 1.019x | 230.8 | 98.1 | 64.6% | 1.012x | 0.990x | 19.9% | 8.9% | 47.2% | 1.6% | 127 | 1.00 |
| 64 | 16 | `v06` | 2000.7 | 0.721x | 0.667x | 514.4 | 257.7 | 49.0% | 1.025x | 1.034x | 24.8% | 3.9% | 37.5% | 0.6% | 48 | 1.00 |
| 64 | 16 | `grouped` | 3359.7 | 1.000x | 1.000x | 919.3 | 384.9 | 70.7% | 1.000x | 1.000x | 21.5% | 7.4% | 50.2% | 2.3% | 128 | 1.00 |
| 64 | 16 | `writer_final` | 3297.2 | 0.827x | 1.000x | 977.8 | 385.7 | 62.1% | 0.989x | 0.986x | 18.4% | 7.7% | 48.6% | 1.8% | 127 | 1.00 |

Writer-final reduces global-load sectors versus corrected grouped by 21.6%--23.4% for uint32 and 16.4%--17.3% for uint64. Warp and integer instruction counts change by at most 1.8%; uint64 remains at the same two-CTA/SM shared-memory limit and essentially the same 127/128 registers per thread. The numeric crossover is therefore not caused by extra butterfly work, register spill, or lower theoretical CTA residency.

The saved sectors do not translate uniformly into DRAM traffic. Uint32 DRAM reads fall by 6.6%--7.6% despite a 3.2--5.4 point L2-hit loss, which is sufficient for the measured event gain. Uint64 L2 hit rate falls by 6.1--8.6 points and DRAM reads rise by 0.7%--6.4%; this cancels the sector reduction and explains its neutral event result. The transform-major writer alternates pre-final low/high streams separated by `N/2` and does not reuse stage-19 coefficients across batch transforms.

Relative to v0.6, writer-final still issues about 1.50x/1.15x load sectors for uint32/uint64. Uint64 also retains exactly 1.50x store sectors because APPT materializes producer state, static tail publication, and natural output, whereas the `10+10` v0.6 path has two coalesced value stores. The next controlled mapping should traverse the batch data-time coordinate inside a fixed spatial subgraph so homogeneous transforms reuse coefficients and cache lines; another arithmetic-only radix change is not indicated.

CUDA-event timing in `results/appt_writer_final/confirmed/analysis.md` remains the ranking authority. This report determines whether stage-19 writer fusion reduces requests and whether writer occupancy or arithmetic latency offsets that reduction.
