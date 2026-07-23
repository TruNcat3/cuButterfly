# logN=20 NCU gap attribution

Configuration: Tesla V100-SXM2-16GB, modulus `576460756061519873`, batch 4,
1000 warmups, steady caches, NCU kernel replay. NCU timing is used only for
counter attribution; CUDA-event medians remain the performance headline.

## Transform totals

| Implementation | Kernels | NCU time (us) | DRAM read (MiB) | DRAM write (MiB) | Integer thread instructions |
|:--|--:|--:|--:|--:|--:|
| cuNTT fused | 2 | 473.600 | 128.408 | 64.408 | 1,923,094,528 |
| GPU-NTT Merge | 3 | 441.472 | 118.778 | 95.976 | 2,877,292,544 |

cuNTT is 7.28% slower under NCU, consistent with the 8.30% paired CUDA-event
gap. GPU-NTT executes about 49.6% more integer thread instructions and writes
about 49% more data, so neither arithmetic count nor total full-array passes
explains its advantage.

## Phase attribution

| Phase | Time (us) | DRAM read (MiB) | L2 hit | Long scoreboard | Registers | Active warps |
|:--|--:|--:|--:|--:|--:|--:|
| cuNTT fused first pass | 227.456 | 32.030 | 89.42% | 19.98% | 32 | 94.17% |
| cuNTT fused second pass | 246.144 | 96.378 | 43.11% | 42.12% | 32 | 94.20% |
| GPU-NTT kernels 0+1 | 250.528 | 64.042 | - | - | 48 | about 60% |
| GPU-NTT kernel 2 | 190.944 | 54.736 | 43.83% | 24.68% | 48 | 60.25% |

cuNTT's first pass is 23.07 us faster than GPU-NTT's first two kernels
combined. Its second pass is 55.20 us slower than GPU-NTT's final kernel,
creating the entire net 32.13 us deficit.

The comparable final phases have nearly identical L2 hit rates, but cuNTT
reads 41.64 MiB more and spends 17.44 percentage points more issue cycles on
long-scoreboard stalls. The observed 96.38 MiB is consistent with about 32 MiB
of input data plus 64 MiB of fused root/Shoup traffic across four transforms.
This makes the expanded per-`k2` coset-twiddle representation the primary
`logN=20` bottleneck.

GPU-NTT indexes a conventional stage table as
`m + (omega_address >> t_2)` and reuses it across blocks. It does not materialize
a separate root/Shoup sequence for every matrix row. Its third kernel therefore
accepts another global data pass but reads only about 22.7 MiB beyond the
32 MiB input stream, whereas cuNTT's fused second pass reads about 64 MiB of
roots beyond its input stream.

## Barrett control

The single-table fused Barrett second pass reduces DRAM reads from 96.38 to
64.09 MiB and long-scoreboard stalls from 42.12% to 22.47%. However, it raises
integer thread instructions from 962.6M to 1,457.5M, registers from 32 to 40,
and lowers active warps from 94.20% to 71.72%. Its second pass therefore grows
from 246.14 to 272.58 us. Compacting the table is useful, but a full Barrett
substitution is too expensive on V100.

## Next implementation target

Keep the 32-register Shoup butterfly path, but replace the fully expanded
`N2 * (N1 - 1)` fused table with a factored or generated representation. The
design target is to approach GPU-NTT's final-phase root traffic without adding
a second modular multiplication per butterfly or extending wide-reduction
temporaries across the local NTT. A useful next control is a hardware-selected
three-kernel compact-stage mode: it establishes whether one extra coalesced data
pass is cheaper on V100 than streaming the expanded fused table, without
changing the architecture-level spatial/temporal decomposition.
