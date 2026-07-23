# CTA DFT8 Space-Time Mapping

## Mapping

The generated FFT point separates the processing unit from its GPU mapping:

```text
processing unit: one FP32 register DFT8
spatial factor Us: tile_threads in {32, 64, 128, 256}
temporal factor Ts: three fused butterfly levels per codelet
CTA workset: 8 * Us complex points
transforms per CTA: 8 * Us / N
shared-memory capacity: 64 * Us bytes
```

Natural-order global loads are coalesced. Threads read bit-reversed locations
from shared memory into registers, synchronize before an aliasing write when
`N>8`, and repeatedly compose three-stage codelets. One or two residual stages
use shared-memory butterflies. Thus `Us` changes the spatial batching and
resource footprint, while the DFT8 arithmetic remains unchanged.

## V100 Sweep

The table uses about `2^22` FP32 complex points, 20 warmups, 100 repetitions,
five CTA trials, and the median. The scalar and cuFFT controls use the same
shape from `results/fft_cta_composed_v100_raw.csv`; the final DFT8 row is in
`results/fft_cta_logN3_final_v100_raw.csv`.

| logN | selected Us | CTA ms | best shared radix-8 ms | scalar / CTA | cuFFT ms | CTA / cuFFT latency |
|---:|---:|---:|---:|---:|---:|---:|
| 3 | 32 | 0.093420 | 0.743189 | 7.96x | 0.086323 | 1.082x |
| 4 | 64 | 0.106004 | 0.373709 | 3.53x | 0.084306 | 1.257x |
| 5 | 64 | 0.103352 | 0.189655 | 1.84x | 0.084716 | 1.220x |
| 6 | 32 | 0.095222 | 0.097454 | 1.02x | 0.084101 | 1.132x |
| 7 | 64 | 0.098621 | 0.088545 | 0.90x | 0.084521 | 1.167x |
| 8 | 64 | 0.101714 | 0.097004 | 0.95x | 0.083784 | 1.214x |
| 9 | 64 | 0.101693 | 0.094935 | 0.93x | 0.084767 | 1.200x |
| 10 | 128 | 0.110090 | 0.100168 | 0.91x | 0.085709 | 1.284x |

These results establish a crossover rather than universal dominance. Spatially
batched register codelets are effective through `N=64`; from `N=128`, the
existing shared radix-8 schedule is 5-11% faster because it needs fewer layout
synchronizations. cuFFT remains 8-28% lower latency. The selected `Us` changes
with workset capacity and length, directly supporting hardware-dependent
mapping rather than a fixed tile claim.

## Layout Experiment

Writing natural global loads directly to bit-reversed shared addresses removes
one barrier but creates bank conflicts. `results/fft_cta_space_time_reorder_v100_raw.csv`
shows that this is about 10% slower at the longer local FFT sizes. Online
reordering must therefore be modeled by its destination-memory transaction
pattern, not only by permutation arithmetic cost.

## Profiling

Profile the selected CTA points, scalar controls, and cuFFT at representative
lengths with:

```bash
sudo -E ./scripts/profile_fft_cta_mapping_ncu.sh
sudo chown -R "$USER:$USER" results/ncu_fft_cta_mapping
```

The resulting counters localize the remaining gap into barrier stalls, shared
transactions, occupancy, or global-memory behavior before this local unit is
used as the prefix of a long online-reorder FFT.
