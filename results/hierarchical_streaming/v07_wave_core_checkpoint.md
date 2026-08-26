# v0.7 wave and resident-core checkpoint

Device: Tesla V100-SXM2-16GB (`sm_70`), CUDA 11.8. All specialized-core
correctness checks use exact modular comparison. Timings are CUDA-event
measurements and vary slightly with clock state.

## Optimization progression

| 64-bit case | implementation | kernel ms | relative to v0.6 |
|:--|:--|--:|--:|
| `logN=12,batch=2` | v0.6 barrier | 0.0102 | 1.000x |
| `logN=12,batch=2` | v0.7 wave/warp `6+6` | 0.0194 | 0.526x |
| `logN=15,batch=2` | v0.6 barrier | 0.0159 | 1.000x |
| `logN=15,batch=2` | v0.7 wave/warp `5+5+5` | 0.0609 | 0.261x |
| `logN=20,batch=4` | original per-element readiness `7+7+6` | about 6.0 | about 0.09x |
| `logN=20,batch=4` | v0.7 wave/warp `7+7+6` | about 1.49 | about 0.37x |
| `logN=20,batch=4` | v0.7 generated resident `10+10`, optimized handoff | 0.55--0.56 | 0.91--0.92x |
| `logN=20,batch=4` | v0.6 barrier resident `10+10` | 0.50--0.51 | 1.000x |

The generated `10+10` point uses the same mature local radix-4 implementation
as v0.6. Its remaining difference is therefore attributable to fixed-role
resource partitioning, readiness publication, and pipeline fill rather than a
different modular-arithmetic unit.

## Batch boundary

The table below uses the measured `9:11` producer/consumer split. Timings are
representative event results from one matched run; the crossover trend is more
stable than any single absolute time under application-controlled clocks.

| bits | batch | v0.6 barrier ms | v0.7 resident stream ms | v0.7 speedup |
|--:|--:|--:|--:|--:|
| 32 | 4 | 0.2711 | 0.3500 | 0.775x |
| 32 | 8 | 0.5283 | 0.5492 | 0.962x |
| 32 | 16 | 1.0694 | 1.0052 | 1.064x |
| 32 | 32 | 2.9668 | 2.3414 | 1.267x |
| 64 | 4 | 0.5080 | 0.5545 | 0.916x |
| 64 | 8 | 0.9666 | 0.9710 | 0.995x |
| 64 | 16 | 1.8760 | 1.8402 | 1.019x |
| 64 | 32 | 3.7168 | 3.6222 | 1.026x |

This is the expected pipeline boundary: more homogeneous packets amortize
fill/drain and fixed-role tails. v0.7 is effectively tied by batch 8 and crosses
v0.6 at batch 16. The unusually large 32-bit batch-32 gain needs a repeated,
clock-controlled release sweep before it is used as a headline result.

The word width changes the residency point. For 32-bit `logN=20`, two CTAs/SM
leave capacity unused; four CTAs/SM improve batch-4 time from about `0.465 ms`
to `0.363 ms`. The V100 generated default is therefore 4 CTAs/SM for 32-bit and
2 CTAs/SM for 64-bit. The corresponding v0.6 32-bit point remains faster at
about `0.271 ms` at batch 4, so numeric width remains a search dimension rather
than a universal mapping constant. A role-weight scan selects `8:12` or `9:11`
at batch 4 and consistently selects `9:11` for larger batches; the generated
V100 default is `9:11`.

## Final NCU diagnosis

`scripts/profile_hierarchical_streaming_ncu.sh` collects three matched
`logN=20,batch=4` kernels: v0.6 barrier, generic v0.7 `7+7+6`, and generated
v0.7 `10+10`. The optimized run confirms that 64-bit barrier stall falls from
15.79% to 3.48%, while 32-bit falls from 20.59% to 5.02%. Runtime improves much
less because explicit polling adds warp instructions and the two-layer
factorization still has a full-transform handoff. The next implementation
target is therefore a generated local-dependency wavefront, not another
barrier variant.
