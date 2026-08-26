# Fine batch boundary result

The scan compares three plateau candidates per numeric/length region at batch
1 and every multiple of 16 through 640. All 492 points pass exact reference
verification. The full per-batch table is in `summary.md`.

The stable boundary is uint32 `logN=12`: `Td=12` wins through batch 160 and
`Td=6` wins every sample from 176 through 640. On the 80-SM V100 this is the
strict boundary `batch > 2*SM_count`.

Other winner changes are wave-dependent rather than monotonic. In particular,
uint64 `logN=10` prefers `Td=5` at batch 1--160, `Td=9/10` at 176--320, and
returns to `Td=5` when the 88-register kernels cross their four-CTA/SM capacity
at batch 320. Later regions alternate again. These data support a wave-aware
cost model, but not an unbounded threshold table inferred from batch <=640.
