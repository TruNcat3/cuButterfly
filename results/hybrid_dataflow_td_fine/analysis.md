# HybridDataflow packet-depth breakpoints

## Coverage

The primary scan contains 384 exact-reference-checked points: uint32/uint64,
`logN=10/12`, six representative batches, and every `Td` from 1 through 16.
A 576-run confirmation repeats plateau candidates three times with alternating
scan order. A further 492-point scan covers batch 1 and every multiple of 16
from 16 through 640. The raw tables are in this directory and the sibling
`hybrid_dataflow_td_confirm` and `hybrid_dataflow_td_batch_fine` directories.

## Structural model

For a full-fusion fold, let `R=Ur` be the replicated role warps and let `K` be
the number of independent tokens (`K=32` for the screened `logN=10` graph and
`K=64` for `logN=12`). A packet depth has:

```text
packet_count       P = ceil(K / Td)
serial warp slots  L = ceil(Td / R)
critical slots     S = P * L
token utilization  E = K / (P * L * R)
```

Crossing `Td=kR+1` increases one replica from `k` to `k+1` tokens while the
other replicas remain shorter. The CTA waits for that longest warp, producing
the measured sawtooth at `Td=6/11/16` for `Ur=5` and `Td=7/13` for `Ur=6`.
Packet depth is therefore not a smooth reuse knob: useful candidates cluster at
or just below multiples of the physical replica count.

## Confirmed preferences

- uint32 `logN=10`: `Td=10` has 1.002 mean regret and 1.009 worst regret over
  the six representative batches. It is the robust default.
- uint64 `logN=10`: `Td=5` wins through `2*SM_count`; `Td=9/10` wins in the
  next wave region, and the preference changes again after `4*SM_count`.
  `Td=10` is retained as the simple minimax default rather than encoding a
  periodic table beyond measured coverage.
- uint32 `logN=12`: `Td=12` wins through batch 160. Starting at batch 176 and
  continuing through 640, `Td=6` wins every sampled point, by as much as 1.50x.
  The generated table expresses the boundary as `2*SM_count+1`.
- uint64 `logN=12`: `Td=12` has 1.000 mean regret and 1.001 worst regret in the
  repeated representative scan. It remains the robust default.

## Hardware interpretation

The packet optimum combines control amortization, replica balance, and CTA
waves. For uint64 `logN=10`, `Td=5` uses 66 registers/thread and admits about
five CTAs/SM, while `Td=9/10` uses 88 registers and admits four. Their first
capacity discontinuities are therefore near 400 and 320 transforms on the
80-SM V100. For uint32 `logN=12`, `Td=6` uses 70 registers/thread (about four
CTAs/SM) and has 11 critical token slots; `Td=12` uses 102 registers (three
CTAs/SM) and has 12 slots but half as many packet boundaries. Low batch favors
the latter's control amortization; saturated execution favors the former's
shorter critical path and higher residency.

This yields a portable search rule: enumerate `Td` near `k*Ur`, reject points
with poor `E`, compute register/shared-memory CTA capacity, and explicitly test
batches around `m*SM_count*resident_ctas`. It replaces an arbitrary tile list
with architecture-derived candidate and breakpoint locations.
