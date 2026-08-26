# HybridDataflow packet-depth and token-interleave study

## Scope

The V100 scan covers uint32/uint64, `logN=10/12`, and batch
`1,16,80,160,256,512`. Every row in `summary.csv` passed exact reference
verification. Each case uses 20 warmups and 200 timed repetitions.

Three effects are separated:

- the previous full-fusion point (`Td=4` for `logN=10`, `Td=8` for `logN=12`);
- a deeper packet (`Td=10/12,Ti=1`);
- the same deeper packet with two register states alternated at every fused
  stage (`Td=10/12,Ti=2`).

## Findings

Relative to the previous packet depth, `Ti=2` appears 1.08x--1.37x faster over
the screened matrix. The matched-depth control changes the attribution:
`Ti=1` is normally 1.03x--1.08x faster than `Ti=2`. The main improvement is
therefore packet-boundary amortization, not explicit instruction interleaving.

The deeper `Ti=1` point beats `Ur=1` by 1.35x--1.82x for `logN=10`. For
uint64 `logN=12`, it wins by 1.13x--1.78x and removes the former 160-transform
crossover. uint32 `logN=12` remains non-monotonic: `Ur=1` wins at low batch and
batch 512, while deep full fusion is competitive near batch 160 and wins at
batch 256. No default is inferred for that region.

Cubin resource inspection shows `LOCAL=0` for all new points. For uint64,
`logN=10` grows from 68 to 88 registers/thread under `Ti=2`; `logN=12` grows
from 106 to 112 registers/thread and its stack frame grows from 16 to 32 bytes.
The matched-depth slowdown is therefore consistent with longer register live
ranges rather than local-memory spilling.

## NCU attribution

The matched batch-256 NCU capture confirms the CUDA-event result:

| logN | point | time (us) | active-cycle IPC | active warps | long scoreboard | integer inst. | registers |
|---:|---|---:|---:|---:|---:|---:|---:|
| 10 | old `Td4,Ti1` | 65.696 | 0.36 | 24.89% | 30.69% | 146,604,032 | 68 |
| 10 | deep `Td10,Ti1` | 55.264 | 0.38 | 24.88% | 30.00% | 112,459,776 | 88 |
| 10 | deep `Td10,Ti2` | 58.464 | 0.39 | 24.82% | 27.08% | 128,049,152 | 88 |
| 12 | old `Td8,Ti1` | 369.696 | 0.27 | 18.27% | 28.62% | 449,658,880 | 106 |
| 12 | deep `Td12,Ti1` | 308.928 | 0.32 | 18.51% | 33.70% | 443,301,888 | 106 |
| 12 | deep `Td12,Ti2` | 322.560 | 0.32 | 18.39% | 33.64% | 480,116,736 | 112 |

Deep `Ti=1` is 1.189x and 1.197x faster than the old packet at `logN=10`
and `logN=12`. At `logN=10`, it removes 23.3% of integer instructions while
occupancy remains effectively unchanged. At `logN=12`, register count,
shared-memory residency, and active warps are also unchanged; IPC rises from
0.27 to 0.32. This rules out occupancy as the source of the improvement and is
consistent with less packet-boundary control and idle issue time.

The software pipeline has the intended narrow effect at `logN=10`: `Ti=2`
reduces long-scoreboard stall from 30.00% to 27.08%. It simultaneously executes
13.9% more integer instructions and is 5.8% slower. At `logN=12`, scoreboard
stall is unchanged, integer instructions rise 8.3%, registers rise from 106 to
112, and time regresses 4.4%. `Ti=2` is therefore retained as a physical-unit
ablation but is rejected as the V100 default.

## Reproduction

```bash
./scripts/benchmark_hybrid_dataflow_interleave.sh
sudo -E ./scripts/profile_hybrid_dataflow_interleave_ncu.sh
```

The NCU script profiles the old point, matched-depth `Ti=1`, and `Ti=2` at
batch 256, then writes `summary.csv`. Its output is intentionally separate from
the CUDA-event results so counter collection does not contaminate the timing
table.
