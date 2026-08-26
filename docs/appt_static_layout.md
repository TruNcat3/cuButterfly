# APPT Static Layout and Online Writer

## Why This Boundary Exists

The `7+7+6` register-tail kernel already keeps each dependency-closed tail on
chip. Its remaining final store was nevertheless expensive: one tail task owns
a fixed `c`, while natural order makes adjacent threads land 128 elements
apart. This is a layout/ownership mismatch, not an NTT arithmetic problem.

APPT resolves the mismatch by changing the representation at the subgraph
boundary. For natural index

```text
i = (a << 14) | (d << 7) | c
```

the V100 logN=20 layout is

```text
phi(i) = (a << 14) | (c << 7) |
         ((d & ~mask) | ((d XOR c) & mask))
mask   = fragment_width - 1
```

`fragment_width` is a searched hardware parameter in `{8,16,32}`. The XOR is
an involution within each fragment, so the inverse uses the same low-bit XOR.
The public compatibility ID includes the transform, stage partition, XOR
scheme, fragment width, and ABI version. Two plans may chain static buffers
only when their IDs match exactly.

## Architecture Mapping

```text
natural/static input
        |
        v
 producer CTAs       S-space: stages 0..6; D-time: four c microgroups
        |
        | state1 [a,c,d] + dependency counters
        v
 tail CTAs           S-time: stages 7..19; low/high halves remain resident
        |
        | phi(a,d,c), coalesced APPT static publication
        +-------------------------------> APPT-static output
        |
        | fragment readiness
        v
 writer CTAs         D-space: independent warp tiles (fragment x 32)
        |
        | shared XOR transpose, warp-local synchronization
        v
 natural output
```

Producer, tail, and writer weights are physical service-rate parameters. They
are deliberately separate from the three logical stage descriptors. The
cooperative grid fixes all roles concurrently, allowing the writer to consume
ready fragments while other CTAs continue producer and tail work. There is one
kernel launch and no grid barrier. Natural output reuses `state1` for completed
static values; it does not allocate another N-element workspace.

Each writer warp owns one `fragment_width x 32` shared slot. Loads follow the
static `c,d` representation and stores follow natural `d,c` order. Independent
warps use `__syncwarp`, avoiding the CTA barriers that dominated the first
writer implementation.

## Public Contracts

The C++ API adds `InputOrder::ApptStatic`, `OutputOrder::ApptStatic`,
`ApptRoleMapping`, `Plan::appt_layout_info()`, `appt_static_index()`, and
`appt_natural_index()`. The C API exposes the same contract through
`cubutterflySetNttLayouts` and `cubutterflyPlanGetNttLayoutInfo`.

The physical kernel is currently restricted to a packed, out-of-place,
logN=20 NTT with partition `7+7+6` and the APPT register-tail core. Other
shapes fail explicitly. Natural output remains the fair external-library
contract. Static output is reported separately as the compatible chaining
ceiling.

## Preliminary V100 Results

CUDA-event timings below are medians of three independent processes, each with
three warmups and twenty repeats. They validate the selected neighborhood;
they are not the final paper table. `v0.6` is the mature resident `10+10`
radix-4 kernel.

| word bits | batch | v0.6 ms | APPT static ms | warp-writer natural ms | natural/v0.6 throughput |
|---:|---:|---:|---:|---:|---:|
| 32 | 1 | 0.147 | 0.177 | 0.220 | 0.668x |
| 32 | 4 | 0.346 | 0.582 | 0.688 | 0.503x |
| 32 | 16 | 1.126 | 3.097 | 3.628 | 0.310x |
| 64 | 1 | 0.203 | 0.265 | 0.327 | 0.623x |
| 64 | 4 | 0.554 | 0.861 | 1.074 | 0.516x |
| 64 | 16 | 1.844 | 4.728 | 5.726 | 0.322x |

The experiment establishes two points. First, changing only the output
representation removes the original fixed-`c` scatter mechanism; the u32
static path is faster than the previous register-tail natural scatter, and the
warp writer brings u32 b1 back to approximately that prior kernel time. Second,
natural conversion remains constrained on u64: 128 registers and 48 KiB
dynamic shared memory limit residency to two CTAs/SM, so three physical
services compete inside a smaller resident grid. Both word widths show a new
batch-16 scaling cliff even after local role-weight search. NCU must determine
whether the cliff is readiness ordering, cache working set, or memory-service
competition. The mature v0.6 kernel remains the production/default winner;
APPT static/register-tail remains an explicit research backend.

## Reproduction

Generate the parameter space and run the CUDA-event scan with:

```bash
python3 scripts/generate_appt_static_space.py \
  --word-bits 32 --batch 1 --output /tmp/appt-u32-b1.json

TRIALS=3 WARMUP=3 REPEAT=20 BATCHES=1,4,16 \
  ./scripts/benchmark_appt_static_layout.sh
```

NCU requires administrator counter permission on the current server:

```bash
cd /home/wt/git/Hermes/cuNTT-v05
sudo -E env BIN="$PWD/build-cuda118/cuntt_bench" \
  "$PWD/scripts/profile_appt_static_layout_ncu.sh"
```

The profiler writes raw reports, `summary.csv`, and `analysis.md` under
`results/ncu_appt_static_layout`. Acceptance thresholds are recorded in
`config/v100_appt_static_layout.json`; no default promotion should be claimed
until the NCU sector limits and multi-trial natural-order thresholds pass.
