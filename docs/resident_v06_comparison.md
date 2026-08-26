# Resident M/G Lowering versus v0.6

This experiment asks two separate questions:

1. Does lowering `M` logical subgraphs into fewer `G` physical execution
   groups remove the boundary cost introduced by the first v0.7 realization?
2. After that cost is removed, how far is the current physical implementation
   from the best measured v0.6 path?

Conflating these questions would attribute processing-core coverage to the
architecture decomposition. The benchmark therefore reports every layer, not
only the final selected row.

## Protocol

The confirmed V100 matrix covers:

- `word_bits = {32, 64}`;
- `logN = {12, 14, 16, 18, 20}`;
- `batch = {1, 4, 16}`;
- three order-rotated independent processes per implementation;
- 10 warmups and 50 CUDA-event repetitions per process.

The script checks forward correctness at batch 1 by default. Each workload
measures these paths:

| Path | Meaning |
|:--|:--|
| `v06_hybrid2d` | mature two-pass Hybrid2D radix-4 implementation |
| `v06_resident` | established generated `10+10` dataflow core, at `logN=20` |
| `v07_full` | four logical subgraphs and three materialized boundaries, `G=M=4` |
| `v07_resident_generic` | the same logical M, lowered to two physical groups, generic descriptor core |
| `v07_resident_core` | the same lowering with the requested dataflow physical mapping |

`v0.6-best` is selected from both measured v0.6 candidates. `M/G-best` is
selected only from the three resident-lowering ablations. It is deliberately
not called `v0.7-best`: HybridDataflow resident-radix4, packet-shared radix-4,
and the other previously optimized v0.7 physical cores are not present in this
controlled experiment.

## Confirmed Result

Across 30 workloads, M/G-best reaches **0.394x** v0.6-best geometric-mean
throughput and wins **1/30** points. This measures the generic coverage gap of
the new lowering path; it is not a version-level v0.7 result and does not
supersede the earlier v0.7 wins.

For context, the actual release-level 18-shape HybridDataflow comparison
remains authoritative for its domain: v0.7-search reaches 1.028x v0.6-search
geometric-mean throughput with 8 wins, 1 parity point, and 9 losses. Its
uint64 `logN=10` group reaches 2.127x, while its two `logN=12` groups reach
0.785x and 0.651x. Those resident-radix4 candidates and the winning `logN=10`
workloads are absent from this M/G ablation. See
[`v06_v07_comprehensive_comparison.md`](v06_v07_comprehensive_comparison.md).

| Length | Points | M/G-best / v0.6-best | Wins | Range |
|---:|---:|---:|---:|:--|
| 12 | 6 | 0.355x | 0/6 | 0.269x--0.471x |
| 14 | 6 | 0.336x | 0/6 | 0.256x--0.446x |
| 16 | 6 | 0.327x | 0/6 | 0.255x--0.409x |
| 18 | 6 | 0.302x | 0/6 | 0.218x--0.396x |
| 20 | 6 | 0.803x | 1/6 | 0.649x--1.007x |

The version-level result is negative, but the layered attribution is
decisive:

| Layer | Geomean throughput versus v0.6-best | Incremental gain |
|:--|---:|---:|
| `v07_full` | 0.089x | baseline |
| `v07_resident_generic` | 0.321x | 3.598x from M-to-G lowering |
| `v07_resident_core` | 0.372x | 1.157x from the requested core mapping |
| `M/G-best` | 0.394x | per-workload choice among lowering ablations |

The lowering gain grows from 2.88x at batch 1 to 4.43x at batch 16. More
homogeneous work therefore does amortize physical execution groups better; the
earlier large-batch loss came from materializing logical boundaries, not from
the logical decomposition itself.

## Physical Equivalence

The only currently generated two-group dataflow specialization is `10+10`.
After matching its stage partition, core, target residency, and `9:11` CTA
weights, logical `M=4, G=2` v0.7 and physical `M=2, G=2` v0.6 agree as follows:

| bits | batch 1 | batch 4 | batch 16 |
|---:|---:|---:|---:|
| 32 | 1.0022x | 1.0040x | 1.0066x |
| 64 | 1.0007x | 1.0023x | 1.0002x |

Values are v0.6-resident time divided by v0.7-resident-core time. The
0.02%--0.66% interval is measurement-level parity. Template metadata and
resident lowering therefore add no observable execution cost once the
physical schedule is identical.

This equivalence does not rank the complete v0.7 repository. Within this
ablation, Hybrid2D is selected at 29/30 workloads, including five of the six
`logN=20` rows. The one M/G-candidate win is the uint32, `logN=20`, batch-16
point where both sides choose the resident core and agree within 0.66%.

## Root Cause And Next Step

For execution groups of 6, 7, 8, or 9 stages, the current
`dataflow-radix4` request falls through to the persistent descriptor radix-4
kernel. It does not instantiate the generated row/column dataflow core used by
`10+10`. The generic path pays a fixed cooperative resident grid, runtime task
indexing, descriptor branches, readiness bookkeeping, and a shared-memory
tile implementation. Hybrid2D instead dispatches compile-time local radix-4
passes and remains 2.1x--4.6x faster than lowered generic at these lengths.

The next optimization target is consequently narrow:

1. generate compile-time physical group cores for stage logs 6--9;
2. search threads, rows/units, target CTAs, and role weights per physical
   `(stage_log, word_bits)` core;
3. reuse those cores for every logical partition that lowers to the same
   physical execution partition;
4. repeat this matrix, then send only crossover and unexplained rows to NCU.

Searching more logical partitions before those physical cores exist would
repeat equivalent generic schedules and cannot close the observed gap.

## Reproduction

From the repository root:

```bash
./scripts/benchmark_resident_v06_comparison.sh
```

The default output is `results/resident_v06_comparison/`. Override dimensions
without editing the script, for example:

```bash
OUTPUT_DIR=results/resident_v06_comparison/extra_batch \
LOGNS="12 13 14 15 16 17 18 19 20" BATCHES="1 2 4 8 16 32" \
TRIALS=5 WARMUP=20 REPEAT=100 \
./scripts/benchmark_resident_v06_comparison.sh
```

The confirmed table, all trial samples, and the machine-readable summary are
in [`results/resident_v06_comparison/v100_confirmed/`](../results/resident_v06_comparison/v100_confirmed/).
