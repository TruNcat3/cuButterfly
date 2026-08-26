# v0.7 Resident 7+7+6 Checkpoint

## Scope

This checkpoint evaluates the first generated three-layer kernel that can
pipeline subgraphs inside one `logN=20` transform. The point uses a `7+7+6`
stage partition, three fixed CTA roles, two dependency-wave boundaries, and a
resident radix-4 core in every role. It is a candidate in the design space; it
does not replace the current default `10+10` point.

The physical mapping exposed by the CLI is:

| parameter | V100 default |
|:--|:--|
| threads per CTA | `256+256+256` |
| subgraph units per CTA | `32+32+32` |
| CTA service weights | `6+6+8` |
| target CTAs/SM, uint32 | 3 |
| target CTAs/SM, uint64 | 2 |
| boundary storage | two full-size scratch buffers |

`units_per_cta` is a physical-unit parameter, independent of the logical
`7+7+6` factorization. The scan covered 8, 16, and 32 resident units per CTA;
32 was the best tested value for both numeric widths. The final role receives
the larger service weight because it also owns the natural-order output
permutation.

## Semantics And Correctness

The kernel publishes 64 readiness waves at the first boundary and 128 at the
second. Release/acquire counters make a consumer visible only after the exact
producer set for its wave has completed. Forward, inverse, and delta-vector
tests pass exactly for both supported word widths.

`--trace-pipeline` confirms a real single-transform wavefront. In one uint64
capture, segment 0 ran over timer interval
`[1786980318096130016, 1786980318096623584]`, while segment 1 began at
`1786980318096191456`, before segment 0 ended. Segment 2 began 4096 timer ticks
after segment 1 ended. Thus the first edge overlaps, while the second edge
still has an almost transform-wide dependency closure.

## Matched V100 Timing

Conditions: Tesla V100-SXM2-16GB, `logN=20`, `batch=1`, forward transform,
30 warmups, 100 CUDA-event iterations, exact verification.

| words | v0.6 barrier ms | resident 10+10 ms | generic 7+7+6 ms | resident 7+7+6 ms |
|---:|---:|---:|---:|---:|
| 32 | 0.087122 | 0.147599 | 0.432241 | 0.460524 |
| 64 | 0.156498 | 0.228731 | 0.567828 | 0.683448 |

The generated three-layer point is 6.5% slower than the generic wave kernel
for uint32 and 20.4% slower for uint64. It is also well behind the mature
barrier and `10+10` units at batch 1. This result does not reject the dataflow
factorization: the trace verifies the intended overlap. It shows that the
current physical scheduler pays more for a second full boundary, fixed waiting
roles, polling, and final permutation than it recovers from the available
overlap.

The architectural conclusion is therefore narrower than a speedup claim:

1. `7+7+6` creates a valid intra-transform dependency wave that `10+10` cannot.
2. Logical factorization, units per CTA, role weights, and residency must be
   searched as separate dimensions.
3. The next performance implementation needs distributed ownership or a local
   boundary whose dependency closure lets the third role start earlier. Adding
   more global queue arbitration is unlikely to help; the rejected queue
   prototype already serialized on atomics.

## Reproduction

Run the focused design-space scan:

```bash
BIN="$PWD/build-v07/cuntt_bench" \
  WARMUP=10 REPEAT=50 \
  ./scripts/benchmark_hierarchical_resident_776.sh
```

Collect matched NCU counters after the event-time screen:

```bash
sudo -E env BIN="$PWD/build-v07/cuntt_bench" \
  OUTPUT_DIR="$PWD/results/ncu_hierarchical_streaming_776" \
  ./scripts/profile_hierarchical_streaming_ncu.sh
```
