# v0.7 Packet Streaming versus v0.6

Device: Tesla V100-SXM2-16GB (`sm_70`). Numeric case: uint32 forward NTT,
`logN=20`, modulus 998244353, full-scratch `10+10` decomposition.

## Throughput

CUDA-event values are medians from three independent processes. Execution
order rotates between trials; each process uses 10 warmups and 50 timed
iterations. Forward v0.6 and forward/inverse v0.7 online correctness are
checked before timing.

| batch | v0.6 (ms) | v0.7 aggregate (ms) | v0.7 online (ms) | aggregate/v0.6 | online/v0.6 | online/aggregate |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 1.460367 | 1.081549 | 1.129964 | 1.350x | 1.292x | 0.957x |
| 32 | 4.113428 | 2.417971 | 2.502431 | 1.701x | 1.644x | 0.966x |
| 64 | 9.128469 | 4.930581 | 5.137776 | 1.851x | 1.777x | 0.960x |

Across these points, v0.7 aggregate reaches 1.620x geomean throughput over
v0.6 and v0.7 online reaches 1.557x. Both replace v0.6 for this numeric and
length regime. Online has not yet replaced aggregate: packet visibility and
wave staging cost about 3.4-4.3% throughput.

## Fixed-Clock Evidence

The valid fixed-clock batch-16 physical-core capture measures v0.6 at
1259.5 us and packet128 at 1061.2 us, a 1.187x throughput improvement. Dynamic
warp instructions fall to 0.948x of v0.6 and CBU instructions fall from 8.56M
to 3.62M. This establishes that the resident packet radix-4 codelet is more
efficient than the v0.6 physical core.

The corrected wave-16 fixed-clock attribution is:

| batch | aggregate (us) | online (us) | online/aggregate | warp inst. | load sectors | shared-load conflicts | CBU inst. |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 1226.2 | 1209.4 | 0.986x | 1.102x | 1.096x | 1.241x | 1.250x |
| 32 | 2779.2 | 3000.8 | 1.080x | 1.116x | 1.088x | 1.243x | 1.300x |

At batch 16 the overlap repays the extra instructions in the single fixed-clock
capture. At batch 32 it does not. In both cases online executes materially more
control and memory work. Long-scoreboard stall rises from 28.97% to 34.65% at
batch 16 and from 26.50% to 27.45% at batch 32. CUDA-event medians remain the
throughput-selection authority; NCU replay is used here to identify mechanism.

Relative to the earlier fixed-clock v0.6 control at 1259.5 us, the current
batch-16 aggregate and online points are 1.027x and 1.041x faster. The isolated
packet128 core remains the stronger 1.187x physical-core result.

## Architectural Conclusion

The current performance hierarchy is:

1. The packet128 physical subgraph is better than the v0.6 physical core.
2. The complete v0.7 aggregate path is substantially faster than v0.6 under
   steady CUDA-event timing.
3. Correct online packet handoff preserves most of that gain but remains about
   4% behind v0.7 aggregate.
4. The remaining online deficit is orchestration overhead: per-packet global
   acquire loads, wave barriers, shared staging conflicts, and long-scoreboard
   latency. It is not a modular-arithmetic throughput deficit.

Therefore v0.7 can replace v0.6 for the tested uint32 `logN=20`, batch 16-64
table entries, while aggregate remains the selected v0.7 mode. Online stays a
research candidate until it beats aggregate with exact correctness.

The first shared-staging padding candidate was rejected: it did not reduce
bank conflicts, increased registers from 64 to 71, and did not improve event
timing. Packet acquire polling backoff is now an explicit runtime design
parameter. Its event screen selects 128/64/512-cycle sleeps for batch 16/32/64
and closes online to 0.954x/0.965x/0.967x of aggregate. Because the selected
points improve the prior 64-cycle default by at most 0.4%, readiness polling
frequency is a tuning parameter rather than the remaining architectural
bottleneck.

The fixed-clock polling capture confirms that conclusion. Across batch
16/32/64, per-packet readiness adds 8.7%/9.4%/8.1% global-load sectors,
25.0%/22.5%/24.3% shared-load conflicts, and 10.2%/11.5%/8.6% warp
instructions relative to aggregate. Batch-64 replay time is nevertheless only
1.0% slower, showing that the repeated work is increasingly hidden as the
homogeneous stream grows.

An identity-preserving bitmap follow-up packs the 256 packet flags into eight
words and acquires four quadrant masks per q-wave. Event medians are:

| batch | aggregate (ms) | per-packet (ms) | wave-bitmap (ms) | bitmap/per-packet | bitmap/aggregate |
|---:|---:|---:|---:|---:|---:|
| 16 | 1.086300 | 1.132974 | 1.136128 | 0.997x | 0.956x |
| 32 | 2.424771 | 2.522071 | 2.444616 | 1.032x | 0.992x |
| 64 | 4.942787 | 5.152133 | 4.990546 | 1.032x | 0.990x |

Bitmap publication is therefore selected only from batch 32 onward pending
fixed-clock counter confirmation. At batch 16, atomic-OR contention offsets
the smaller polling footprint and per-packet readiness remains the better
online mode.

The matched readiness NCU capture revises the mechanism: bitmap reduces
per-packet global-load sectors by only 0.6%-0.7% and leaves shared-load
conflicts unchanged. Its batch-16/32 replay improvement instead corresponds to
6.4%/11.0% fewer CBU instructions. Flag compression solves control work, not
the residual 8%-9% online sector excess.

A warp-row stage-0/1 mapping was then added to test whether mixing four rows in
one warp caused coefficient-broadcast and shared-bank phase loss. Event timing
finds only a 1.9% per-packet batch-16 gain; bitmap changes by -0.7%/-0.2%/-0.5%
at batch 16/32/64. It is therefore retained as a counter-attribution ablation,
not selected for dispatch.

The matching fixed-clock capture verifies that the source mapping was correct:
warp rows reduce bitmap load sectors by 8.7%-8.8%, shared-load conflicts by
18.9%-19.2%, warp instructions by about 1%, and LSU instructions by
1.6%-2.1%. Those reductions improve replay time at batch 16/32 but regress it
at batch 64 and do not move steady CUDA-event timing. They remove work that is
not the warm critical path.

A producer/consumer allocation scan selects `8:12` at batch 16 and retains
`5:15` at batch 32/64. Best warp-row throughput is
0.965x/0.987x/0.982x aggregate, so the faster consumer does not expose unused
producer capacity. Folding the packet core's post-compute wave barrier into
the next staging barrier is correct in forward and inverse execution, but
changes event time by at most 0.6%; the best folded-or-unfolded online points
reach 0.967x/0.989x/0.992x aggregate.

These controls close four suspected local causes: readiness polling frequency,
flag representation, row mapping, and redundant wave barriers. The remaining
fixed-`10+10` deficit is the fine-grained publication/staging protocol itself.
The next useful v0.7 experiment is a generated stream with more than two
homogeneous subgraphs, where a longer intra-transform dependency stream can
amortize that protocol. More micro-tuning of the two-segment instance is not
expected to reverse its ranking against aggregate.
