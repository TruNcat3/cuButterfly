# Dual Warp128 Row-Group Screen

V100, uint32 Shoup NTT, `logN=20`, 10+10 factorization, four-row static IO.
The dual point places two independent synchronous four-warp row subgraphs in
one 256-thread CTA and targets two CTAs/SM. The single point uses one group in
a 128-thread CTA and four CTAs/SM. Both expose 16 resident warps and use the
same register-resident stage-7/8/9 continuation.

Both binaries compile to 101 registers/thread and a 16-byte stack frame. The
dual CTA uses approximately 41.2 KiB shared memory; the single CTA uses 20.6
KiB.

Three confirmation trials use five warmups and thirty repetitions. Values
below are trial medians.

| batch | v0.6 ms | single ms | dual ms | dual/single throughput | dual/v0.6 throughput |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.136 | 0.177 | 0.177 | 0.999x | 0.768x |
| 4 | 0.320 | 0.426 | 0.470 | 0.906x | 0.681x |
| 16 | 1.106 | 1.506 | 1.305 | 1.154x | 0.847x |

The result is a workload crossover. At small batch, V100 already interleaves
enough independent CTAs, and the larger CTA is neutral or harmful. At batch
16, two independent named-barrier groups give each resident CTA more ready
row work and improve the physical core by 15.4%. Subgraph replication is
therefore selected by available workload and resource residency, not fixed by
the architecture template.

The matched NCU script includes this dual-group point with its selected
batch-16 producer/consumer weight `8:7`.

The completed counter pass measures 1495.808 us for dual versus 1614.944 us
for single group and 1255.808 us for v0.6. Dual and single execute essentially
the same global-load sectors (39.819M/39.823M) and warp instructions
(237.821M/237.504M), and both expose about 24% active warps. The dual mapping
instead lowers long-scoreboard stall from 35.59% to 28.58%. Thus the measured
gain is caused by issue interleaving between independent row subgraphs, not
by an accidental reduction in work.

Against v0.6, the dual point still performs 33.5% more global-load sectors and
30.6% more warp instructions, uses 101 rather than 40 registers/thread, and
provides roughly half the active-warp percentage. This separates the next
design-space axis cleanly: preserve dual row-group scheduling, but substitute
a more efficient physical butterfly codelet.

The first physical-core ablation explicitly reuses each lane's stage-0--5
coefficient pair across register slots. It is exposed separately as
`homogeneous-warp128-coefficient-reuse-static-io-radix2`; the original core
remains unchanged for controlled comparison. An unlocked-clock event screen
was inconclusive, but the matched NCU pass resolves the result: coefficient
reuse lowers global-load sectors from 39.759M to 27.058M (-31.9%), lowers
long-scoreboard stall from 28.86% to 24.39%, and improves time from
1484.672 us to 1468.992 us (1.011x throughput). Registers rise from 101 to
104/thread and active warps remain 24.6%, so the request reduction is real but
only weakly exposed as elapsed-time gain.

The cached point is still 1.149x slower in time than v0.6 and reaches 0.870x
its throughput. It now performs fewer global-load sectors than v0.6, but
executes 28.7% more warp instructions and exposes only 51.7% as many active
warps. This moves the residual target decisively from coefficient traffic to
physical-codelet instruction count and resident parallelism.
