# Single-Slot Tile-Online Screen

V100 CUDA-event medians, uint32 Shoup NTT, `logN=20`, 10+10 factorization,
four-row static IO, and matched producer/consumer weights. Each command uses
three warmups and ten timed repetitions.

| batch | v0.6 ms | warp256 ms | warp128 ms | tile-online ms | online/v0.6 throughput |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.149094 | 0.176538 | 0.189440 | 0.497152 | 0.300x |
| 4 | 0.351027 | 0.426496 | 0.485683 | 1.354342 | 0.259x |
| 16 | 1.068749 | 1.438720 | 1.569485 | 4.375859 | 0.244x |

The tile-online core is correct and has the requested four-CTA resource
shape: 128 registers/thread, no static local allocation, and approximately
20.6 KiB dynamic shared memory. It nevertheless supplies only 0.36--0.38x the
throughput of the synchronous warp128 core. The ratio does not improve with
batch, so launch amortization is not the issue.

The matched NCU capture confirms that attribution. Active warps remain 24.05%
and the launch admits four CTAs/SM, but barrier stall rises to 21.08%, versus
4.86% for synchronous warp128. The 128-register cap also creates 2.904M local
load and 0.461M local store sectors. In contrast, wait stall falls to 10.53%
and long-scoreboard stall falls to 22.26%, so token polling and global-memory
latency are not the controlling limits.

The single merge warp owns all three continuation stages. Once prefix tiles
have been published, the other three warps cannot cross the row-reuse barrier
and the SM is left with too few runnable continuation warps. The next point
must distribute ready continuation tasks across warps or use a work-stealing
tile DAG; another buffer-depth point is not justified. Raising the register
budget alone is also insufficient because it does not remove the 4.3x barrier
stall increase.
