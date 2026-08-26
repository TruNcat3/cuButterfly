# v0.7 Global-Load Sector Root-Cause Audit

## Status

**Source/SASS attribution passed the configured coverage threshold.**
The analytical values remain independent predictions and are compared with the PC-level capture below.

## Controlled Observation

The matched V100 base-clock capture compares uint32 `logN=20`, batch 16, Shoup, `10+10`, and full-scratch implementations.

| metric | v0.6 | v0.7 vector d6 | ratio |
|:--|--:|--:|--:|
| kernel time (us) | 1259.0 | 1345.7 | 1.069x |
| DRAM read (MiB) | 268.5 | 276.8 | 1.031x |
| global-load sectors | 29,851,046 | 39,449,055 | 1.322x |
| global-store sectors | 8,388,611 | 8,337,912 | 0.994x |
| active warps (%) | 47.75 | 24.67 | 0.517x |
| registers/thread | 40 | 106 | |
| shared bytes/CTA | 16416 | 41184 | |

DRAM reads change by +3.1% and store sectors by -0.6%, while global-load sectors change by +32.2%. The gap is repeated or fragmented cache requests, not an additional N-sized external state.

## Stage Sector Model

There are `32,768` local 1024-point row transforms across the producer and consumer roles.

| implementation | stage/address class | sectors/row | predicted sectors | basis |
|:--|:--|--:|--:|:--|
| `v06` | external data | 192 | 6,291,456 | same input and boundary data movement |
| `v06` | stage 0-1 coefficients | 48 | 1,572,864 | CTA radix-4 pair |
| `v06` | stage 2-3 coefficients | 48 | 1,572,864 | CTA radix-4 pair |
| `v06` | stage 4-5 coefficients | 96 | 3,145,728 | CTA radix-4 pair |
| `v06` | stage 6-7 coefficients | 192 | 6,291,456 | CTA radix-4 pair |
| `v06` | stage 8-9 coefficients | 192 | 6,291,456 | CTA radix-4 pair |
| `v07-d6` | external data | 192 | 6,291,456 | same logical input and boundary data |
| `v07-d6` | stage 0-1 coefficients | 48 | 1,572,864 | register-local radix-4 pair |
| `v07-d6` | stage 2-5 coefficients | 128 | 4,194,304 | dense lane loads and shuffles |
| `v07-d6` | stage 6 coefficients | 512 | 16,777,216 | four half-warp 16-byte-stride loads per tile |
| `v07-d6` | stage 7-9 coefficients | 224 | 7,340,032 | four-warp shared-tail continuation |

The model predicts an extra **11,010,048** sectors; NCU measures **9,598,009**. The model therefore explains **87.2%** of the measured delta before PC-level fitting.

The dominant term is stage 6. `coefficient_reuse_stages=6` means `stage < 6`; stage 6 still uses four slot-wise loads. With four consecutive values per lane, each instruction uses half a warp with a 16-byte coefficient stride.

## Reuse-Depth Difference Check

| newly distributed stage | measured reduction | predicted reduction | measured/predicted |
|--:|--:|--:|--:|
| 2 | 1,344,727 | 1,572,864 | 0.855x |
| 3 | 1,977,590 | 1,572,864 | 1.257x |
| 4 | 3,208,643 | 3,145,728 | 1.020x |
| 5 | 7,239,982 | 6,291,456 | 1.151x |
| 6 (d7 candidate; focused capture) | see stage-6 report | layout-dependent | - |

## Source/SASS Attribution

| implementation | category | theoretical sectors | ideal | excessive | instructions |
|:--|:--|--:|--:|--:|--:|
| `v06` | external data | 6,291,456 | 4,194,304 | 2,097,152 | 1,048,576 |
| `v06` | ready/trace | 68,496 | 68,496 | 0 | 68,496 |
| `v06` | v0.6 coefficients | 23,855,104 | 20,905,984 | 2,949,120 | 7,864,320 |
| `v07-d6` | external data | 6,291,456 | 4,194,304 | 2,097,152 | 1,048,576 |
| `v07-d6` | ready/trace | 806,846 | 806,846 | 0 | 806,846 |
| `v07-d6` | stage 0-1 coefficients | 1,572,864 | 1,572,864 | 0 | 1,572,864 |
| `v07-d6` | stage 2-6 coefficients | 23,330,816 | 8,388,608 | 14,942,208 | 4,194,304 |
| `v07-d6` | stage 7-9 coefficients | 7,118,848 | 5,767,168 | 1,351,680 | 1,441,792 |
| `v07-d6` | unclassified global load | 1,941,504 | 1,572,864 | 368,640 | 393,216 |

### Attribution coverage

| implementation | source theoretical sectors | aggregate load sectors | source/aggregate | classified source sectors |
|:--|--:|--:|--:|--:|
| `v06` | 30,215,056 | 29,851,046 | 1.012x | 100.00% |
| `v07-d6` | 41,062,334 | 39,449,055 | 1.041x | 95.27% |

Largest unclassified PCs:

| implementation | PC | file:line | theoretical sectors | source/SASS |
|:--|:--|:--|--:|:--|
| `v07-d6` | `0x71b3e3261b90` | `/home/wt/git/Hermes/cuNTT-v05/src/ntt.cu:135` | 327,680 | `const std::uint32_t quotient = __umulhi(value, twiddle_shoup);` |
| `v07-d6` | `0x71b3e3261b60` | `/home/wt/git/Hermes/cuNTT-v05/src/ntt.cu:234` | 327,680 | `right = u >= v ? u - v : modulus + u - v;` |
| `v07-d6` | `0x71b3e3261b40` | `/home/wt/git/Hermes/cuNTT-v05/src/ntt.cu:3612` | 327,680 | `asm volatile(bar.sync %0` |
| `v07-d6` | `0x71b3e3267b40` | `/home/wt/git/Hermes/cuNTT-v05/src/ntt.cu:135` | 319,488 | `const std::uint32_t quotient = __umulhi(value, twiddle_shoup);` |
| `v07-d6` | `0x71b3e3267b10` | `/home/wt/git/Hermes/cuNTT-v05/src/ntt.cu:234` | 319,488 | `right = u >= v ? u - v : modulus + u - v;` |
| `v07-d6` | `0x71b3e3267af0` | `/home/wt/git/Hermes/cuNTT-v05/src/ntt.cu:3950` | 319,488 | `values[tile] = exchange[tile * 128U + offset];` |

## Pipeline Audit

The current v0.7 dual-group kernel is resident but not a four-phase token pipeline. Within each warp group its program order is:

```text
packet load -> barrier -> row 0 complete NTT -> row 1 complete NTT
            -> row 2 complete NTT -> row 3 complete NTT -> packet store
```

The two warp groups may interleave through normal SM warp scheduling, but there is no software dependence that overlaps one row's load, another row's prefix, a third row's tail, and a fourth row's store. v0.6 already benefits from the same hardware scheduler while resident at four CTAs/SM. v0.7 is limited to two CTAs/SM by 106 registers/thread and 41,184 shared bytes, reducing the sampled active-warp pool by about half.

## Root-Cause Decision

1. The logical external dataflow of v0.6 and v0.7 is effectively the same.
2. The excess load sectors originate primarily in coefficient request organization, especially the vector layout's unreused stage 6, rather than input/boundary traffic.
3. Cache hits prevent those requests from becoming proportional DRAM bytes, but they still consume LD/ST issue, L1 tag, dependency, and scoreboard resources.
4. The current subgraph residency does not provide enough explicit phase overlap to mask that physical-core loss, and its resource footprint halves the resident warp pool.

No kernel or selector change is made by this audit. Candidate corrections remain separate follow-up experiments; the unclassified PC rows stay explicit because inline CUDA line mappings do not support a reliable finer attribution.
