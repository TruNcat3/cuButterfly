# Candidate Implementations

This document separates implemented choices from candidate processing units and
kernel forms. A candidate enters the generated design space only after its
numeric contract, resource envelope, layout, and correctness test are explicit.

## Common Kernel Families

| Family | Current role | Main parameters | Candidate extension |
|:--|:--|:--|:--|
| temporal tile | complete local transform in one CTA | local length, threads, radix, exchange | double-buffered layout and asynchronous copy |
| hierarchical | resident prefix plus global suffix stages | local stages, threads, radix | generated suffix groups and cache-aware handoff |
| online reorder | permuted prefix store plus resident suffix | split, columns, threads, layout | generated multi-level reorder and epilogue fusion |
| Hybrid2D | two factor transforms with fused cross twiddle | factorization, rows, threads, word width | multi-factor and mixed-radix decomposition |
| compact stage | V100 NTT stage groups with compact roots | stage split, output order | generated splits from hardware model |
| warp hybrid | controlled warp/register/shared experiment | warp stages | operator-specific shuffle networks |
| stage pipeline | explicit stage-role experiment | `Us`, pipeline warps, handoff | persistent CTA or cluster handoff on newer GPUs |

## FFT

| Status | Unit | Design-space value | Main limitation or next step |
|:--|:--|:--|:--|
| implemented | radix-2/4/8 scalar complex butterfly | baseline stage grouping | shared synchronization and twiddle traffic |
| implemented | four-multiply and Gauss-3 complex multiply | interchangeable arithmetic core | Gauss dependencies offset saved multiply on V100 |
| implemented | thread-register DFT8 | three stages in one thread | uncoalesced global layout without CTA staging |
| implemented | CTA DFT8, 32/64/128/256 threads | variable `Us`, repeated three-stage folding | crossover after `N=64` from layout synchronization |
| implemented | FP16/FP32 WMMA DFT8 | Tensor Core local core | mixed precision and memory-bound small transform |
| candidate | split-radix or Stockham codelet | lower operation or permutation cost | must preserve explicit output-layout contract |
| candidate | generated long register/shared FFT | reuse established codelet hierarchy | register growth and occupancy thresholds |
| candidate | Tensor Core block FFT | alternative arithmetic service | packing, precision, twiddle, and composition cost |
| candidate | CTA DFT8 as online-reorder prefix | generated long-transform composition | suffix layout and multi-CTA handoff |

Tensor Cores are not assumed to solve the long FFT gap. Existing NCU data shows
the WMMA DFT8 executes Tensor instructions, but local arithmetic acceleration
does not remove global permutation, shared exchange, twiddle access, or suffix
composition costs.

## NTT

| Status | Unit | Design-space value | Main limitation or next step |
|:--|:--|:--|:--|
| implemented | radix-2/4/8 with Shoup multiplication | stage grouping and precomputed reduction | expanded tables increase long-scoreboard stalls |
| implemented | Barrett multiplication | smaller coefficient representation | wider temporaries and register pressure on V100 |
| implemented | fused coset twiddle | removes standalone Hybrid2D multiply | second-pass coefficient bandwidth at large length |
| implemented | compact root-plus-Shoup stage core | closes native-layout GPU-NTT gap | natural-order permutation remains expensive |
| candidate | Montgomery or lazy reduction | alternative reduction pipeline | exact range proof and overflow contract required |
| candidate | on-chip root recurrence | reduce coefficient bandwidth | extra dependency chain and arithmetic cost |
| candidate | compressed/staged roots | trade shared capacity for HBM/L2 traffic | reuse distance must be hardware-calibrated |
| candidate | 32-bit packed residue units | increase integer throughput | restricted modulus and accumulation range |

## FWHT

| Status | Unit | Design-space value | Main limitation or next step |
|:--|:--|:--|:--|
| implemented | scalar radix-2/4/8 | common shared-memory control | barriers dominate at small local lengths |
| implemented | warp-register/XOR-swizzle hierarchy | imported validated local core | registers reach 255/thread at `logN=15` |
| candidate | Tensor Core Hadamard tile | matrix processing unit | conversion overhead and precision support |
| candidate | multi-CTA long FWHT with online reorder | extend common mapping beyond local core | inter-CTA residency and permutation policy |

The register hierarchy is adapted from Dao-AILab's fast-hadamard-transform;
provenance is recorded in `THIRD_PARTY_NOTICES.md`. It is evidence that the
framework can absorb an established core, not a claim of inventing that core.

## XOR-Zeta And Related Layered Operators

| Status | Unit | Design-space value | Main limitation or next step |
|:--|:--|:--|:--|
| implemented | uint32 radix-2/4/8 | lightweight non-twiddle operator | no same-machine external baseline |
| candidate | packed narrow integer unit | greater lane-level work | overflow and semiring semantics |
| candidate | subset/Mobius transforms | reuse topology and inverse contract | operator traits and reference validation |
| candidate | structured matrix layers | broader butterfly-like workloads | graph regularity and coefficient storage |

## Admission Checklist

Before integrating another core, record:

1. upstream source, revision, license, and modifications;
2. operator, length, precision, direction, and numeric-error contract;
3. native input/output layout and in-place restrictions;
4. required warp/block shape, registers, shared memory, and architecture;
5. standalone reference validation and sanitizer result;
6. generated adapter where cuButterfly still owns mapping and scheduling;
7. a controlled core-only comparison and a complete-library comparison.

If item 6 cannot be satisfied, the implementation remains an external baseline
rather than evidence of processing-unit integration.
