# Related-work dimensions and evidence (2026-10-29)

Scope: `paper/generated/related_work_table.tex` and `paper/sections/related.tex`.
This bounded review uses existing citation keys only. Operator lists are
representative, not exhaustive; omission does not mean unsupported.

## Row-by-row evidence

| Row and existing keys | Primary evidence and claim boundary |
| --- | --- |
| FFTW (`fftw2005`) | The freshly retrieved [2005 paper](https://www.fftw.org/fftw-paper-ieee.pdf), Secs. IV.C--IV.D and IX, explicitly documents recursive plans, generated codelets, vector-loop ordering, buffered layouts, and SIMD. Sec. IV.D.5 can express both depth-first and breadth-first traversal; this row does not imply a fixed stage organization. The [official manual introduction](https://www.fftw.org/fftw3_doc/Introduction.html) independently verifies DFT, DCT/DST, codelets, reusable plans, and general layouts. One extraction attempt failed; a retry successfully exposed the cited sections despite a recoverable PDF syntax warning. This is a CPU planning example, not a GPU-kernel baseline. |
| SPIRAL / FFTX (`spiral2005`, `fftx2018`) | The freshly retrieved [FFTX author paper](https://www.spiral.net/doc/papers/hipc_2018.pdf), Sec. III.B, describes SPIRAL operator-language rewriting, generated FFT calls and glue code, composed plans, source translation and inlining user code. Sec. IV provides 1D/3D pruned convolution examples. Its introduction explicitly discusses SIMD, threads, accelerators and distributed parallelism. The row limits the operator examples to FFT/FFT applications and does not imply these exhaust SPIRAL's transform domain. The 2005 SPIRAL claim retains the prior audit's algebraic-generation scope. |
| VkFFT (`vkfft2023`, `vkfft`) | The freshly retrieved [official README](https://raw.githubusercontent.com/DTolm/VkFFT/master/README.md) documents multidimensional FFT and DCT; radix choices; Rader/Bluestein; Four-Step reshuffling; shared-memory-dependent limits; runtime-compiled kernels; grouped nearby FFTs for striding; and automatic layout creation. The row reports these explicit choices, without inferring internal tile boundaries or a spatial pipeline. |
| GPU-NTT (`gpuntt2023`, `gpuntt2025`) | The freshly retrieved [2023 ePrint abstract](https://eprint.iacr.org/2023/1410) explicitly optimizes global-memory accesses for synchronization and spatial locality and selects kernel count, block size and block shape per polynomial degree. The [official README](https://raw.githubusercontent.com/Alisah-Ozcan/GPU-NTT/main/README.md) identifies Merge and Four-Step algorithms, NTT/INTT examples, and a separate paper-version branch. The row's detailed organization claims rest on the 2023 source and official repository, not an inference from the 2025 title. |
| FlashFFTConv (`flashfftconv2024`) | Local author v1 text `paper/literature/text/flashfftconv-iclr2024-author.txt`, Secs. 3.1--3.2, Algorithm 1, and Appendix A.2--A.3; freshly checked [author HTML v1](https://arxiv.org/html/2311.05908v1). The method broadcasts matrix operations across sequence, tiles batch/hidden dimensions, and uses Tensor Cores. Decomposition order changes SRAM fusion and HBM handoffs; Appendix A.3 explicitly materializes intermediates in HBM for the four-way case. This is convolution, not a same-contract standalone FFT baseline. |
| HadaCore (`hadacore2024`) | Freshly checked [author HTML v1](https://arxiv.org/html/2412.08832v1), Secs. 3.1--3.2: a warp processes 256-element chunks using 16-point Tensor-Core base transforms and transposition; larger vectors use a full vector per threadblock, shared-memory transpose, synchronization, and warp shuffles. Sec. 2 identifies the operation as FWHT. The row records both within-vector and independent-vector organization. |
| cuButterfly | `paper/sections/method.tex`, dependency-closed subgraphs, space--time organization, communication/readiness, and reuse across variants: stage-bit closure constructs local units; index intersections derive handoffs; stage/data spatial and temporal choices determine residency and resource demands. FFT/NTT/FWHT are representative operator projections. This manuscript supplies its own derivation; this row does not establish that prior systems cannot express equivalent programs. |

## Meaning of the dimensions

- **Operators** names actual arithmetic families or an application composition.
- **Scheduled unit** names the source's plan, kernel, block, chunk, or subgraph;
  these exist at different scopes, which is explicit in the cell text.
- **Stage / data organization** states concrete decomposition, mapping, locality,
  or residency decisions rather than a generic label such as “adaptation.”

A GPU kernel is an execution carrier. A tile subgraph is a scheduling and
residency abstraction. A kernel can execute many tiles, and a larger logical
subgraph can span kernels. Neither abstraction excludes the other. Stage fusion
does not alone show simultaneous stage execution or a spatial pipeline; it can
reuse execution resources over time while retaining intermediates.

General tensor compilers and accelerator mapping work remain in the prose.
They are not assigned transform-specific capabilities based on generality alone,
and missing evidence is not converted into a claim of lack of support.

## Retrieval and validation boundary

The web tool returned an HTTP 404 before source access; official-source `curl`
requests were used instead. No unpublished manuscript content was sent in a
query. Local FlashFFTConv version/provenance remains in the literature manifest.
No GPU, `/sdb`, external repository write, bibliography edit, or performance
claim was involved. The parent agent owns the combined manuscript build and
visual layout review; this task checks the edited source and citation keys.

## Short-feature matrix revision

The next revision separates the former compound organization cells into five
features after Work: representative operators, scheduling unit, stage
decomposition, independent-work mapping, and boundary movement/layout. Each
cell names one reported mechanism in one to three words, rather than a
sentence or a support checkmark. The source map above is unchanged.

The features describe different questions: which arithmetic is requested,
what object is scheduled, how stages are decomposed, how independent data
work maps to workers, and how boundaries are transferred or laid out. They
are not mutually exclusive system categories. SIMD/threads in SPIRAL/FFTX,
vector loops in FFTW, grouped FFTs in VkFFT, CUDA blocks in GPU-NTT, and warp
chunks in HadaCore are source-specific mappings, not a claim that these works
use only data parallelism. The stage column simultaneously records their
decomposition mechanisms. SRAM/HBM in FlashFFTConv denotes the documented
residency/materialization boundary, not a universal configuration.

Abbreviations are expanded under the table. Entries are illustrative and
do not assert unsupported features elsewhere. No unknown or unreported
capability has been encoded as a negative. General compiler descriptions
remain in the shortened prose with their original citations. The matrix uses
footnote-sized type instead of the previous smaller script-sized type;
the parent agent owns final column-width and rendered-layout verification.

### Consistent placement/exchange dimension

The Handoff column now answers one question: how boundary values are placed or
exchanged. FFTW's buffered copies, SPIRAL/FFTX's generated glue, VkFFT's
reshuffled layout, GPU-NTT's global-memory exchange, FlashFFTConv's
retain/materialize choice, HadaCore's shared-memory transpose, and
cuButterfly's successor-aligned stores are all implementation strategies.
Set intersections remain in the prose as the logical dependency construction;
they no longer appear beside physical placement strategies. FlashFFTConv's
retain/materialize cell refers to the previously audited fusion and HBM
materialization paths, without implying that every path uses the same memory
boundary.

Data now denotes independent-work partition. GPU-NTT's block size/shape
parameters and FFTX's list of supported parallel targets alone do not establish
the precise partition from the audited sources, so these cells are NR (not
resolved by this source audit), explicitly distinct from unsupported. The other
partition cells retain their source-specific evidence above. Stage terms “matrix factors” and “16-point
groups” replace Monarch and bases and are explained directly under the table.
