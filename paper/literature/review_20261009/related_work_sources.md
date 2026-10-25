# Related-work claim review, 2026-10-09

Scope: sections/related.tex and generated/related_work_table.tex.
This bounded nearest-work review reuses metadata in [REFERENCES.md](../../REFERENCES.md).
No new BibTeX record was authored. Public source requests contained no unpublished text.

## Claim-to-source map

| Keys | Claim used | Primary evidence and review status |
| --- | --- | --- |
| fftw2005 | Codelets and empirical planning | Existing audit: [official paper](https://www.fftw.org/fftw-paper-ieee.pdf). No stronger new claim. |
| spiral2005, fftx2018 | Algebraic generation/search; composed FFT plans and surrounding user computation | Existing metadata/author-paper audit: [SPIRAL DOI](https://doi.org/10.1109/JPROC.2004.840306), [FFTX DOI](https://doi.org/10.1109/HiPCW.2018.8634111), [author PDF](https://www.spiral.net/doc/papers/hipc_2018.pdf). Project homepage and PDF returned HTTP 465 on this refresh. Broad claims retained from prior verified scope. |
| vkfft2023, vkfft | Complete FFT implementations and runtime generation | [Official README](https://raw.githubusercontent.com/DTolm/VkFFT/master/README.md), fetched 2026-10-09. Metadata retained from [DOI](https://doi.org/10.1109/ACCESS.2023.3242240). Detailed paper scheduling unknown to this review. The table uses only verified runtime generation and makes no detailed scheduling claim. |
| gpuntt2023, gpuntt2025 | Memory-access optimization and CUDA kernel/block parameters | [ePrint abstract and BibTeX](https://eprint.iacr.org/2023/1410), fetched 2026-10-09: global-memory access/locality, kernel count, block size and shape. [Official README](https://raw.githubusercontent.com/Alisah-Ozcan/GPU-NTT/main/README.md), fetched same date: Merge/4-Step and current code versus paper_version. PDF returned HTTP 403. Detailed 2025 method is not inferred from its title; claim rests on the 2023 abstract. |
| hadacore2024 | Hardware-aware Hadamard decomposition, tensor cores and data exchange | [Author abstract/metadata](https://arxiv.org/abs/2412.08832) and [full text v1](https://arxiv.org/html/2412.08832v1), sections 2.3--3, fetched 2026-10-09. Seven authors and 2024-12-12 date match existing record. No journal/DOI metadata on landing page; retained as preprint. |
| flashfftconv2024 | Matrix decomposition, tensor cores and memory-aware fusion | Re-read local author v1 text, sections 3.1--3.2. [Author HTML v1](https://arxiv.org/html/2311.05908v1) accessible on 2026-10-09. Hash/version in [manifest](../manifest.json). |
| tvm2018 | Computation/schedule separation | Existing [USENIX paper audit](https://www.usenix.org/conference/osdi18/presentation/chen), unchanged claim. |
| ansor2020 | Hierarchical sketches and annotation search | Re-read local official text, sections 3--5, Table 1; [official PDF](https://www.usenix.org/system/files/osdi20-zheng.pdf). Hash/version in manifest. |
| roller2022 | Hardware-aligned tiles across memory levels | Re-read local official text, section 3, Figures 6--8; [official PDF](https://www.usenix.org/system/files/osdi22-zhu.pdf). Hierarchical rTile configuration and single-core construction explicit. Hash/version in manifest. |
| tensorir2023 | Block signatures, outer transformations and tensorization | Re-read local author v2 text, sections 2--4; [author PDF](https://arxiv.org/pdf/2207.04296v2). [arXiv metadata](https://arxiv.org/abs/2207.04296), fetched 2026-10-09, matches eleven existing authors. Existing ASPLOS DOI metadata retained; arXiv date is not substituted for conference year. |
| welder2023 | Hierarchical tile-graph, producer/consumer connection, shape propagation, traffic/footprint scheduling | Re-read local official text, sections 3.1--3.3, SetConnect/Propagate and traffic/footprint discussion; [official PDF](https://www.usenix.org/system/files/osdi23-shi.pdf). Welder already discusses dependent input regions and irregular-access upper bounds; no claim that it lacks dependency analysis. Hash/version in manifest. |
| eyeriss2016 | Data movement under controlled hardware resources | Existing audit and local author text, sections IV--VII; [MIT PDF](https://eems.mit.edu/wp-content/uploads/2016/04/eyeriss_isca_2016.pdf). Accelerator energy results are not GPU speedup evidence. |
| timeloop2019 | Workload/architecture/mapping separation; hierarchical spatial/temporal tiling | [Author PDF](https://parashar.org/ispass19.pdf), fetched and text-extracted in memory 2026-10-09 (726123 bytes), sections IV--VI. No claim about current software extensions. |

## Method and evidence boundaries

- The final table column describes cuButterfly's construction, not proof that other compilers cannot express it.
- Closure, adjacent-set intersections and successor-aligned writeback require this manuscript's own derivation and implementation checks.
- Semantics/workload/target are separated as inputs and construction concerns. Their performance effects remain coupled.
- No cited speedup is copied into the table. Precision, semantics, versions, adapters and timing boundaries remain necessary for experiments.
- APPT is an internal unpublished draft and is excluded from formal related-work positioning.

## Validation and unresolved metadata

The citation-management validator parsed 33 existing entries, found no duplicate
or unresolved citation in related.tex, and reported ten pre-existing missing
year fields in software/documentation entries. These entries carry access-date
notes; publication years were not guessed. Three recommended-page warnings
concern Gentleman--Sande, Stockham and FlashFFTConv; ICLR papers need not have
proceedings page ranges. Unused-entry warnings from checking one section do not
imply the entries are unused by the manuscript. Table keys were checked separately.

No new external capability, detailed VkFFT schedule or 2025 GPU-NTT method claim
remains implicit. A fresh FFTX full-text check remains an access limitation for
its unchanged broad composition claim.
