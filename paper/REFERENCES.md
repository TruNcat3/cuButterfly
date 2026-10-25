# Reference verification notes

Initial references checked on 2026-09-16; literature expansion checked on
2026-09-17; nearest-work claim review refreshed on 2026-10-09.
Bibliographic metadata and product documentation are
separate from the versions of libraries used in the performance archives.
The browser connector returned an HTTP 404 service error; public primary
sources were accessed directly over HTTPS instead.

The 2026-10-09 revision reuses existing bibliography records and source audits;
it adds no new entry. Claim locations, refreshed primary-source checks, and access
failures are recorded in
[related_work_sources.md](literature/review_20261009/related_work_sources.md).
SPIRAL/FFTX retains its previously audited broad generation/composition
positioning: the project server returned HTTP 465 during this refresh.
The GPU-NTT ePrint landing page and official repository were accessible, but
its PDF returned HTTP 403. Detailed VkFFT paper scheduling remains unknown
to this review. No negative capability claim follows from these access limits.
The internal unpublished APPT draft is excluded from formal prior-work positioning.

The application-evaluation revision adds three software citations,
`neuraloperatorcode`, `quarotcode`, and `heongpucode`, using their official
repositories. The pinned commits, licenses, source call sites, and integration
boundaries are in [END_TO_END.md](END_TO_END.md) and
[experiments/end_to_end.json](experiments/end_to_end.json). These references
justify application selection and execution contracts. They do not supply
cuButterfly application measurements. Software licensing is recorded separately
from model-weight and dataset terms.

On 2026-09-21 the author-requested UnionSparse manuscript
([arXiv:2608.09291](https://arxiv.org/abs/2608.09291), downloaded PDF v1)
was read for its evaluation structure, especially Section VII-B--D,
Figure 10, Table III, and Figure 12. Its title and author list were checked
against the PDF. The `unionsparse2026` BibTeX entry is retained as a writing
reference, without a main-text citation implying butterfly evidence or
confirmed journal publication. The source hash is in `literature/manifest.json`.

| BibTeX key | Verification source | Scope |
| --- | --- | --- |
| cooley1965 | Publisher-deposited Crossref record, DOI 10.1090/S0025-5718-1965-0178586-1 | Title, authors, journal, year and pages |
| gentleman1966 | Publisher-deposited Crossref record, DOI 10.1145/1464291.1464352 | Actual title is *Fast Fourier Transforms*; no invented longer title |
| stockham1966 | Publisher-deposited Crossref record, DOI 10.1145/1464182.1464209 | Title, author, proceedings and year |
| fftw2005 | Crossref DOI 10.1109/JPROC.2004.840301; [official paper](https://www.fftw.org/fftw-paper-ieee.pdf); [official introduction](https://www.fftw.org/fftw3_doc/Introduction.html) | Metadata; empirical planning and codelet generation |
| spiral2005 | Publisher-deposited Crossref record, DOI 10.1109/JPROC.2004.840306 | Title, complete author list, pages; broad transform-code-generation positioning. The project bibliography endpoint was unavailable during checking |
| roofline | Publisher-deposited Crossref record, DOI 10.1145/1498765.1498785 | Authors, journal, year and pages; used for the standard work/traffic perspective, not as a subgraph-capacity proof |
| dao2019 | [PMLR paper page](https://proceedings.mlr.press/v97/dao19a.html) | Complete authors, pages and abstract; learned butterfly representation distinguished from execution mapping |
| cufft | [NVIDIA documentation](https://docs.nvidia.com/cuda/cufft/index.html) | Whole-transform API; live documentation is not the experimental binary version |
| cufftdx | [NVIDIA execution methods](https://docs.nvidia.com/cuda/cufftdx/api/methods.html) | Thread/block components, ownership and synchronization contracts |
| vkfft | [Official repository](https://github.com/DTolm/VkFFT), README retrieved via raw.githubusercontent.com | Implementation scope and runtime GPU generation; citation is to software, not fabricated conference metadata |
| gpuntt | [Official repository](https://github.com/Alisah-Ozcan/GPU-NTT), README | Merge/4-Step implementation, current arithmetic vs older paper branch; manuscript measured adapter remains separately scoped |
| daofht | [Official repository](https://github.com/Dao-AILab/fast-hadamard-transform), README | CUDA/PyTorch interface, fp32/fp16/bf16 and shape scope |
| volta / ampere | [Volta guide](https://docs.nvidia.com/cuda/volta-tuning-guide/index.html), [Ampere guide](https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html) | Hierarchy/occupancy; Ampere asynchronous global-to-shared copies |
| flashfftconv2024 | [Author manuscript, Sections 3.1--3.2](https://arxiv.org/html/2311.05908v1); [official repository citation](https://github.com/HazyResearch/flash-fft-conv) | Author list and title; ICLR 2024 venue from the authors' BibTeX, distinct from the 2023 preprint; tensor-core decomposition, memory-aware fusion and cost model |
| quipsharp2024 | [PMLR proceedings and BibTeX](https://proceedings.mlr.press/v235/tseng24a.html) | Metadata and abstract; randomized Hadamard transforms in low-bit LLM quantization, used as application motivation |
| timeloop2019 | [Publisher-deposited metadata](https://doi.org/10.1109/ISPASS.2019.00042); [author manuscript](https://parashar.org/ispass19.pdf), Sections IV--VI | Full author list, venue and pages; workload/architecture/mapping separation and hierarchical spatial/temporal tiling; original 2019 paper, not every capability of current software |
| fftx2018 | [Author/project manuscript](https://www.spiral.net/doc/papers/hipc_2018.pdf); [publisher DOI](https://doi.org/10.1109/HiPCW.2018.8634111) | Metadata and method: composed plans, SPIRAL-backed generation, user computation and data-motion optimization; preliminary study, not a claim about current FFTX performance |
| vkfft2023 | [Publisher-deposited metadata](https://doi.org/10.1109/ACCESS.2023.3242240); [official project](https://github.com/DTolm/VkFFT) | Published author/title/volume/pages; portable complete FFT implementation. IEEE full text was blocked, so the manuscript avoids detailed claims about the paper's internal scheduling algorithm |
| gpuntt2023 | [IACR paper and PDF](https://eprint.iacr.org/2023/1410) | Two-author ePrint, Merge/Four-Step algorithms, stage partitioning and shared-memory reuse; preprint arithmetic and repository branches are distinct from measured adapters |
| gpuntt2025 | [Publisher-deposited metadata](https://doi.org/10.1109/ACCESS.2025.3570024); [official project citation](https://github.com/Alisah-Ozcan/GPU-NTT) | Three-author published follow-up, IEEE Access 13:87862--87883; metadata/title verified. Detailed method discussion is supported by the separately cited 2023 manuscript, not assumed from the 2025 title |
| hadacore2024 | [Author arXiv record](https://arxiv.org/abs/2412.08832); [full manuscript](https://arxiv.org/pdf/2412.08832) | Authors, title and preprint identifier; tensor-core chunks and shared-memory exchanges, low-precision evaluation. Cited as an arXiv preprint without inventing a publication venue |
| triton2019 | [Publisher-deposited metadata](https://doi.org/10.1145/3315508.3329973) | Authors, workshop and pages; only the broad tiled-language/compiler role is used. ACM PDF endpoint returned anti-bot HTML; detailed current Triton implementation claims are not attributed to this paper |
| tvm2018 | [USENIX proceedings and paper](https://www.usenix.org/conference/osdi18/presentation/chen) | Authors, venue/pages; compute/schedule separation, graph/operator optimization and search |
| ansor2020 | [USENIX proceedings and paper](https://www.usenix.org/conference/osdi20/presentation/zheng) | Authors, venue/pages; hierarchical sketches, evolutionary search and learned cost model |
| roller2022 | [USENIX proceedings and paper](https://www.usenix.org/conference/osdi22/presentation/zhu) | Authors, venue/pages; hardware-aligned tiles and micro-performance model |
| welder2023 | [USENIX proceedings and paper](https://www.usenix.org/conference/osdi23/presentation/shi) | Authors, venue/pages; tile-graph, graph connecting/subgraph tiling, traffic/footprint, code-generation layout and liveness |
| tensorir2023 | [Publisher-deposited metadata](https://doi.org/10.1145/3575693.3576933); [author manuscript v2](https://arxiv.org/pdf/2207.04296v2), §§2–5 | ASPLOS 2023, full author list/pages; local block signatures, independent outer transformations, tensorization and automatic optimization. The downloaded author version predates formal publication |
| eyeriss2016 | [Publisher-deposited metadata](https://doi.org/10.1109/ISCA.2016.40); [MIT author PDF](https://eems.mit.edu/wp-content/uploads/2016/04/eyeriss_isca_2016.pdf), §§IV–VII | ISCA 2016; dataflow taxonomy, primitive/mapping construction and resource-controlled movement/energy comparison. Spatial CNN accelerator, not GPU latency evidence |

The GS/CT matrix identities and index correspondence are derived in the paper
and exercised by `tools/check_theory.py`. No quotation from an external source
is used as a substitute for those derivations. URLs and full author lists are
kept in `references.bib`; institutional software authors are explicitly named
as projects rather than guessed individual authors.

The comparison rationale is maintained in [RELATED_WORK.md](RELATED_WORK.md).
All paper citations are BibTeX keys. The bibliography is a working reference
collection, not an exhaustive survey or a novelty certificate.
Before submission, refresh the nearest-work search and replace live software
citations with tagged versions where the experiments record them. Literature
coverage does not certify performance, supported lowering coverage, or matching
semantics for a newly mentioned external baseline.

Six complete exemplars are now stored locally, with pinned versions, source
URLs and SHA-256 in [literature/manifest.json](literature/manifest.json).
[Reading notes](literature/README.md) map concrete sections/figures to the
revision. Ansor, ROLLER and Welder were read from official USENIX PDFs;
TensorIR and FlashFFTConv from identified author versions; Eyeriss from MIT.
