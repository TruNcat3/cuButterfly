# Theory and implementation alignment audit (2026-09-10)

This is the historical audit before the unified framework refactor. Several
limitations below have since been repaired; current implementation and measured
evidence are tracked in [framework_refactor_progress.md](framework_refactor_progress.md).

The implementation is **partially aligned**, not a complete executable search
of the theoretical space. Adding isolated winning mappings concealed this gap.
This audit separates the paper's claims, the abstract design space, compiled
units, executable plans, calibration coverage and measured selection.

## What the paper actually requires

Source: `../APPT26/samplepaper.tex`, especially the hybrid-dataflow discussion,
on-chip architecture, Figure DAP2, and Algorithm 1 (fragmentation mapping).
The paper describes a U280 NTT architecture; the repository generalizes its
stage/data space-time decomposition to other butterfly operators and GPUs.

* Stage and data unfolding are independent. The local arithmetic implementation
  is one axis, not a reason to exclude other mappings of the same graph.
* The partial-stage pipeline operates over multiple iterations. Its results
  stay in URAM for subsequent iterations; the intermediate live domain, its
  capacity and lifetime matter separately from the size of a local compute unit.
* Algorithm 1 maps `bank=(i xor floor(i/Npart)) mod (2p)`, `offset=floor(i/(2p))`.
  This is a bank/offset layout for a specific access schedule. It requires
  agreement between producer stores and consumer loads, and enough ports.
* The FPGA discussion assumes a pipelined core and matching memory initiation
  interval. GPU warp issue, occupancy, barriers and shared-memory bank widths
  need an explicit mapping and measurements; II=1 does not transfer automatically.

The paper demonstrates a resource/throughput trade-off, not a theorem that any
finite GPU template set must outperform every vendor implementation at every
size. Its conclusion explicitly leaves the optimal resource/throughput trade-off
as future work. A best measured point is optimal only among the points measured.

## End-to-end audit

| Layer / theoretical requirement | Finding before this change | Current status |
|---|---|---|
| Graph and independent architecture/core/layout/realization axes | `butterfly_architecture_space.json` and `butterfly_design_space.py` describe these separately | Description is substantially aligned; this alone does not establish executable coverage |
| Ordered partitions and independent local mappings | Large install cases used handwritten splits and launch parameters | Two-segment FFT product now comes from the linked library's actual local-unit availability; prefix/suffix threads and EPT vary independently |
| Processing-unit availability | FP64 code existed but install calibration omitted it; probing one EPT pair incorrectly suggested an entire partition was unavailable | `--list-processing-units` queries the same FP32/FP64 availability functions as dispatch; supported local pairs are composed rather than inferred from one failed launch |
| Plan validation versus unit validation | The old scalar minimum of five local stages rejected a compiled cuFFTDx 4+12 mapping | Imported unit ranges now use their own availability predicates; scalar constraints only apply to scalar units |
| Measured candidate to runtime mapping | Candidate strings were a second handwritten C++ mapping table | Newly generated selectors serialize the resolved configuration, including logical partitions, physical groups and boundaries; names are opaque identifiers |
| Workload semantics | Exact-match keys omitted direction, accumulation and matrices; a legacy structured selection could replace the caller's matrix | Generated match predicates check numeric/layout semantics and matrices; generated mappings preserve caller coefficients |
| Target identity | Model name/SM checks without memory capacity; V100 branch bypassed local points | New records fingerprint model/SM/memory; local Butterfly points take precedence, including on V100; historical fallback remains explicitly V100-only |
| Internal candidate versus external baseline | cuFFT could become a local selector winner | Whole cuFFT rows remain reference measurements and cannot be promoted as cuButterfly candidates |
| Width and boundary cost | FP32/FP64 FFT both charged as 8-byte values; real transforms also got FFT width | Storage widths now distinguish scalar/complex and precision; traffic feature uses physical execution groups; prefix/suffix thread and EPT features are separate |
| Theory demand versus actual cost model | `unfolding_rank.csv` is an analytical demand table; regression model is a separate empirical fit | Still not a calibrated, per-hierarchy pipeline/port/occupancy model; do not interpret it as a proof or universal search oracle |
| Search coverage versus search budget | Small smoke set could be described as calibrated design space | `search_coverage.json` lists the complete enumerated two-segment product, budget omissions, screening results, failures and confirmed finalists |
| All operators and all numeric regimes | NTT/FWHT/zeta/structured had selected smoke points and separate search scripts | The installer now enumerates the in-tree FWHT, zeta and structured-2x2 families through the same candidate/measurement path; NTT remains on its independent modulus/backend contract and is still reported separately |
| Arbitrary multi-segment architecture | Abstract compositions and FP32 multi-segment launch path exist | Not enumerated by the new install search; FP64 runtime explicitly restricts online execution to two segments |
| Shared-resident iteration across subgraphs | Unified IR is a projection of existing backends; resident/ring NTT routes exist separately | `shared-iterative` now lowers dependency-closed radix-2 groups through double-buffered shared memory, including writer-aligned stores. The executable implementation is deliberately limited to two groups; larger partitions are rejected until their prefix/index contract is implemented |
| APPT producer/consumer layout contract | Experimental permutation helper has a disabled stage-contract gate and prior correctness failures | No claim of a working generalized Algorithm 1 GPU lowering. An invertible permutation alone does not establish bank-conflict freedom or an executable iteration schedule |
| Build-time candidate selection | A100 build still uses `config/v100_fft_codegen.json`; direct local dimensions 11/12 have separate dispatch constraints | Still a build-policy gap. Runtime inventory now makes the actual compiled set visible, but does not manufacture uncompiled templates or remove the historical build defaults |

## What changed in installation

`config/install_search_workloads.json` specifies **workloads**, not winning
kernel parameters. For each workload, the installer enumerates all pairs of
compiled local units whose stage counts sum to logN. It crosses independent
prefix/suffix threads and EPT with table/recurrence coefficients and supported
FP64 shared layouts. This projection uses direct-strided global boundaries.

By default the installer screens a deterministic, stratified subset of every
configured workload. FFT candidates are formed from the compiled cuFFTDx
local-unit inventory; FWHT, zeta and structured-2x2 candidates use the shared
in-tree Butterfly realization families. Three valid finalists are confirmed
using the requested operator trial/warmup/repeat settings. Only confirmed
correct measurements enter the generated selector. GPU exclusivity is checked
around each measurement. This is bounded sampling, not exhaustive search or
model-based pruning. `--search-budget 0` measures the entire enumerated
product, but still does not cover uncompiled theoretical realization families.

The current selector is remeasured as an incumbent for every workload; its
resolved configuration participates alongside the new candidates. This retains
an existing better internal mapping without hardcoding its parameters into the
new enumerator. An incumbent that resolves to cuFFT stays a reference.
Explicit `--resume-search` reuses completed search measurements only for an
unchanged binary and device model/SM/memory, with sufficient trial conditions.

The regression is fitted after these measurements and exported as an aid for
subsequent analysis. It does not rank or prune the current installation search.
This distinction is intentional and explicit in the coverage report.

## Remaining work should proceed by contracts, not winning-size patches

1. Make the abstract candidate schema authoritative across code generation,
   operator lowerings, search and runtime. Preserve separate states for
   described, resource-feasible, compiled, executable, correct and measured.
   Remove the remaining historical compile-selection defaults through this
   schema, with a deliberate compile budget and a supported-target policy.
2. Implement stage/data temporal iteration and intermediate lifetimes at each
   GPU hierarchy. Validate the producer/consumer bank-and-offset schedule before
   admitting an APPT shared-resident realization. A logN20 FP32 transform has
   8 MiB of values; a small local unit does not by itself prove that its entire
   dependent intermediate domain fits in one CTA. A tiled schedule needs its
   own live-state and synchronization proof.
3. Calibrate arithmetic and transport by precision, unit, hierarchy, transaction
   pattern and synchronization scope. Fit and validate on held-out workloads;
   use prediction uncertainty to drive measurements. Report which dimensions
   the model cannot distinguish instead of calling its minimum a global optimum.
4. Extend the same candidate/lowering/measurement contract to all Butterfly and
   NTT families. Comprehensive baselines must show the selected mapping and its
   search coverage, with external libraries kept separate.

No new kernel performance claim follows from this audit or from the host tests.
The accompanying A100 calibration artifacts contain the actual measurements.
