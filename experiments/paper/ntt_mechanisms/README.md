# Figure 11: measured NTT mechanism controls

This package redraws the latest six-panel Figure 11 from frozen A100 measurements. It requires Python 3.9+ and Matplotlib; no GPU, CUDA, Nsight, network access, or manuscript files are required for the offline export.

From this directory:

```sh
python render_ntt_mechanism_profiles.py --output-dir output
python verify.py --output-dir output
```

The renderer exports `ntt_mechanism_profiles_20261118.{pdf,png,svg,csv,manifest.json}`. Verification compares every field of all 82 exported rows against the frozen source CSV and checks sanitized input hashes. The figure uses a 3.45 by 3.02 inch local paper style. No synthetic measurements are generated.

Panels a–d use three fixed radix-2 arithmetic primitives: Native (legend: cuButterfly), GPU-NTT, and PhantomFHE. These are arithmetic-provider controls inside the common shared-iterative organization template, not a comparison of complete vendor-library NTT implementations. Each split/resident pair fixes its arithmetic provider and suffix, 128 threads, and writer-aligned boundary. Grouping changes the shared tile and grid. The transform is natural cyclic forward word64 NTT, modulus 576460756061519873, N=2^16 and batch 16. Exact stage grouping is:

| Prefix points | Split prefix | Resident prefix |
| --- | --- | --- |
| 256 | 4 + 4 + 8 stages | 8 + 8 stages |
| 1024 | 5 + 5 + 6 stages | 10 + 6 stages |

The last group is the unchanged suffix. Panel a uses split/resident complete-call ordinary device latency; panel c normalizes resident latency to the same provider's resident-256 latency. Each configuration has three independent timing processes, each reduced to its process median. Summary bars use ratios of process medians; the corresponding process-index ratios and min/max are descriptive spread, not confidence intervals. The fixed-core analysis retains raw timing samples, exact-correctness counts, launch descriptors, all per-node NCU metrics, and complete-call DRAM totals.

Panel b measures complete-call DRAM read plus write bytes summed over the profiled stage kernels. Panel d is only resident prefix node 0's active-warp percentage, not complete-call occupancy. NCU captures are separate from ordinary timing; profiler times never supply performance ratios.

Panels e–f preserve the negative local-layout control: N=2^20, batches 4 and 16, 10+10 stages, 256 threads, radix2-shoup arithmetic, shared-iterative backend. Only `dataflow_layout` changes between linear and hermes-xor; arithmetic, stage partition, global I/O, natural input/output, and storage allocation remain fixed. The recorded XOR execution-group shared layout is writer-aligned. All twelve ordinary processes verify each requested batch exactly. Panel e divides each ordinary complete-call device latency by that batch's linear process median. Its summary is the ratio of layout process medians; independent processes are not recorded pairs, and the retained index-wise ratios are diagnostic only. XOR increases latency by about 4.7% and 7.4%. Panel f sums shared-load/store conflicts from the final two stage kernels covering one complete NTT in separate captures (IDs 2 and 3). XOR reduces load conflicts by about 87% but increases store conflicts by about 13–14%. Positive means reduction; negative means increase. Clocks and cache were uncontrolled.

`data/` contains the two dependency-complete sanitized analyses, the twelve compact ordinary layout timing CSVs, four compact layout NCU CSVs, original chart rows, and source figure manifest. Fixed-core raw samples and compact NCU counters are embedded in its analysis. `provenance.json` records each original input SHA256 separately from its bundled sanitized/adapted SHA256. Absolute checkout prefixes and device UUIDs were anonymized; profiler process banners were removed. Historical `upstream/` references and upstream source hashes identify provenance and are not dependencies supplied by this package. Renderer/style hashes refer to adapted local files; source-manifest hashes describe original inputs.

Offline redrawing does not reproduce GPU measurements. An exact fixed-primitive remeasurement route for panels a–d is provided in [gpu/README.md](gpu/README.md), with the original harness, exact local dependencies, portable runner/analyzer/guard, and hash-locked external arithmetic-source/license resolution. Compilation and CPU preparation passed and reproduce the original prepared bytes; GPU execution was not run. External vendor archives must be supplied separately. The e–f GPU layout runner is not included. New GPU results form a separate cohort.
