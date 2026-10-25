# FFT hybrid-dataflow validation on A100

This note records the fixed-candidate experiment used to validate the
long-chain FFT execution path. It separates three questions that must not be
collapsed into one result:

1. Is the online two-group candidate present in the compiled design space?
2. Does `stage-overlap` actually execute as an internal batch pipeline?
3. Does that pipeline win for the selected workload?

The measurements below use the A100 80GB PCIe device on GPU 2, FP32 C2C,
`N=65536`, contiguous out-of-place batches, and CUDA-event timing after plan
construction. The cuFFTDx build was configured with `CUBUTTERFLY_ENABLE_CUFFTDX=ON`
and `CUBUTTERFLY_ENABLE_XD_CUFFTDX=ON`. The ordinary `build-e13-shared`
build has the main cuFFTDx unit disabled, so its rank-one online candidates are
not executable and must not be used to judge the hybrid path.

## The candidate is present and the pipeline is real

The explicit mapping uses a `6+10` decomposition, FP32 cuFFTDx block cores,
`128x8` for the prefix and `256x16` for the suffix. With
`stage-overlap=true` and `batch_tile_count=4`, the lowered plan reports:

```text
dataflow_dispatch: batch-pipeline
dataflow_boundary: ring-buffer
stage_overlap: yes
execution groups: 2
workspace: 4 MiB
```

Nsight Systems confirms four prefix and four suffix kernels. They run on two
non-blocking streams, and the prefix kernel for tile `i+1` overlaps the suffix
kernel for tile `i`. The non-overlap mapping emits one full-batch prefix kernel
and one full-batch suffix kernel on one stream. Thus the experiment validates
the intended internal small-batch schedule rather than merely changing a
mapping label.

The inventory previously stopped the geometric tile loop at `tile < batch`.
That omitted the legal full-batch tile for every batch larger than one, such as
`tile=16` at `B=16`. The C++ registry and the installation-side Python
projection now retain the power-of-two interior tiles and add `tile=batch`.
The rebuilt A100 inventory contains overlap tiles `{1, 2, 4, 8, 16}` for the
`N=65536, B=16` cell, and an explicit `tile=16` run verifies the output
(`correct=1`, maximum absolute error `2.24e-4`). This expands the search space
without declaring the full-batch pipeline a winner.

The raw reports are:

- [`online_pipeline_n16_b16.nsys-rep`](../results/fft_trace_20261022/online_pipeline_n16_b16.nsys-rep)
- [`online_pipeline_n16_b16_stats.csv`](../results/fft_trace_20261022/online_pipeline_n16_b16_stats.csv)
- [`online_nooverlap_n16_b16.nsys-rep`](../results/fft_trace_20261022/online_nooverlap_n16_b16.nsys-rep)
- [`online_nooverlap_n16_b16_stats.csv`](../results/fft_trace_20261022/online_nooverlap_n16_b16_stats.csv)

## Batch and tile sweep

The table reports only kernel time. The tile is the number of transforms
advanced by each physical group before the next group is scheduled.

| N | B | cuFFT | online, no overlap | online, overlap tile 4 | best tested overlap tile | Interpretation |
|---:|---:|---:|---:|---:|---:|:--|
| 65536 | 4 | 0.0138 ms | 0.0177 ms | 0.0192 ms | 0.0192 ms (4) | too little work for a pipeline |
| 65536 | 8 | 0.0165 ms | 0.0248 ms | 0.0311 ms | 0.0311 ms (8) | whole-batch kernels are better fed |
| 65536 | 16 | 0.0237 ms | 0.0367 ms | 0.0565 ms | 0.0402 ms (16) | overlap is visible but launch and occupancy costs dominate |
| 65536 | 32 | 0.0492 ms | 0.0655 ms | 0.1123 ms | 0.0694 ms (32) | larger tiles recover occupancy; no-overlap remains faster |
| 65536 | 64 | 0.1055 ms | 0.1257 ms | 0.2182 ms | 0.1289 ms (64) | same trend at larger batches |
| 65536 | 128 | 0.2012 ms | 0.2231 ms | 0.4305 ms | 0.2270 ms (128) | the full-batch point is close to the serial path |
| 65536 | 256 | 0.4000 ms | 0.4146 ms | 0.8447 ms | 0.4181 ms (256) | no-overlap is the correct choice for this shape |

For `N=1048576, B=16`, the same `8+12` cuFFTDx decomposition gives
`0.8765 ms` without overlap and `0.9597 ms` with tile 4. Increasing the tile
to the full batch gives `0.8770 ms`, again showing that the overlap schedule
does not create a speedup for this mapping. These latency measurements alone
do not establish that either full-batch stage saturates the device.

The result is consistent with a scheduling tradeoff. A small tile creates more
independent stage boundaries, but each prefix and suffix launch has fewer CTAs
and the scheduler must honor event dependencies. The tested full-batch launches
are faster. The mixed-dataflow mechanism remains a valid option, while measured
selection should reject these slower instances. Under-filled stages, resource
complementarity and boundary stalls are hypotheses to investigate with traces,
not conclusions that follow from batch size alone. See the later
[scheduler and NCU follow-up](fft_trace_gap_analysis_20261022.md) for the
register-prefix mapping; it revises the earlier saturation interpretation.

## Why the default run did not select the pipeline

There are two separate selection states:

- In `build-e13-shared`, the main cuFFTDx unit is compiled out. The online
  processing-unit inventory is empty, so a rank-one cuFFTDx pipeline cannot be
  selected.
- In the cuFFTDx build, an uncalibrated `AUTO` request intentionally returns
  the first executable platform-prior candidate. For `N=65536, B=16` this is
  an online register-tile mapping with `stage_overlap=false`. The selection
  diagnostic says `no local timing estimate`; it does not mean the overlap
  candidates were omitted.

The calibrated install path must therefore use a cuFFTDx-enabled build when
that provider is available and persist the measured mapping for the exact
semantic workload. A static fallback must remain conservative because the
fixed-candidate data above shows that forcing overlap would regress this
shape.

## Scope of the conclusion

This experiment validates the implementation of the hybrid schedule and
identifies the current long-FFT boundary as a utilization and scheduling
tradeoff. It does not prove that hybrid dataflow is useless for FFT, nor does
it claim that every batch or stage shape should use overlap. The correct
research statement is that the framework exposes both whole-batch and
internal-batch organizations, and the hardware-calibrated selector chooses
between them per semantic workload.
