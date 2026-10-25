# E13 FNO execution

The FFT application uses NeuralOperator commit
`00b7d86f8d74ff0af55da53eb585fe26df9c71f0` and the public Darcy dataset.
Execution records are in `results/paper_profile_rc2/end_to_end/`; the original
`end_to_end.json` remains the preparation contract. The current run state is
`e13_progress.json` in the result directory.

## Frozen application conditions

Both variants use the same trained checkpoint, 100 held-out samples, FP32,
four FNO layers, 24 hidden channels, grid positional embeddings, projection
ratio 2, and 16-by-16 modes. Inputs are preloaded on the GPU. The two payloads
are 64-by-64 with batch 8 and 128-by-128 with batch 4. The final 64-by-64 batch
contains four samples and is retained in both variants.

These are source-based application experiments, not a byte-for-byte run of
the upstream tutorial. Our preparation wrapper trains on 1,000 samples for
300 epochs with H1 loss, PyTorch AdamW, learning rate 0.002, weight decay
0.0001, and no scheduler or input/output normalization. The tutorial uses a
smaller resolution, different optimizer settings and a scheduler. Checkpoint
configuration and hashes are retained. The training log's H1 scalar includes
the loss's batch-sum reduction and must not be interpreted as mean held-out
error. The benchmark independently computes mean held-out L2 and H1 errors.

The native timed model uses the upstream default `enforce_hermitian_symmetry=True`.
The treatment implements the same inverse through full 2-D Hermitian spectrum
reconstruction, instead of the upstream split inverse. Protocol v4 checks
treatment output against that default before any measurement. It requires every
treatment forward to replace four
`rfftn` and four `irfftn` calls and rejects unsupported calls rather than
silently falling back.

This distinction matters at 128-by-128. The earlier non-default native
`irfftn` path differed from the explicit-symmetry reference by 0.03768 in the
application, matching the upstream warning about size-dependent boundary
handling. Protocol v3 stopped before timing that case. Its failure report is
retained as `fno_darcy_128_main.trial0.json`; v4 uses the correct default
baseline and passes the frozen tolerance without relaxation.

The already frozen checkpoints were trained with the earlier non-default
inverse setting. They are reused identically by both v4 inference variants;
the reported held-out quality is measured using the default inference
semantics, not inferred from training logs. The training script now preserves
the upstream explicit-symmetry default for future checkpoints. This change
does not rewrite or relabel the existing checkpoint configuration or hashes.

cuButterfly currently implements this real transform through full complex
FFTs, half-spectrum packing and Hermitian reconstruction. The v8 adapter
revision fuses real packing with forward scaling, fuses Hermitian expansion,
reuses intermediate buffers, and writes the packed boundary directly into the
plan scratch region for the serialized FNO inference stream. That
work is part of the timed application. It is still not a native reduced-work
R2C/C2R kernel and there is no presumption that an operator-level C2C speedup
survives this adapter. Plans are reused, and execution follows the caller's
current CUDA stream. A single adapter's workspace must not be used concurrently
by two streams.

TF32 is disabled for both inference variants. Protocol v2 checks output
maximum absolute error and relative L2 against the predeclared `1e-4` limits
before collecting latency. Protocol v5 runs three fresh processes with balanced
variant ordering, five warmups per batch and ten measured forwards per batch.
The reports retain all samples, trial medians, synchronized host wall time and
CUDA-event elapsed time. The latter includes dispatch gaps within a complete
forward and must not be called the sum of kernel durations. Compilation,
training, data I/O, plan creation, online candidate measurement and the first
lazy CUDA/module preparation are outside inference timing. The report records
their setup cost and plan-selection source, and rejects a run if a new plan is
created after timing starts. Trace fractions and isolated adapter overhead have
not yet been measured.

The environment variable `CUBUTTERFLY_E13_ALGORITHM_POLICY=measure` enables
online candidate selection for research runs. Candidate measurement remains a
setup activity and is never folded into the steady-state application latency.
The end-to-end runners expose the same choice explicitly through
`--algorithm-policy default|measure`; the selected policy and every plan's
selection source are retained in the report. Use `measure` for a new hardware
or workload point, then replay the resulting registry/cache for the final
steady-state comparison. A report that still says `static_model` is a
selection miss and must not be presented as the calibrated application result.
For compact rank-2 FP32 FFT axes, this policy uses a bounded comparison of the
temporal-radix4 and vendor C2C candidates so the physical flattened batch is
included in the decision without launching the full generic search. The
default policy is bounded and uses the library's static/cache selection.

## Reproduce after preparing sources and data

Use `/home/wt/yes/envs/cubutterfly`. Verified archives are extracted under
`results/e13_assets/data/extracted64` and `extracted128`.
Their MD5 values are respectively `49027fc1972647915f6c0893a79bbc59` and
`43e85ec9bff754d0f6d3925e26d5a8ac`.

Build the shared C API with `CUBUTTERFLY_BUILD_SHARED=ON` and CUDA architecture
80 in `build-e13-shared`. The artifact is `libcuntt.so`. The result records
its SHA256 as well as the runner, adapter, checkpoint and test tensor hashes.
Select an exclusively available GPU by UUID, for example:

```bash
export CUDA_VISIBLE_DEVICES=GPU-2971e500-d3f6-d7e6-e5f2-cb67a06fff02
/home/wt/yes/envs/cubutterfly/bin/python results/e13_assets/validate_real_fft_adapter.py \
  --library build-e13-shared/libcuntt.so \
  --output results/paper_profile_rc2/end_to_end/fft_adapter_correctness_v5.json
/home/wt/yes/envs/cubutterfly/bin/python results/e13_assets/run_fno_end_to_end.py \
  --data-root results/e13_assets/data/extracted64 --resolution 64 \
  --n-test 100 --batch-size 8 --trials 3 --warmup 5 --calls 10 \
  --library build-e13-shared/libcuntt.so \
  --checkpoint results/paper_profile_rc2/end_to_end/fno_darcy_64_checkpoint_official.pt \
  --output results/paper_profile_rc2/end_to_end/fno_darcy_64_v8.json
```

For 128-by-128, use `extracted128`, resolution 128, batch 4 and
`fno_darcy_128_checkpoint_main.pt`. Do not rerun training while the existing
`run_fno_128_when_ready.sh` pipeline is active. Initial random-weight,
short-training and MSE-checkpoint reports remain diagnostic history; they
are not application evidence for the frozen H1-checkpoint runs.
