# Contributing

cuButterfly is a research artifact. Contributions are welcome when they keep
mathematical semantics, architecture mapping, processing units, and external
baselines distinguishable.

## Development Setup

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-11.8/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=70
cmake --build build -j
cmake --build build --target test
python3 scripts/check_repository.py
```

Use the CUDA architecture and compatible compiler pair for your GPU. Do not
overwrite a checked-in hardware profile with measurements from another card.

## Change Categories

### Processing Unit

Record operator, precision or modulus, supported length, radix/stage group,
coefficient form, native layout, block/warp requirements, registers, shared
memory, and numeric error. Add standalone reference tests before performance
claims.

### Mapping Or Kernel Family

Explain which fields of `(Us,Ts,Ud,Td,Hs,Rs,Rd,L,F,Q)` change. Report state
residency and handoff boundaries explicitly. A reduced launch count alone is
not an architectural explanation.

### External Baseline

Pin repository revision, license, local patch, GPU, CUDA version, workload
semantics, native output layout, and timing boundary. Keep external source trees
outside the repository or under ignored `external/`.

### Performance Result

Include raw CSV, the reduction command, and a concise report. Match operator,
length, batch, direction, normalization, precision/modulus, placement, layout,
warmup, repeats, and resident/end-to-end boundary. Label diagnostic runs that
intentionally produce invalid output.

## Correctness And Review

Before submitting a change:

1. run both CTest targets;
2. verify each selected benchmark point against the CPU reference;
3. run Compute Sanitizer for new synchronization or layout code;
4. run `python3 scripts/check_repository.py`;
5. update the feature matrix and relevant result report;
6. state unsupported combinations and remaining risks.

Avoid unrelated formatting of raw measurement files. Preserve user data and
third-party provenance.

## Commit Scope

Keep source changes, generated-design specification changes, and new evidence
easy to audit. Generated build-directory files are not committed. Graph source
and reduced profiler CSV are preferred over machine-specific binary reports.
