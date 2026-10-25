# cuFFTDx provider capability contract

Revision `0.9.3.11-provider.1` checks installed core implementations before
selecting or compiling dynamic FFT mappings. It does not change arithmetic,
precision, transform size, batch, or the definition of the butterfly space.

## Classification and shared source

| Classification | Evidence | Action |
| --- | --- | --- |
| Invalid mapping geometry or request | Existing semantic, divisibility and resource checks | Reject with the relevant geometry/semantic reason |
| `provider-implementation-missing` | Completely parsed provider table lacks the requested local size/EPT/direction | Exclude this provider combination; retain the original transform and other cores |
| `provider-capability-unknown` | Missing/unrecognized table, unknown SM, runtime package override or changed package contents | Preserve the candidate and let actual compilation establish executability |
| Supported provider entry | Matching implementation in the installed table | Continue geometry, resource, compilation and numerical checks |

`scripts/cufftdx_capabilities.py` reads the installed MathDx C2C block records
and declared SM aliases. The lookup uses SM, precision, local size, explicit
EPT and direction. CMake generates the CPU C++ table from this same parser;
Python JIT queries it directly. Current JIT modules instantiate both directions,
so both must be available. This is the non-folded C2C/default TRP contract.
Forced TRP settings make the C++ query unknown. FFTsPerBlock suggestions do not
limit the query; launch geometry remains a separate check.

This removes the two SM800 FP64 EPT32 combinations for local N1024/N4096.
Their EPT16 alternatives remain legal. EPT32 remains available for supported
FP32 and small FP64 cores, including FP64 N32. Native register factors and
native prefixes do not inherit cuFFTDx's capability restrictions.

## Integration and audit

Dynamic register-suffix and cuFFTDx factor candidates use the provider check.
Explicit register mappings validate before module preparation; the common
module preparation path checks imported factors as well. Python validates
before invoking NVCC or creating a cache directory, and direct rendering
entry points use the same check.

The public C API returns `NOT_SUPPORTED` for a known missing provider
implementation and retains the detailed error. Invalid parameters still return
`INVALID_VALUE`. Neither classification changes the requested transform.

The public C++ candidate function has an optional `std::string*` audit argument;
the original one-argument API remains available. The benchmark can export the
same compact JSON without timing kernels:

```bash
CUBUTTERFLY_COMPILE_MODE=research build-provider/cubutterfly_bench \
  --operator fft --precision fp64 --logN 16 --batch 64 \
  --candidate-capability-audit
```

The audit records excluded or unknown provider keys, source fingerprint and
whether the compiled table matches the runtime package path and contents. It
does not claim to list every geometry/resource rejection. An override, upgrade
in place or missing source file defers the C++ capability decision to Python's
actual-package check. The generated table binds each parsed source's canonical
path and FNV-1a64 digest; SHA256 provenance remains in its metadata. Verification
runs once per candidate enumeration, outside kernel execution and timing.

The parser and MathDx `*.hpp.inc` files are included in JIT/broker hashes;
previous hashes covered headers but missed these database fragments. CMake
also tracks provider inputs and the generated table. Installation includes
the parser beside both installed calibration tools and JIT templates.

## Validation and experiment isolation

- 90 affected CPU tests passed, including Python/C++ table consistency, actual
  candidate-function execution, both directions, provider overrides, unknown
  capability, native-core retention, install dependencies, cache identity,
  upgrades at the same path and public C API error classification.
- CMake configuration and the affected `plan_registry.cpp`, `jit_module.cpp`,
  `c_api.cpp`, `fft_register_dispatch.cu`, and `cubutterfly_bench.cpp` objects compiled.
- Both original rejected mappings stopped before NVCC/cache creation. Keeping
  their two-column geometry and selecting EPT16 allowed both actual CUDA modules
  to compile. These are compile checks, not new GPU accuracy or timing results.

The implementation lives in `/home/wt/git/cuButterfly-capability-IPwAsa`, an
isolated worktree containing the current research source snapshot. The original
campaign under `results/install_compare_repair_20261014` retains its
frozen sources and binaries. Do not replace that source tree, its installed
templates, or its broker while it is active. Adopting this revision requires
its own build/cache identity and compatible new calibration; previous measured
results must not be relabeled as measurements of this revision.

The local compile evidence is `results/provider_contract_20261014/compile_checks.json`.
Full-library rebuilding and installation into the active conda environment were
not performed during the running campaign.
