# Mature core reuse correction, 2026-10-14

Deployment update: the user authorized adopting this correction directly into
the continuing development campaign. The main worktree and shared conda
environment now contain it, built with the original xD cuFFTDx/VkFFT options
enabled. The earlier isolated validation below remains a historical record.
The installed FP64 N65536 B64 EPT2 mapping passed full-batch verification.
Continuation uses `--adopt-current-build` and `--resume-build-priors`, retaining
old observations with their sources and confirming selected mappings anew.
See [current campaign status](calibration_repair_20261014.md).

Revision `0.9.3.11-provider.3` corrects the interpretation of a provider capability
gap. The research objective is to organize replaceable processing cores through
hierarchical hybrid dataflow. Reimplementing arithmetic for every missing vendor
parameter combination is not a prerequisite for supporting the user's transform.

## Existing reusable core and the actual gap

The grouped FFT suffix already instantiates a cuFFTDx `Block` core and calls its
`FFT::execute` method. cuButterfly owns its surrounding subgraph assignment,
writeback exchange, output layout and normalization. This is direct reuse of
the mature core through a template, not a newly written FFT arithmetic kernel.

On the installed SM800 provider, FP64 local N1024 has EPT 2/4/8/16 and local
N4096 has EPT 4/8/16. EPT32 is absent for both. That does not make an FP64
transform with either subgraph size unsupported. The earlier native suffix
filled an exact physical mapping, not missing FFT semantics; it also lost to
the cuFFTDx controls in the preserved measurements.

Two contracts must remain distinct. Automatic planning keeps the user's
operator, precision, direction, length, batch and layout unchanged while
searching legal core and mapping choices. Explicit mapping replay must honor
the named provider and EPT, or return the specific implementation gap. It must
not replace a requested configuration silently.

## Changes

- Enumerate the online register FFT's cuFFTDx suffix over power-of-two EPT values
  up to the local extent, checked against the installed provider and the existing
  thread/shared-memory geometry constraints. Preserve the historical preference
  16/32/8/4 before the newly exposed legal choices. Provider checks run once per
  local extent, outside the prefix-organization product.
- Include EPT2, previously absent from this template's automatic enumeration.
  Known missing provider combinations remain in the audit. Unknown provider
  capabilities remain eligible for compilation; they are not treated as proof
  of support or absence. Resource checks and compilation remain necessary.
- Stop automatically enumerating the newly added `suffix_core="native"`.
  Preserve its explicit mapping, replay, calibration identity and negative
  measurement records. Existing native prefix and factor templates are unchanged.
- Record the mature-core priority in the project-wide research principles and
  integration strategy. This rule applies to all butterfly families.

This correction does not expose cuFFT's private arithmetic kernels, expand every
other template's parameter domain, or claim that an unmeasured new EPT is faster.
The factor template's existing parameter sampling and xD inventories retain their
separate scope. Existing cuFFTDx suffix controls and their raw timings are in
[the previous bounded experiment](native_suffix_core_20261014.md).

## Validation and deployment boundary

The affected checks completed on this revision:

- Seven C++ provider/candidate contract tests passed, including actual candidate
  generation, runtime provider identity, unknown capability handling and legal
  parameter coverage. An additional 72 CPU plumbing, compilation, capability and
  calibration-seed tests passed after updating one obsolete test that expected
  automatic native-suffix enumeration. The original failure log is preserved.
- The real `plan_registry.cpp` compiled with the existing Release build flags.
  The candidate tests compile the actual source function with CPU hardware stubs;
  they do not substitute a second implementation of its enumeration.
- Four cuFFTDx modules compiled on CPU before GPU acquisition: FP32/FP64, EPT2/16,
  local N1024, two columns, complete N65536 B64. The existing bounded checker now
  accepts an explicit case list, avoiding a new experiment-specific runner.
- On A100 40GB UUID `GPU-2b759faa-3e58-7106-10aa-ea6e387c3f8e`, all 144 checks
  completed under the exclusive-process guard. Forward, inverse with and without
  normalization, strides 1/2, padded batches, combined/group launches and three
  input patterns are covered. Outputs were finite, padding unchanged and exact
  zero/origin-impulse checks passed. Maximum relative L2 errors were 2.42e-7
  (FP32) and 4.50e-16 (FP64), versus matched cuFFT errors 2.71e-7 and 4.92e-16.
  These are observations against same-precision CPU references, not new thresholds.

The FP64 complete two-group CUDA Graph controls measured 507.96 us for EPT2 and
406.12 us for EPT16. Both use the same surrounding prefix and chunk-2 writeback;
EPT2 uses 1,024 threads and EPT16 uses 128. This is a comparison of explicit core
mapping variants, not an automatically selected result or an organization-only
ablation. For EPT2, chunk 2 covers the whole tile; automatic enumeration represents
that extent with its whole-exchange option and also enumerates chunk 1. These
measurements validate core reuse and do not claim a speedup from adding EPT2.
Compilation, setup and CPU reference generation are outside the graph timings.

Evidence is in `results/provider_contract_20261014/provider_suffix_controls`:
`prepared.json` identifies sources, modules and references; `gpu/status.json`
records the guard; `gpu/release/run.json` preserves errors, resources and timings.
The original slower native-core records remain unchanged. Preparation and replay:

```bash
python tests/check_native_suffix.py --prepare results/provider_suffix_check \
  --cases tests/data/provider_suffix_controls.json
# Run only on an exclusive device, with the existing GPU guard.
CUDA_VISIBLE_DEVICES=<gpu-uuid> \
  python tests/check_native_suffix.py --run results/provider_suffix_check
```

At the initial validation boundary, the implementation remained isolated and
the `0.9.3.11-calibration.3` experiment retained its original executable sources,
library and measurements. The candidate function and JIT execution path were
checked separately; that validation did not include relinking or installation.
The deployment update at the top of this document records the subsequent full
main-tree build, installation and explicit development continuation. Earlier
measurements retain their provenance while the new search scope is evaluated.
