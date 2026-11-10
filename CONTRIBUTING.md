# Contributing

cuButterfly studies a common butterfly dependency structure, resource
characterization and hierarchical hybrid dataflow, with replaceable cores,
subgraphs and boundaries. Search, calibration and code generation support that
method; they are engineering mechanisms rather than separate research claims.
Read [research principles](docs/research_principles.md) before method or experiment
changes, and [getting started](docs/getting_started.md) for installation.

## Development environment

Use a Python environment and CUDA compiler compatible with the target, following
[getting started](docs/getting_started.md). Reuse a compatible existing environment
and build. Preserve an explicitly selected compile policy; research workflows can
use `CUBUTTERFLY_COMPILE_MODE=research` for implemented on-demand specializations.
Hardware-specific setup is indexed in [hardware profiles](docs/hardware/README.md).
A documentation change needs no installation, recalibration or GPU run.

From the repository root, the CPU repository check is:

```bash
python scripts/check_repository.py
```

It checks source-release links, JSON, Python syntax and citation fields. It reports
explicitly unbundled historical artifacts; it does not validate their contents or
experimental conclusions. Use `--strict-local-artifacts` only when checking an
archive that is meant to include that historical inventory.

## Match validation to the change

| Change | Boundary to preserve | Matching verification |
| --- | --- | --- |
| Documentation | Current API, commands, links and evidence scope | Repository check; inspect parser/source for command examples |
| Semantics or workload | Operator, direction, normalization, precision/modulus, shape, batch, strides, placement and order | Affected descriptor/reference tests; CPU semantic tests; GPU numerical cases when execution changes |
| Organization | Dependencies, segmentation, residency, spatial/time reuse, layouts, handoffs and overlap | Affected mapping legality and GPU correctness tests; paired measurements with the same core for performance claims |
| Local core/provider | Arithmetic, support domain, local exchange, scratch/register demands and numerical behavior | Core/reference tests and affected public execution cases; compiler resource evidence; matching performance cases |
| Hardware/profile | Capacity facts versus sampled service rates, exact device/build/protocol provenance | Profile compatibility tests and resource checks; explicit target calibration only when relevant inputs change |
| Search/generation/cache | Candidate legality, implemented lowering, compilation, measured coverage and selection source | Affected CPU enumeration/policy/cache tests; affected GPU candidates if execution changes |
| External baseline or result | Matched semantics, precision and timing boundary; pinned source/license/patch | Matched correctness and performance cases, raw trials and reproducible reduction |

Useful targeted entries, selected according to the affected contract:

```bash
# CPU ctypes boundary with a mocked library; no installation or GPU needed.
python -m unittest discover -s tests -p test_public_frontend.py
# Static profile and installer policy tests; no calibration.
python -m pytest -q tests/test_static_hardware_profile.py tests/test_install_profile_policy.py
# Existing compiled CPU semantic checks.
ctest --test-dir build --output-on-failure \
  -R '^(semantic_contract_cpu|cpp_descriptor_cpu)$'
# Discover available execution checks before choosing the affected GPU cases.
ctest --test-dir build -N
```

Use the actual build directory instead of `build` where necessary. For CPU shared
library integration, `tests/test_semantic_api_cpu.py` accepts
`CUBUTTERFLY_TEST_LIBRARY=/absolute/path/to/libcuntt.so` under pytest. For changed
synchronization or layout code, run relevant Compute Sanitizer checks on the
target GPU. Record unsupported combinations and unresolved failures. Expand
validation when failures or shared-contract changes justify it; full release or
experiment matrices remain separate, explicitly scoped milestones.

## Evidence and review

Describe which layer changed and why. For organization changes, identify the
actual subgraphs, state residency and handoff boundaries; launch count alone does
not explain the method. A core may be replaced without changing transform
semantics. A missing provider configuration, uncompiled candidate and unmeasured
candidate are different states and must remain distinguishable.

Performance reports retain raw trials, reduction commands, hardware UUID/model/
memory, compiler/build, compile policy, actual mapping, warmup/repeats and timing
boundary. Separate automatic-library comparisons, fixed-core organization
experiments and end-to-end evidence. Preserve negative results and restrict claims
to measured workloads; a finite search does not prove global optimality.

Keep source, generated specifications and evidence easy to audit. Preserve
unrelated edits, user data and third-party provenance. Keep external sources in
ignored `external/` or outside the repository. Do not commit generated build
files or overwrite another GPU's profile. Update affected capability documentation
when support changes, and identify unavailable local artifacts explicitly.
