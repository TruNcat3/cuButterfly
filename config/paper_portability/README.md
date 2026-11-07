# Portable main operator comparison

`full.json` freezes the 66 semantic requests from the A100 main operator
comparison. `quick.json` selects 12 of those requests for a bounded migration
check. `scope.json` records their hashes and semantic source provenance. These
files contain request populations, not transferable A100 measurements or tuning
profiles. Preserve the hashes when moving the source package.

The main entry is `scripts/research_platform.py`, configured by
`config/platforms/pro6000_server96.json`. Follow the
[PRO6000 Server96 runbook](../../docs/hardware/pro6000_server96.md) for actual
installation, baseline builds and GPU commands.

| Action | Work | GPU access |
| --- | --- | --- |
| `plan` | Print exact delegated commands | None |
| `doctor` | Check environment metadata and compiler target | None |
| `build` | Build/install shared library and Python API; compile-probe MathDx | None |
| `prepare --mode quick/full` | Bind manifests, create CPU references, precompile declared proposals | None |
| `run --mode quick/full` | Verify device; calibrate in results directory, compare and summarize | Required |
| `report --mode quick/full` | Regenerate summary from recorded measurements | None |

Quick12 uses calibration/search budgets 16/32, full66 uses 32/512; both request
24 stage calibration points. These are finite sampling budgets. Quick is a
subset check, and full means the main66 population only. Neither is an exhaustive
search or a certificate for all paper tracks. Reports keep missing baselines,
unsupported requests and accuracy review visible.

The full population contains FFT17, FWHT13, NTT13, subset-zeta8, superset-zeta8
and structured-2x2 seven. It spans ranks 1/2/3/4/8, with 40 forward and 26 inverse
requests, all out-of-place. Cyclic90, public FP16/FFT-L nine requests, fixed-core
ablations, mechanism profiles and applications remain separate follow-up tracks.
BF16 migration is deferred.

Create new target-local builds, caches, preparations, profiles and results.
Rebind the semantic population to the new shared library; do not reuse an A100
build-bound manifest, calibration checkpoint or timing receipt. The portable
[research campaign](../research_campaign/manifest.json) is a separate, broader
historical semantic inventory; its counts do not replace this main66 contract.

The build installs the library and conda lookup hooks. Research `run` calibration
stays under the mode's results directory; it is not published to the conda
default profile tree. See the runbook's explicit environment binding for subsequent
applications. A successful report is not evidence of a calibrated default install.
`doctor` also checks Torch's compiled-in SM120 target without initializing CUDA;
an older Torch build without that target blocks migration even with CUDA 12.8 installed.
