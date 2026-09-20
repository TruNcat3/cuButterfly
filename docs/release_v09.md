# v0.9 Framework Release

v0.9 separates the reusable cuButterfly framework from the historical V100
research baseline. The framework has five explicit layers:

1. Templates contain CUDA kernels, processing-unit adapters, and code generators.
2. Platform profiles describe GPU capabilities, compiler targets, and measured hardware facts.
3. Parameter spaces describe legal decompositions, layouts, precision, and launch choices.
4. Search and calibration produce correctness-checked measurements and fit local cost models.
5. Runtime selection promotes only exact, fingerprinted local points or generated fallback entries.

The authoritative map is [v09_architecture_manifest.json](../config/v09_architecture_manifest.json).
The configure step validates the map through
`scripts/check_v09_architecture.py`, so a new layer cannot silently drift away
from the build.

The old `config/v100_*` files remain intentionally unchanged as compatibility
and evidence inputs. They describe the frozen V100 profile; they are not the
generic framework defaults. A new GPU is integrated through the install-time
profile and calibration workflow, which records the device name, compute
capability, compiler target, candidate identity, correctness, and timing in the
local profile directory.

For large transforms, v0.9 also treats launch mapping as platform data. The
A100 logN=18 and logN=20 points are measured during calibration because their
CTA/EPT choices materially change performance. The runtime selector cannot use
them on another GPU unless that GPU has its own matching calibration points.

This release does not claim that every processing unit is novel. Its boundary
is the reusable mapping framework: graph and operator contracts, parameter
spaces, lowering, hardware calibration, search, and runtime promotion.
