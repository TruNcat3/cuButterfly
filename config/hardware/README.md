# Reusable hardware profiles

Each directory is keyed by device model, compute capability and exact memory
capacity. For example, `NVIDIA-A100-80GB-PCIe-sm80-84974239744B` does not match an
A100 40GB, an SXM model, or a MIG device. UUIDs identify observations, not the
portable hardware class.

The A100 80GB PCIe seed contains verified capacity facts and historical measured
hardware rates. It does **not** contain a validated current-build cost model,
stage-service coverage or selected mappings. Its source hashes identify existing
local calibration and NCU records; publishing the seed required no new GPU run.

`hardware_model.json` uses the strict `cubutterfly-hardware-model-v1` schema.
Missing capacities must remain an explicit gap until acquired. In particular,
per-SM and per-CTA limits are different; use the opt-in shared-memory capacity
for the latter. Never infer registers per kernel from device register capacity.

To add another platform, run the common hardware calibration entry on that
target, inspect the resulting `hardware_model.json`, and place it under the
matching tag. Keep performance artifacts and their build/protocol provenance
separate. Do not relabel another platform's measurements. The CPU-only
`scripts/static_hardware_profile.py inspect <hardware_model.json>` command
checks schema and reports individual performance-layer readiness.

Installation fills missing seed files and preserves existing local profiles.
Updating an installed profile requires `--refresh-profile`. See the
[installation guide](../../docs/hardware_profile_install.md) and
[architecture decision](../../docs/adr/0001-static-profile-driven-planning.md).
