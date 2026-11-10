# Platform Tuning Rules

Platform tuning is an optional source of candidate priors for the public
planner. It is not a second registry and it does not turn an unmeasured
mapping into a calibrated result.

## Interface

The interface is declared in `src/platform_tuning.hpp` and implemented in
`src/platform_tuning.cpp`:

- `platform_candidates(const ButterflyConfig&)` and
  `platform_candidates(const PlanConfig&)` add matching rule mappings to the
  normal executable candidate inventory.
- `match_platform_rules(document, hardware, semantics)` is the pure matching
  interface used by the CPU contract test and by the typed candidate helpers.
- `platform_tuning_identity()` returns a single-line, collision-free hex
  encoding of the effective rule-file contents, so a changed file has a new
  selection namespace without corrupting the line-oriented C API cache.

Set `CUBUTTERFLY_PLATFORM_TUNING` to the path of a JSON rules file. An unset
variable means that no file candidates are added. The file is read as content;
the pathname alone is never treated as the identity.

## Rule Schema

The document uses `cubutterfly-platform-tuning-v1` and contains exact hardware
targets. A target must identify all three fields below, with no extra or
missing fields:

```json
{
  "schema": "cubutterfly-platform-tuning-v1",
  "targets": [
    {
      "hardware": {
        "device": "NVIDIA A100 80GB PCIe",
        "compute_capability": "8.0",
        "global_memory_bytes": 84974239744
      },
      "rules": [
        {
          "match": {
            "operator": "ntt",
            "precision": "word64",
            "direction": "inverse",
            "logN": {"min": 12, "max": 20},
            "output_order": "natural"
          },
          "mapping": {
            "backend": "shared-iterative",
            "compute_unit": "radix2"
          }
        }
      ]
    }
  ]
}
```

Use the target GPU's integer byte count for `global_memory_bytes`. The hardware
object is compared exactly against the current device identity. A
different model, SM version, or memory capacity does not match. Rule matches
must include `operator`, `precision`, and `direction`; additional fields are
operator-specific semantic fields. `min` and `max` ranges are numeric and
inclusive. NTT rules can distinguish `input_order` and `output_order`; FFT and
other butterfly rules can distinguish placement, normalization, accumulation,
strides, batch, and length. Butterfly `normalization` is the canonical value
`inverse` or `none`; the legacy `normalize_inverse` boolean is accepted as an
alias. `batch_stride` is the effective stride, including the contiguous
default when the C++ configuration stores zero. Structured-2x2 rules may match
`stage_matrices` using the registry form
`m00:m01:m10:m11xm00:m01:m10:m11`.

The mapping is parsed as a normal serialized mapping. Invalid rule syntax is
reported as a configuration error. An individual mapping that cannot be
parsed is warned about and skipped so other rules and the generated inventory
remain available. The planner still checks its backend capabilities, resource
limits, and operator semantics before it can execute. A semantically
incompatible mapping is therefore a rejected candidate, not an override of
validation.

## Selection Rules

Platform rules prepend candidate priors to the shared portable inventory.
They cannot remove generated candidates, change the requested semantic fields,
or claim that a mapping is calibrated. The exact measured registry is checked
first by `select_butterfly_mapping` and `select_ntt_mapping`; a matching
`calibrated-registry` record outranks all unmeasured hints. If no measured
record matches, DEFAULT takes the first executable candidate from the portable
inventory, including platform-rule candidates and ordinary generated candidates.
MEASURE and installation search can time and rank these candidates. A DEFAULT
fallback is explicitly unmeasured and makes no optimality claim.

The historical selector now lives in
`src/platforms/legacy_selector.cpp`. Its results are compatibility priors in
the same candidate inventory only. They are not a replacement for the measured
registry and do not carry calibrated status.

Explicit NTT backends also consult `src/platforms/ntt_defaults.hpp` for legacy
platform defaults when a launch parameter is unspecified. These defaults remain
separate from kernel/resource legality and never replace an explicit mapping.

## Identity And Resume

The C API workload key includes `platform_tuning_identity()`. Editing a rules
file in place therefore prevents a process-local measured winner from being
replayed under a different rule document.

Install-time calibration records `platform_tuning_sha256` in
`measurement_identity` when the environment variable is set. The resumable
recalibration campaign stores the same content hash in its queue state and
rejects `--resume` when the current file differs. Reusing a path after editing
it is consequently a new measurement identity, even when the benchmark
binaries and hardware are unchanged.

## Finite Install Matrix

`config/install_search_workloads.json` is a finite representative workload
matrix, not an exhaustive enumeration of all semantic combinations or all
mappings. It includes forward and inverse cells and NTT output-order cells so
that direction and ordering remain part of calibration identity. Missing cells
must not be described as measured coverage.

The historical `scripts/analyze_hybrid_dataflow_td_model.py` V100 analysis is
offline research evidence. It is not the modern installation `stage_model` and
must not be presented as the installation stage-service calibration or as
evidence that platform-specific behavior has no hardware dependence.
