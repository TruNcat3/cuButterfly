# Paper-to-artifact alignment

The paper studies **execution organization** as a performance dimension that
can be varied while retaining specialized arithmetic kernels. The artifact
represents dependency-closed tiles, stage/data space–time choices, hierarchical
composition and boundary exchange. Unification applies to constructing and
realizing this space; it does not require every operator to select the same
mapping or benefit from every mechanism.

## Measured evidence

The [evaluation bundle](../experiments/paper/README.md) is the canonical map
for the current experimental figures. Its [manifest](../experiments/paper/manifest.json)
pins the bundled inputs and executable analyses. The following command
reconstructs the figures and table without CUDA or access to a development
checkout:

```bash
python experiments/paper/reproduce.py --output-dir results/paper-figures
```

| Evidence | Isolated question | Published reproduction |
| --- | --- | --- |
| Fixed-kernel organization, Fig. 8 | Does organization matter with arithmetic held fixed? | 150 source observations and fixed-core analysis |
| Complete transforms, Fig. 9 | How do generated implementations compare with external libraries? | 411 CPU-normalized observations, separate A100/PRO 6000 populations, paired timing inputs and coverage records |
| Applications, Fig. 10 | Do operator gains reach the application? | CKKS compared with the faster matched native alternative; other application and integration cohorts kept separate |
| NTT mechanism controls, Fig. 11 | What changes when retention or boundary placement changes? | Fixed arithmetic-provider controls, ordinary timing and separate NCU counters; losses retained |
| Cross-operator resource analysis, Fig. 12 | Why do different operators prefer different group depths? | 18 profile records and resource plots |
| Positive boundary control, table | Does boundary realization improve a fixed implementation? | Six paired processes, fixed global layout and launch resources, numerical checks and local-bank counters |

All exported values come from stored measurements. The bundle separately
checks numerical equivalence and sanitized file hashes; original input hashes
are retained as provenance. It contains no paper text, manuscript source or
manuscript PDF.

## Running new experiments

Offline figure regeneration and new measurement are separate entry points.
Use the [benchmark guide](../benchmark/README.md) for configuration-driven
requests and the [hardware workflow](hardware_profile_install.md) for local
calibration. The [NTT mechanism package](../experiments/paper/ntt_mechanisms/gpu/README.md)
also includes the exact fixed-primitive harness, pinned source resolver,
CPU preparation, guarded GPU campaign and analyzer for Fig. 11(a–d).

The main66 platform manifest does not include all supplemental comparison,
mechanism and application cohorts. The generic full inventory is not the
paper's exact matrix. Likewise, the historical E1–E14 preparation plan covers
research protocols beyond the current figures. E14 now emits all declared
cases and explicit journal/variant bindings; an unbound case is not an
executed measurement. Its generic collector has narrower kernel/counter
coverage than the current NTT harness.

Current Fig. 11(e–f) can be redrawn from the bundled timing and counter
records, but this package does not yet provide an exact standalone GPU replay
for that local-layout control. Some application/historical controls require
external dependencies or their original builds. These limits are documented
at the corresponding figure entry; the offline command does not claim to
recreate them.

Use complete-call timing, keep conversion costs, preserve arithmetic and
layout semantics, and report numerical error and failed/unsupported requests.
Never pool unrelated cohorts or infer a performance guarantee from the model.
The [reporting principles](reporting_contract.md) and
[design overview](design_overview.md) describe the common method; older V100
protocols remain historical material rather than evidence for this evaluation.
