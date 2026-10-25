# NTT APPT Variant Audit (2026-10-23)

This audit compares the public `NttSubgraphCore::Appt*` enum with the CUDA
launcher predicates, `Plan::validate_config`, and the portable candidate
inventory. A launcher path is executable support; it is not automatically a
portable search candidate.

| Variant | Launcher/predicate | Validation | Portable candidate | Status |
|---|---|---|---|---|
| `ApptPipeline` | yes | yes | no | executable, descriptor pending |
| `ApptOnline` | yes | yes | no | executable, descriptor pending |
| `ApptOnlineRadix4` | yes | yes | no | executable, physical-core descriptor pending |
| `ApptOnlineFusedTail` | yes | yes | no | executable, tail-liveness descriptor pending |
| `ApptOnlineSplitTail` | yes | yes | no | executable, tail-liveness descriptor pending |
| `ApptOnlineRegisterTail` | yes | yes | yes | portable candidate with a complete 7+7+6 descriptor |
| `ApptOnlineRegisterTailWarp` | yes | yes | no | executable, codelet descriptor pending |
| `ApptOnlineRegisterTailColumnWarp` | yes | yes | no | executable, codelet descriptor pending |
| `ApptOnlineRegisterTailRadix8` | yes | yes | no | executable, codelet descriptor pending |
| `ApptOnlineRegisterTailGrouped` | yes | yes | no | executable, grouped producer descriptor pending |
| `ApptOnlineRegisterTailGroupedWriterFinal` | yes | yes | no | executable, natural-output descriptor pending |
| `ApptOnlineRegisterTailGroupedWriterFinalDataTime` | yes | yes | no | executable, role-mask descriptor pending |
| `ApptOnlineRegisterTailGroupedWriterFinalResident` | yes | yes | no | executable, shared-memory liveness descriptor pending |
| `ApptOnlineRegisterTailGroupedWriterFinalResidentQuarter` | yes | yes | no | executable, shared-memory liveness descriptor pending |

The omission is intentional. The runtime dispatcher selects these variants
from serialized fields such as stage partition, physical core, data-space,
fragment width, role weights, writer mode, data-time mask, and resident-memory
mode. Until those fields have a variant-specific descriptor and a replayable
candidate contract, adding the enum to the inventory would make search report
an unmeasured configuration as a portable candidate.

The negative contract is tested by
`cubutterfly_ntt_template_candidate_tests`: every omitted APPT variant must
have no portable name and a non-empty omission reason. The register-tail
candidate additionally checks the complete role, ring-boundary, and
serialization round-trip contract.

