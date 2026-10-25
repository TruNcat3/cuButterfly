# Group-local residency follow-up

This is an independent FWHT follow-up to the medium/large layout contract. It
does not modify or reuse the frozen layout manifests as executable specifications.

The pair fixes `N=2^20`, `B=4`, FP32 forward out-of-place semantics, scalar radix2
`FwhtOperator<float>`, physical stages `[10,10]`, 256 threads, linear shared
layout, one batch tile and no overlap. The baseline has no local resident
partition. The treatment uses five two-stage register leaves in each physical
group: `[[2,2,2,2,2],[2,2,2,2,2]]`.

The declared public change is only `local_stage_partitions`. It necessarily
induces a different kernel template (`shared_iterative_kernel` to
`shared_resident_kernel`), four values per resident leaf, five rather than ten
shared publication units per group, different pair ownership/arithmetic schedule,
and different register pressure. Therefore this is an organization-parameter
study with explicitly reported induced machine changes, not a strict machine
single-variable claim and not a core x organization 2x2 result.

Both modules are CPU-prepared and hash-checked through the existing compile
broker. The guarded command is in `results/paper_targeted_20261012/medium_large_residency_v2/launch.json`.
Normal timing, preparation, mapping, numeric and counter gates remain separate.
FP32 verification is internal sanity only; full user error acceptance requires
the separate fixed-mapping report-only validator and has no invented threshold.
