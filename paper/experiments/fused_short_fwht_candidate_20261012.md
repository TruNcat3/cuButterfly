# Fused short FWHT candidate

The independent provider is `python/cubutterfly/fused_short_fwht.py` and its
standalone CUDA source. It is not registered in the C++ public planner/search,
and does not modify QuaRot's adapter, the running GPU1 application queue, frozen
libraries or earlier manifests. Existing `register_fwht.cuh` accepts FP32
storage; existing low-precision traits narrow intermediate butterflies. Neither
implements this requested FP16-storage/FP32-register/scale-before-final-narrowing
short-transform contract.

This provider handles contiguous out-of-place N32/N64 FP16 tensors with arbitrary
positive row count. The caller supplies scale. One launch uses four warp-owned
rows per CTA, shuffle butterfly arithmetic in FP32, explicit FP32 scale, then
one FP16 store. N64 holds two FP32 values per lane. No normalization or QuaRot
dimension is hard-coded in the provider.

The finite benchmark compares Dao, unchanged `fp32_composed`, and the candidate
on identical seeded FP16 random tensors at `(N,B)=(32,128),(32,65536),(64,172),
(64,88064)`. Benchmark scale is the existing application scale associated with
full dimensions 4096/11008. These are real application requests, not padding or
substituted larger N. Scale remains explicit in the reusable provider.

CPU preparation:

```bash
CUDA_VISIBLE_DEVICES='' /home/wt/yes/envs/cubutterfly/bin/python paper/tools/benchmark_fused_short_fwht_20261012.py --prepare
```

Main-agent GPU2 dispatch only (existing finite 16:30 deadline):

```bash
/home/wt/yes/envs/cubutterfly/bin/python -c 'import json,os; a=json.load(open("results/paper_targeted_20261012/fused_short_fwht_candidate/prepared.json"))["launch_argv"]; os.execv(a[0],a)'
```

Each attempt writes result/error arrays inside `XD_REPORT_DIR`. Source/module
identity is preserved in `prepared.json`. The benchmark excludes compilation,
plan creation, allocation warmup and graph capture from timing. Eager and graph
timings are separate; three event trials share a process and are not independent
process replicates. Three pairwise zero-difference checks are recorded. Any
numerical or execution failure rejects the run without changing tolerance, and
the full difference arrays remain available. This subpath is not an application
end-to-end gain or proof of full-model equality.

Next integration requires a public local-core/provider identity exposing FP16
storage, FP32 accumulation and caller scale, plus descriptor/lowering/search
registration and full application zero-difference validation. That would change
the public runtime identity and is intentionally not done in this frozen cohort.
