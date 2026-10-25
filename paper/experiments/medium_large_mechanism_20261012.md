# Medium/large controlled mechanism contract

This is a new finite cohort, not a rewrite of small-size raw evidence. It follows
[research principles](../../docs/research_principles.md), with the complete
machine-readable contract in [configuration](medium_large_mechanism_20261012.json).
The first round isolates shared layout at N2^20 B4; each operator has its own
guarded launch so the main agent can inspect one pair before starting the next.
Preparation is CPU-only. Nothing here launches GPU automatically.

## Fixed controls

| Control | All first-round pairs |
| --- | --- |
| Shape | 1D N2^20, B4, forward, out-of-place, unnormalized |
| Organization | shared-iterative, physical partition `[10,10]` |
| Launch | 256 threads, 1024 values/CTA, 4096 CTAs/group |
| Core | scalar radix2; exact operator trait identity in manifest |
| Overlap/batching | overlap false, batch tile one |
| Local resident units | empty; unchanged between layouts |
| Protocol | three independent processes; 100 warmup, 100 repeat |
| Verification | full batch in preflight; benchmark internal rules unchanged |

FFT FP32 fixes native accumulation, four-multiply complex arithmetic and table
twiddles. FWHT FP32 fixes native scalar addition/subtraction. NTT word64 fixes
Shoup/table arithmetic, modulus 576460756061519873, natural input/output order.
The only requested change is `shared_layout: linear -> writer-aligned` for FFT
and FWHT, or `dataflow_layout: linear -> hermes-xor` for NTT. Both spellings lower
to the same consumer-bit shared permutation, not a substituted arithmetic core.

Two 10-stage physical groups produce two kernel launches. This is a parallel
tile baseline, not one launch per butterfly stage or a deliberately tiny CTA.
Dynamic shared demand is 8 KiB for FWHT and 16 KiB for FFT/NTT. A conservative
memory-planning envelope is 144 MiB for FWHT and 288 MiB for FFT/NTT; this is not
a measured allocation. The first-round envelope ceiling is 4 GiB. Actual memory
or launch failure remains a failed item, not permission to reduce batch/precision.

Source/API support is established by the existing shared JIT specialization in
`scripts/compile_module.py:302`, mapping projection in `src/jit_module.cpp:176`,
and the frozen `shared_iterative.cuh`/operator traits. CPU compilation verifies
the frozen template is buildable; only the later full-batch GPU preflight can
verify resolved runtime support, values and launch resources.

## Prepare and dispatch

Use the existing environment and frozen A100 80GB GPU2 build/runtime identity.
The tool selectively queues/compiles only two modules per invocation with
`CUDA_VISIBLE_DEVICES` empty and one CPU compiler; it never drains the shared
broker or changes its configuration. Compiled module identity is checked before
the guarded child enters the existing telemetry/controlled runner.

```bash
/home/wt/yes/envs/cubutterfly/bin/python paper/tools/prepare_medium_large_mechanism_20261012.py --operator fft --compile
/home/wt/yes/envs/cubutterfly/bin/python paper/tools/prepare_medium_large_mechanism_20261012.py --operator fwht --compile
/home/wt/yes/envs/cubutterfly/bin/python paper/tools/prepare_medium_large_mechanism_20261012.py --operator ntt --compile
```

Each writes `results/paper_targeted_20261012/medium_large_mechanism/OPERATOR/`
with study, prepared commands, resource/identity contract, modules and launch.
Launch only a manifest whose status is `ready-for-guarded-preflight`. The main
agent controls GPU dispatch; the existing isolation guard targets UUID
`GPU-0eae5f07-33de-321b-70a9-ffeae49309e6` and leaves GPU1's QuaRot queue untouched.
The authorized finite deadline remains 16:30 local; expiration is not renewed.

For FFT (substitute FWHT/NTT only after inspection):

```bash
/home/wt/yes/envs/cubutterfly/bin/python -c 'import json,os; a=json.load(open("results/paper_targeted_20261012/medium_large_mechanism/fft/launch.json"))["launch_argv"]; os.execv(a[0],a)'
```

All timing, telemetry and preparation events are attempt-local through
`XD_REPORT_DIR`. After guard release, run the operator manifest's `audit_command`.
The audit compares actual resolved controls, full-batch checks, all three trials,
runtime fingerprint, core/group mapping fields and exactly the declared layout
axis. A rejected audit exits nonzero. Guard completion alone is not acceptance.
Keep execution, preparation, mapping, numeric and counter gates separately.

## Numeric and evidence boundary

FFT/FWHT existing `--verify` checks are internal sanity, not an approved user
floating-error threshold. Full error distribution remains an interface gap;
performance is exploratory until errors can be reported for the identical fixed
mapping/input/reference. A failed internal sanity check is preserved, never
bypassed or softened. NTT requires exact elementwise reference equality for all
four batches. CPU tests and compilation do not verify GPU numerical correctness.

This is mechanism ablation I, not automatic-library E1, core transfer or
disabled-component re-search. Normal timings are not full C API end-to-end
latencies. Counters are not queued here; collect matching kernel-bound bank
replay, occupancy and L2 evidence separately if scheduled. No counter conclusion
follows from compile-time resource metadata.

## Separate work and scaling

Group-local residency is already expressible with `local_stage_partitions`,
e.g. five 2-stage local units in each unchanged 10-stage physical group. It uses
the same scalar primitive but changes register ownership/active work and shared
publication; it is a distinct one-axis study, not folded into layout results.
No residency experiment is launched by this configuration.

The public core x organization 2x2 still needs a portable leaf contract and a
second interchangeable core under matched conventional/mixed organizations.
Historical standalone native/cuFFTDx FFT evidence is reusable as named historical
evidence, not current public-planner speed. Complete external libraries remain
black-box baselines. Disabled-component equal-budget re-search needs enforceable
public search constraints and candidate provenance, not a hand-written replay.

Small single-segment shared scalar radix2 is already legal; the missing capability
is a second verifiable single-segment leaf adapter usable in both organizations:
warp/register FWHT codelet, word64 Shoup NTT local codelet, and native/imported
FFT codelet at a real semantic intersection. These are engineering dependencies,
not absent scalar support or an instruction to implement new CUDA cores now.

The planned main size/batch candidates cover N2^16/18/20/22 and B1/4/32 through
seven selected points, not their Cartesian product. They are not queued or
compiled by this first round. N2^24 B4 is optional and needs a separate memory
and CPU-reference budget, compatible modules and exclusive window; it is not a
required endpoint and does not silently expand the present experiment.
