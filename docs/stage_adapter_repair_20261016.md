# Stage calibration adapter repair, 2026-10-16

This development repair closes two concrete gaps in the A100 80 GB installation
campaign. It changes probe/resource plumbing, not the transform kernels or the
semantic comparison scope.

| Gap | Cause | Repair |
| --- | --- | --- |
| Two NTT compact-stage mapping records lacked independent services | The probe exposed one non-independent logical group although execution launched three kernels | Expose the original physical `{5, 6, 9}` stage groups, their actual resources and their in-place/writeback contracts |
| Nine FP64 cuFFTDx overlap composition requests rejected serial projection | Overlap descriptors reported unknown shared memory as zero; serial capture reported actual shared memory | Query the exact kernel selected by launch dispatch, including static and dynamic shared memory, registers and local memory |

Resource identity checks remain strict. Natural and bit-reversed NTT output
orders retain their original kernels. FP64 queries use the selected direction,
thread count, elements per thread and shared layout.

## Evidence and continuation

CPU contract tests cover resource-query failures, static plus dynamic shared
memory, projection identity and campaign adoption. GPU checks cover NTT against
its exact CPU reference, independent stage recomposition and the original nine
FP64 requests (`B=9,10,11`, tile count `2,4,8`). The rebuilt runtime fingerprint is
`f6adad384580d5d9db880a3ba38218de648c9f656b81b844077469d3062a9a63:research`.

Validation on GPU `GPU-2971e500-d3f6-d7e6-e5f2-cb67a06fff02`:

- CPU projection/resource/adoption suite: 29 passed, with the nine GPU tests
  deferred to the exclusive-device run below.
- Native stage probe: all 40 recomposition requests passed, including both
  compact-stage output orders.
- NTT C++ probe suite: passed. Its first run exposed an invalid default modulus
  in the new large-size test; the corrected test uses the campaign modulus
  `576460756061519873`. The failed attempt remains recorded.
- FP64 overlap resource checks: all nine original requests passed against actual
  serial kernel capture, including partial final tiles.

Raw logs are under `results/stage_adapter_repair_20261016/`. These are correctness
and metadata checks; current-build performance confirmation is a separate run.

The previous attempt completed all 23 search cells and confirmed 69 mapping
records. Its final stage adapter gate failed before the 66-request GPU semantic
comparison. The completed 83-row CPU external comparison is reusable.

The continuation command is:

```bash
/home/wt/yes/envs/cubutterfly/bin/python \
  results/install_compare_repair_20261014/run.py \
  --resume --adopt-current-build --reuse-completed-search
```

Adoption archives the previous attempt and hashes its search inputs. It retains
mapping proposals, then reruns confirmation, independent stage calibration,
composition validation and public selector replay with the rebuilt library.
Subsequent interruptions use `--resume --resume-after-gpu-interruption` without
another adoption. The GPU idle watcher resumes only interruptions; a numerical,
compilation or calibration failure remains visible for inspection.

The rebuilt campaign resumed at 12:30 CST on October 16. Its status lives in
`results/install_compare_repair_20261014/status.json`; the idle watcher records
`monitor_status.json` beside it. The CPU/GPU comparison follower is bound to that
watcher and will join the retained CPU results with the completed GPU checkpoint.
The previous final confirmation is preserved under
`attempts/adopt-current-build-1792125014061910222/finalized/` within the campaign.

## Limits

The previous finite composition report had 45 pairs, 36 with covered predictions,
and 9 uncovered FP64 pairs. Covered-pair top-one agreement was 100%, but the P90
relative latency prediction error was about 32%. The largest covered errors were
FP32 inverse overlap cases. Closing the resource/probe gaps does not by itself
fix those prediction errors or identify concurrency parameters. New measurements
must report coverage, ranking and latency error separately; this repair does not
claim globally calibrated migration or a global performance optimum.
