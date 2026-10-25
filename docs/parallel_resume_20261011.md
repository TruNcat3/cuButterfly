# Parallel A100 80GB continuation

The October 11 continuation restores the original two A100 80GB assignments
under `0.9.3.5-library-rebound-20261011`. It does not redistribute the 80GB
population onto the separate A100 40GB hardware cohort.

| Device UUID suffix | First phase | Timing refresh | Separate scaled controls |
| --- | ---: | ---: | ---: |
| GPU1, `cb67a06fff02` | 4,601 | 14,607 | 120 |
| GPU2, `ffeae49309e6` | 4,600 | 14,608 | 0 |

The timing population remains 38,416 requests. Controls are reported
separately. These are scheduled counts, not completed or accepted results.
The existing GPU2 controller and its refresh successor remain in place.
GPU1 runs all four original phases in order: remaining timing, refresh timing,
24 real-FFT controls, and 96 C2C-FFT/FWHT controls.

Both queues use library SHA-256
`24486975585fa2ab6399a0cfeae4b9449c398490cb76c5ea5a73a3560abe8c66`,
research compilation, the shared-runtime compilation broker and the explicit
512-candidate search budget. The budget is not exhaustive search. CPU inputs,
reference outputs and compatible compiled modules are reused; historical
measurements from a different build are not adopted.

GPU1 acquires fresh hardware rates on
`GPU-2971e500-d3f6-d7e6-e5f2-cb67a06fff02` and remeasures eight fixed mappings
across four workloads, including their stage and composition services. Only
mapping descriptions are imported from the GPU2 seed profile. Selector replay
and library, GPU, source and JIT identity checks gate queue startup. This finite
calibration does not qualify all migration or concurrency configurations.

The `r2` calibration completed: eight mappings were confirmed, 94 stage
records were acquired, and all four selector replays passed. Seven registry
records were promoted. The result remains `calibrated-local-serial` with
`inventory_complete=false`. Both formal queues are now running, and GPU1
has produced new indexed rows. Live counters remain in the queue reports.

The first attempt stopped because a calibration module had not been registered
under the runtime request's exact identity. The shared module already existed;
the CPU producer registered it about two seconds after the pending event.
The CPU-only preflight then checked all eight mapping dependencies and
registered two further missing request identities. A fresh `r2` attempt keeps
the original failed attempt, events and partial measurements intact.

Local artifacts:

- Active index: `results/active_experiment_queue.json`.
- GPU1 launcher: `results/parallel_resume_20261011_gpu1/resume.py`.
- CPU preflight: `results/parallel_resume_20261011_gpu1/preflight/readiness.json`.
- GPU1 lifecycle: `results/parallel_resume_20261011_gpu1/r2/status.json`.
- GPU1 queue: `results/parallel_resume_20261011_gpu1/r2/run/queue_status.json`.
- GPU1 population and binding evidence:
  `results/parallel_resume_20261011_gpu1/r2/provenance.json`.
- GPU2 queue and successor are indexed under `bounded_search_campaign`.

Each GPU has one measurement worker at a time. The CPU compiler producer is
shared; each queue has its own compilation-service socket. Foreign GPU work,
source changes, preparation failures and abnormal exits stop the affected
queue for investigation. The launcher does not silently retry them. Inspect
live status and newly indexed rows before reporting progress.

Integer outputs still require exact comparison. Floating-point results remain
`accuracy-review-required` until the agreed error-distribution review. Native
and adapted paths keep their full conversion overhead and separate labels.
No precision, batch or semantic substitution is introduced by this recovery.
