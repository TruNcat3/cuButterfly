# Persistent compile queries and grouped measurement workers

Experiment revision `0.9.3.4-async-experiments.3`, October 10, 2026.
The v0.9.3.4 CUDA library, per-device calibration, prepared inputs, candidate
space and measurement protocol remain unchanged.

## Bottleneck

CPU preparation finished all 72,654 known compilation requests without a
compile failure. Nevertheless, completed cases in the preceding experiment
spent about 236–393 seconds constructing a Plan. Every uncached *in-process*
module lookup still started a Python interpreter, even when its compiled file
was ready. The resource proxy also inspected the GPU parent's memory mappings
on every lookup. Thousands of these queries accumulated minutes of host work.

The original remaining manifests alternated module families on every request:
4,623 consecutive families per GPU but only 270 distinct family keys. Loading
another family's specializations often reached the 45,000-VMA guard during a
partially completed search. Process recycling protected correctness and process
resources, but discarded that partial search's initialization work.

## Changes

[semantic_compile_service.py](../scripts/semantic_compile_service.py) keeps the
original broker query implementation in a CPU-only process. A small
[native client](../scripts/semantic_compile_client.cpp) replaces interpreter
startup through the existing `CUBUTTERFLY_JIT_PYTHON` executable hook. The service
checks the calling process and its GPU parent, applies the same resource guard,
and invokes the original broker's runtime CLI. Source/compiler identity,
configuration, mapping keys, module file checks, pending and failure events
retain their existing meaning. The service never compiles or loads CUDA modules.

A connection failure or an unsuccessful response without an event creates a
`broker_error`. If event recording also fails, the worker recognizes captured
transport stderr and rejects the entire search. No transport failure can
silently become a pruned candidate. The client executable checksum and service
sources are included in the experiment identity.

[run_bounded_semantic_queue.py](../scripts/run_bounded_semantic_queue.py) adds
`--group-by-module-family`. Each worker receives cases with matching operator,
storage/compute precision, shape, word width, FFT type and length mode; NTT also
uses output order/layout. The original case objects and their full searches are
unchanged. This is a scheduling choice for module reuse, not a claim that all
members have identical candidate sets. Batch-dependent candidates remain
available, and the exact resource guards remain active.

The new queues use `--in-flight 1`, finishing a case and its CPU verification
before starting another Plan. A new family starts in a fresh worker. Within a
family the existing library module cache can be reused across cases. Resource
recycling and complete rejection of partial searches remain available when a
family's actual module population exceeds a process budget.

## Verification and evidence

139 related CPU tests passed, including transport parity, changed module and
compiler detection, resource limits, missing service, malformed responses,
event-write failure, preservation of all grouped cases and checked handoff.
The host client also builds with `-Wall -Wextra -Werror`.

A real-cache CPU probe compared 32 exact requests and then 12 requests with a
synthetic 30,184-VMA parent. Every original/service response matched exactly,
and no preparation error event was created.

| CPU query setup | Original Python proxy median | Persistent service median |
|---|---:|---:|
| Ordinary parent | 52.79 ms | 8.86 ms |
| 30,184 parent VMAs | 75.43 ms | 31.80 ms |

These numbers describe preparation overhead, not kernel speedup. Raw samples
are in `results/async_fast_dispatch_20261010_v0934_3/cpu_transport_probe.json`.
The original-GPU pilot separately checks complete execution, error reporting,
mapping replay and seven-sample timings for three cases in one family.
All three completed with no numeric failure and with floating accuracy still
pending. The repeated descriptor's first Plan took 141.96 seconds, compared
with 241.02 seconds in the earlier run. The other two family members took
20.67 and 20.99 seconds. No resource recycle occurred. Full MEASURE selected a
different mapping for the repeated descriptor; its reported error distribution
was unchanged. Kernel timings from this small diagnostic are not treated as
evidence of a kernel optimization. Details are in `pilot_summary.json` under
the same campaign directory.

The [handoff utility](../scripts/prepare_bounded_queue_handoff.py) verifies
stopped workers' indexes, artifacts, original descriptors, GPU assignments,
library, calibration, frozen source snapshots and failure boundaries. It
retains completed measurements and all 120 separate scaled controls. Unindexed
rows remain evidence and are rerun, never silently adopted. Historical results
retain their original experiment identities.
The completed audit retained 29,192 processed timing requests, including 232
explicit missing lowerings, plus the 120 controls. Two new grouped queues each
resume 4,612 remaining timing requests. The three pilot records are separate
diagnostics and are not added to the campaign completion count.

At 14:35 CST, the resumed production queues had indexed six and seven new
records respectively, all pending floating accuracy, with no case/baseline
errors or compilation waits. Their first Plans took 133.66 and 127.33 seconds;
subsequent same-family Plans took 30.41–31.47 seconds on GPU1 and 23.53–24.08
seconds on GPU2. Peak VMA counts were 19,845 and 19,141, below the unchanged
45,000 guard. These observations confirm reuse in the live queue, but do not
predict every family's preparation time or the complete matrix's finish time.

## Entry points

Build the CPU transport without rebuilding the CUDA library:

```bash
c++ -O2 -std=c++17 scripts/semantic_compile_client.cpp -o COMPILE_CLIENT
python scripts/semantic_compile_service.py --broker BROKER_DIR --socket SOCKET
```

The socket path must fit the host's Unix-domain socket length limit. Existing
socket files are never replaced. A service accepts one query at a time; the
current campaign uses one service for each GPU, sharing the CPU compilation
broker. The independent compilation producer stays available for newly
discovered requests.

Set `compile_client` and `compile_socket` in a bounded queue plan and launch:

```bash
python scripts/run_bounded_semantic_queue.py \
  --plan PLAN --output FRESH_OUTPUT \
  --group-by-module-family --in-flight 1 --cpu-workers 2
```

Without those plan fields the original Python proxy remains available. The
full matrix and floating-point accuracy acceptance are separate from these
preparation and scheduling checks.
