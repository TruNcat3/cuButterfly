# PRO6000 Figure 9 finite follow-up

Scope: [PR10 follow-up](https://github.com/TruNcat3/cuButterfly/pull/10#issuecomment-6979824753). Original GPU cohorts and raw records remain unchanged. This adds a separately named legal cyclic NTT8 population, fixes derived backend-status ownership, and confirms only the 83 requested CPU denominators in three independent processes.

## Backend status ownership

The three original Full66 and two fixed-mapping NTT campaign failures are external GPU-NTT failures. Derived request tables show campaign_status separately from own_backend_status; backend tables and coverage count each backend_status. cuButterfly is pass because its own and replay checks are exact. GPU-NTT remains correctness-failure with no speedup. The original source status, row bytes and hashes are retained. GPU medians and all 78 prior statistics are unchanged.

## Independent pro6000-cyclic-ntt8

Predeclared eight requests: N=65536, B=1 or 16, forward and inverse, word32 q=1073479681 and word64 q=582090251837636609. Packed natural input/output, out-of-place, forward unnormalized and inverse multiplied by N inverse modulo q. Full descriptions match the A100 cyclic requests; actual primitive roots, input/reference hashes and build identities are recorded. No Full66 modulus or population was edited.

GPU1 uses the already calibrated original SM120 library 3b0668f2… with public MEASURE budget 16 and exact mapping replay. The existing NTT selector counts scheduled candidates; word32 has 14 accepted and two rejected in each 16 scheduled, word64 has 16 accepted. This is a bounded search, not exhaustive coverage. All eight cuButterfly outputs and replays are exact. GPU-NTT has six exact timed pairs: four word32 and two word64 inverse. Both word64 forward results fail; they have no baseline latency or ratio.

| Width | GPU-NTT valid pairs | Geometric speedup | Range | Above / below 1 |
|---|---:|---:|---:|---:|
| word32 | 4 | 1.326× | 1.055–1.649× | 4 / 0 |
| word64 | 2 | 1.346× | 1.201–1.509× | 2 / 0 |

The same original bridge is independently reproduced with one complete call on each failing word64 forward request: 65,534 / 65,536 mismatches at B=1, and 1,048,544 / 1,048,576 at B=16. Input preservation and exclusive GPU guards pass. Minimal reproduction commands and source are retained; no new root/order/residue diagnosis, third-party kernel repair or mechanism profiling was attempted. Required order conversion/inverse scaling is inside the timed complete call. Seven warm Graph samples × twenty calls follow ten warmups; setup is excluded.

## CPU confirmation and remaining provisional status

Three different process IDs complete 83/83 denominators each: Full66 66 (FFTW eight threads only), MMA9 nine explicit FP16-I/O/FP32-FXT adapters, and cyclic NTT8 eight HEXL records. Every process uses the same inputs, thread settings and affinities (16–23 for Full66/NTT8, CPU16 for serial MMA9). Each process retains seven wall-clock samples × twenty calls after ten warmups. The fixed summary is median of each process, then median of the three process medians; no fastest-process selection. Original timings and all new process values/ranges remain available.

No CPU reservation or exclusivity is claimed. Pre-row 0.2-second per-core utilization probes and start/end process tick/affinity snapshots are retained. Rows with some selected core above 5% observed pre-row utilization: 10, 14, 24 of 83 in the three processes; this is a descriptive load indicator, not a numerical acceptance threshold. All confirmed CPU-normalized values remain provisional-shared-host. This does not block the valid GPU/GPU pairs.

Across the 83 denominators, the median process max/min ratio is 1.052, maximum 3.059. Across the 75 existing denominators, new/old geometric ratio is 1.029. These are timing stability observations, not precision error tolerances.

Each process has 37 exact integer and 46 floating review records. FP16 conversion/scaling/narrowing and HEXL uint32-to-uint64 storage adaptation remain inside every execute. Full numerical distributions and reference hashes are retained; no floating acceptance threshold is introduced or relaxed. No GPU Full66 or final FFT cohort was rerun for this CPU confirmation.

## Tables and clean-directory reproduction

[NTT8 every backend](../snapshots/pro6000_figure9_followup_20261118/ntt8_backends.csv), [independent NTT8 statistics](../snapshots/pro6000_figure9_followup_20261118/statistics.csv), [83 CPU denominators](../snapshots/pro6000_figure9_followup_20261118/cpu_confirmation.csv), [249 process records](../snapshots/pro6000_figure9_followup_20261118/cpu_processes.csv), [CPU-normalized plot inputs](../snapshots/pro6000_figure9_followup_20261118/normalized_backends.csv), [exact A100 intersections](../snapshots/pro6000_figure9_followup_20261118/a100_intersections.csv) and [identity registry](../snapshots/pro6000_figure9_followup_20261118/identities.json). The original Figure9 tables retain the old CPU denominators; the confirmation table is separate. Platform/build/cohort means are never pooled.

The self-contained performance-record package includes the four report verifiers, every older cohort source checkpoint, required A100 snapshot/mapping catalog, new NTT8/CPU records and derived tables. Large output-array payloads and binaries are omitted: original array hashes remain metadata, and the Figure9 verifier checks their matching source receipts without claiming to replay absent arrays. The fixed-sample standalone report keeps its stricter default payload verification in a full checkout.

After extracting the standalone package into an empty directory, run `python3 experiments/tools/report_pro6000_figure9.py --check`. Only Python standard library is needed. It checks both Figure9 and this follow-up. All compressed records have compressed/uncompressed hashes, and the package inventory covers every file.

The [confirmed main CPU-normalized figure](../../figures/pro6000_figure9_confirmed.svg), [confirmed MMA9 panel](../../figures/pro6000_figure9_mma9_confirmed.svg) and [independent NTT8 panel](../../figures/pro6000_figure9_cyclic_ntt8.svg) use the fixed three-process medians and retain provisional labels. PDF and PNG exports accompany each SVG.

Download the [self-contained performance records](https://github.com/TruNcat3/cuButterfly/releases/download/pro6000-performance-20261118/pro6000-figure9-followup-standalone-20261118.zip) and [independent SHA256](https://github.com/TruNcat3/cuButterfly/releases/download/pro6000-performance-20261118/pro6000-figure9-followup-standalone-20261118.zip.sha256). Earlier immutable bundles remain available.

Minimal failing-call reproduction on the measurement installation (after sourcing its recorded environment):

```bash
python3 experiments/tools/reproduce_pro6000_ntt8_failure.py \
  --measured-root /home/wt/git/cuButterfly-issue9 \
  --preparation /home/wt/git/cuButterfly-issue9/results/pro6000-figure9-followup-20261118/cyclic-ntt8 \
  --case-id d4b0e39341b4b0f05c12a4e2f104181e16f556b019af57b3fa124d2bbb400560 \
  --output /tmp/ntt8-reproduction.json
```

The B=16 failing case is `04a82eaac33704ba5fdbf53f1073fbd197a55b7b84277240ccce58946e244574`. This diagnostic executes only the original GPU-NTT call, reports exact mismatches and includes no search or substitute kernel. Both direct reproduction records and complete source/bridge/input hashes are in the snapshot; raw output arrays stay on the host.

No Figure11 mechanisms, NCU, fixed-core migration, changed Full66 population, unbounded search or all-green third-party numerical claim is added.
