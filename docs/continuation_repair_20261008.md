# Experiment continuation repair, 0.9.2.2

The October 8 continuation was alive but CPU-bound between GPU shards. At
18:22 it had added only 144 records since 14:37. The target A100 80GB was idle
between short measurement runs. This was an orchestration defect, not evidence
about operator throughput or the dataflow method.

## Root cause and bounded repair

The old controller rebuilt `completed | quarantined` inside a comprehension
for every one of 48,784 parent cases. The processed set had about 35,000 entries.
After every correctness failure, it also reread and revalidated every previous
shard manifest and result. Across successive failures this compounded work
quadratically in the number of shards. The observed gap between shard starts
grew from about two minutes to about seventeen minutes.

The repaired controller builds each exclusion set once, retains verified
progress in memory, and fully checks only newly closed shards. Previously
checked files must retain their device/inode/size/mtime/ctime signatures;
changed or missing evidence stops execution. A new process still fully
validates the history. Large parent case lists are no longer copied twice
when constructing the next immutable submanifest.

The old process was stopped between GPU shards after shard 23 saved its
correctness-failure checkpoint and exit code. Source and state snapshots are
under `results/semantic_campaign_20261008_v0921/controller_repair_20261008/`.
`--upgrade-controller-from` permits exactly the controller-source hash change,
after verifying every existing shard. It archives the old controller envelope
and records old/new source and state hashes. It cannot change GPU, library,
runner, protocol, calibration, baselines or historical row data.

Regression checks cover five successive failure shards being fully checked
five times in one invocation, followed by five fresh validations on resume;
same-process artifact modification/deletion; upgrade rejection before a bad
chain is adopted; and no rerun of completed cases.
The targeted checks passed: 21 continuation tests, five supervisor tests,
25 input-profile/control tests and nine range-queue tests. The queue also
verified both real historical controls on CPU, including exact profile
metadata and executed input/storage/reference arrays, without rerunning them.

The repaired supervisor started at 18:43:52 +08. After full history validation,
the controller adopted the source-only upgrade and launched shard 24 at
18:46:35, then shard 25 at 18:46:55. The latter start interval was 20.16 seconds,
versus approximately 17 minutes near the end of the old run. Checking the new
one-row shard took 1.894 seconds. This is an observed orchestration improvement
for those shards, not a kernel speedup or a full-campaign runtime forecast.
The migration is recorded under
`conformance/controller-upgrade-1791456389502241715/`, with supervision under
`supervision_20261008_v0922/` in the same campaign.

## Downstream experiment audit

Two controller gates required repair. A correctness-stop branch previously
skipped baseline execution-error checks from earlier rows in that shard.
Every branch now rejects them. `external-binding-error` is also fatal even
when the frozen runner records its parent baseline category as missing.
The original row remains unchanged and its baseline detail is authoritative;
the controller does not admit such a shard to the next phase.

The performance phase now requires the same execution inputs, software and
environment as conformance, including calibration and selected baseline
libraries. A different path or protocol requires a separate experiment.
Conformance completion and its full result chain are checked before timing.
Float error reports still do not qualify as accepted accuracy passes.

The frozen measurement runner can record a baseline execution error while its
parent cuButterfly result remains valid; it does not immediately stop on that
baseline status. The repaired outer gate stops at the shard boundary. This
distinction remains explicit rather than rewriting historical measurements.

## Separate scaled-input controls

The earlier single FFT control did not cover all quarantined requests.
`run_repaired.sh` adds a separate control queue between original-input
conformance and performance comparison. It preserves every original failed
row, records unsupported control domains, and reuses existing verified controls
with their original identities. These controls do not qualify excluded cases
for the original-pressure performance population.

The input profile now supports exact power-of-two FP16 C2C FFT and standard
FP16 FWHT. Both derive a nonpositive power-of-two input scale from each full
transform's input L1 bound, with fixed quarter-range headroom. FWHT requires
`fft_output_scale=1`: the current frozen reference and baseline wrapper differ
in handling other values for this non-FFT operator. Such control requests are
rejected explicitly. Every one of the 25 failures present at repair time uses
scale one, so this restriction does not remove a pending failed request.

The new unchanged-mapping FWHT control at N=1,048,576, B=1 uses factor 1/64.
Nonfinite outputs fell from 9 to 0; relative L2 error is 0.000988065. Three input
components became zero in FP16 requantization. Full element-wise errors remain
pending accuracy review; this is not a performance test. Dao-FHT does not bind
this native-half computation contract, so no FP32 substitution was used.
Evidence is in `controller_repair_20261008/fwht_scaled_control/` beneath the
continuation campaign.

## Remaining limits

The static parent population contains 7,680 missing-lowering requests. Among
the others, 8,832 exceed the frozen one-million-element complete-reference
bound. Another 32,272 fit that element bound but can still meet the quadratic
NTT-operation bound or encounter other explicit gaps. Visiting every manifest
request therefore cannot establish full executed performance coverage.
Neither smaller batches, sampled verification, promoted precision nor another
operator will fill those rows silently.

Calibration remains locally qualified with its previously documented gaps.
MEASURE validates the selected final mapping against the nonzero full input;
this does not establish nonzero-input validation of every internal tuning
candidate. FP16 external FFT bindings and several other baseline domains remain
missing. The repair leaves these research obligations visible.
