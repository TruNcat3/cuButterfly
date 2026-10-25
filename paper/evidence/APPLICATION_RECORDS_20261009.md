# Retained application records audited on 2026-10-09

This is a CPU-only audit of historical application evidence, not a new benchmark.
The [machine-readable snapshot](application_snapshot_20261009.json) preserves four
raw-file SHA256 hashes, full embedded identities, extracted field names, values,
and interpretation boundaries. Historical files were not changed.

## Results and aggregation

| Case | Baseline median (ms) | cuButterfly median (ms) | Baseline/cuButterfly |
|---|---:|---:|---:|
| FNO 64-square, B=8 | 3.006523941 | 3.151621670 | 0.953960931 |
| FNO 128-square, B=4 | 2.989774104 | 3.140312154 | 0.952062712 |
| QuaRot random 512/32 | 2668.599895202 | 2647.400203161 | 1.008007740 |
| QuaRot random 2048/128 | 10573.719390668 | 10544.185072184 | 1.002801005 |

FNO uses the median of three independent process-trial medians, separately for
each variant. Each process median pools the measured input-batch forwards;
64-square retains the final batch of four samples. QuaRot uses the median of
three `wall_ms.end_to_end` samples per variant. Ratios divide the baseline median
by the treatment median. Paper rounding to three decimals is respectively
`0.954`, `0.952`, `1.008`, and `1.003`.

## Protocol and quality boundaries

FNO measures the complete device-resident model forward, including real-FFT
packing/scaling, Hermitian expansion, adapter storage and launches. Inputs are
preloaded on the GPU. The synchronized timing function encloses `y = model(x)`;
CUDA-event spans include dispatch gaps and are not sums of kernel durations.
Compilation, training, data I/O, plan creation, online candidate measurement and
first lazy CUDA/module preparation are excluded. Plans must exist before timing.

Protocol v5 uses three fresh processes, five warmups and ten measured forwards
per input batch. The actual process orders are treatment/native, native/treatment,
and native/treatment. The cases use 100 held-out samples each, FP32 with TF32
disabled, four FNO layers, 24 hidden channels, 16-by-16 modes, grid embeddings,
and projection ratio two. Four forward and four inverse real FFT calls are
replaced in every inference with zero fallback. Hardware is A100 80GB PCIe
UUID `2971e500-d3f6-d7e6-e5f2-cb67a06fff02`, PyTorch `2.5.1+cu124`, CUDA `12.4`.
See the [FNO protocol](../experiments/e13/README_fno.md).

Both FNO cases pass the frozen `max_abs <= 1e-4` and `relative_l2 <= 1e-4`
gates before timing. Maximum absolute error is `8.34465e-7` for both; relative
L2 is `1.05608e-7` and `1.32403e-7`. Held-out mean L2/H1 is approximately
`0.04076434/0.10647308` and `0.04051803/0.11074287`, matching across variants
to these displayed precisions. Complete-forward latency nevertheless increases
by approximately 4.83% and 5.04%. This negative result is retained.

The baseline uses the upstream default explicit Hermitian symmetry. Treatment
reconstructs the same full two-dimensional Hermitian spectrum. Frozen H1-trained
checkpoints used an earlier inverse setting; both inference variants reuse them
identically under default inference semantics. This is a source-based application
experiment, not a byte-for-byte tutorial reproduction. The earlier 128-square
non-default inverse failed the symmetry gate with maximum difference `0.0376793`;
no timing from that failed case is accepted.

QuaRot measures a full model with the same random state, seed `20261003`, cloned
into both variants. Batch size is one. The explicit PyTorch SDPA fallback differs
from the upstream FlashAttention configuration. Each region has two warmups and
three measured samples; plans and first use are excluded. Plan creation costs
are `450.57903` and `446.83055` ms. These records support random-weight latency
and memory only. They do not establish trained-model perplexity, logits or token
agreement. Separate mathematical and Dao CUDA adapter gates pass, but do not
constitute a full-model quality gate. QuaRot JSONs embed a library hash but no
runner or adapter hash; the snapshot marks those identities as missing.

HEonGPU has no completed CKKS application comparison. Its retained probe covers
`N=4096`, one 60-bit prime, GPU cores and caller streams with host-assisted twist
and reindexing. Direct twist has 4094 mismatches; frequency multiplier 699
(inverse 627 modulo 4096) gives zero forward mismatches and an exact inverse.
Device-resident adaptation, full RNS coverage and bootstrap timing remain pending.

## Historical identity and summary discrepancy

All four application JSONs embed library SHA256
`90a63df0b568aa0d76231ebd76cecbfcbd16656c06c8ab4e39587069787132e4`.
FNO also embeds runner SHA256
`bfaa11702133604287bfaf367e68f0a103d55131be3b1495e9adbe9cb0dc25cb`
and adapter SHA256
`a6581447219d46e8a4eb53b14351743443524c57b7e6441a1685e39bfe679645`.
Full checkpoint and input identities are in the snapshot. The audited current
shared library and FNO runner differ from these embedded hashes; therefore these
records cannot establish current-build performance. The result README is dated
2026-10-03, but the raw JSONs have no embedded execution timestamps. The snapshot
date must not be presented as their measurement date.

The historical [result README](../../results/paper_profile_rc2/end_to_end/README.md)
and [progress summary](../../results/paper_profile_rc2/end_to_end/e13_progress.json)
report QuaRot 512/32 treatment latency `2648.064075969159` ms and ratio
`1.0077553208312374`. That latency is the second sample, not the median. The raw
treatment samples are `2647.4002031609416`, `2648.0640759691596`, and
`2644.0763222053647` ms. The correct median is `2647.4002031609416` ms and ratio
`1.008007739825422`. The 2048/128 summary agrees with its raw median. This audit
documents the error without rewriting historical results.

## Sources and CPU-only verification

The four primary raw sources and their SHA256 hashes are in `records` of the
snapshot. Additional protocol, correctness and scope sources are listed in
`supporting_sources`. All paths are relative to the repository root.

Run the following from the repository root. It checks raw-file hashes, every
available embedded identity, FNO process medians reconstructed from raw samples,
stored aggregate medians and ratios, QuaRot raw-sample medians, snapshot values,
and three-decimal paper rounding. It performs no writes or GPU operations.

```bash
node <<'NODE'
const fs = require('fs');
const crypto = require('crypto');
const assert = require('assert');
const snapshot = JSON.parse(fs.readFileSync('paper/evidence/application_snapshot_20261009.json'));
const median = values => {
  const v = [...values].sort((a, b) => a - b);
  const n = v.length;
  return n % 2 ? v[(n - 1) / 2] : (v[n / 2 - 1] + v[n / 2]) / 2;
};
const close = (a, b) => assert(Math.abs(a - b) <= 1e-10 * Math.max(1, Math.abs(b)), `${a} != ${b}`);
for (const r of snapshot.records) {
  const bytes = fs.readFileSync(r.source_path);
  assert.strictEqual(crypto.createHash('sha256').update(bytes).digest('hex'), r.source_sha256);
  const raw = JSON.parse(bytes);
  for (const [key, value] of Object.entries(r.embedded_identity)) {
    if (key === 'missing_identity_reason') continue;
    if (value === null) assert.strictEqual(raw[key], undefined);
    else assert.strictEqual(raw[key], value);
  }
  let baseline, treatment;
  if (r.id.startsWith('fno_')) {
    const aggregate = key => {
      const v = raw[key];
      const count = v.samples.length / raw.process_trials;
      assert(Number.isInteger(count));
      const trials = Array.from({length: raw.process_trials}, (_, i) => median(v.samples.slice(i * count, (i + 1) * count)));
      trials.forEach((x, i) => close(x, v.trial_medians[i]));
      const m = median(trials);
      close(m, v.median_of_trial_medians);
      return m;
    };
    baseline = aggregate('native_ms');
    treatment = aggregate('cubutterfly_ms');
    close(baseline / treatment, raw.speedup_native_over_cubutterfly);
    assert.strictEqual(raw.pass, true);
    assert.strictEqual(raw.quality.gate_before_timing, true);
    assert.strictEqual(raw.adapter_coverage.fallback_calls, 0);
  } else {
    assert.deepStrictEqual(raw.baseline.wall_ms.end_to_end, r.baseline_wall_samples_ms);
    assert.deepStrictEqual(raw.cubutterfly.wall_ms.end_to_end, r.cubutterfly_wall_samples_ms);
    baseline = median(raw.baseline.wall_ms.end_to_end);
    treatment = median(raw.cubutterfly.wall_ms.end_to_end);
    assert.strictEqual(raw.timing_contract.quality_claim, false);
  }
  close(baseline, r.baseline_median_ms);
  close(treatment, r.cubutterfly_median_ms);
  close(baseline / treatment, r.speedup);
  assert.strictEqual((baseline / treatment).toFixed(3), r.paper_speedup);
  console.log(`${r.id}: hashes, medians, ratio and rounding PASS (${r.paper_speedup})`);
}
NODE
```
