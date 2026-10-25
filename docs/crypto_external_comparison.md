# Cryptographic extension comparison

`scripts/run_crypto_external.py` compares the existing cryptographic extension
contracts with the merge implementation of
[GPU-NTT](https://github.com/Alisah-Ozcan/GPU-NTT). It is separate from the running
acceptance campaigns and does not replace their measurements or calibration.

Both libraries receive the same modulus, primitive root, transform direction,
batch, input pattern and natural input/output order. Each invocation checks all
coefficients in every batch and modulus channel against an exact CPU reference.
The runner also checks that the reported roots and inverse normalization match.
Compilation, planning, allocation, host transfers and CPU validation are outside
CUDA event timing. Required GPU permutations, twists and normalization are inside.

This is an operator comparison. Multiple modulus channels use the same sequential
composition on one stream. It does not compare native fused RNS scheduling, CRT,
rescaling or complete HE/ZK applications. The GPU-NTT four-step algorithm is not
included in this adapter. These distinctions must remain in result descriptions.

## Coverage and interpretation

At GPU-NTT revision `d03c5eaeadaa780d153496afcb3b6a9b79a13a63`, merge dispatch
supports logN 1 through 28, including a dedicated small-transform path. Its Barrett arithmetic explicitly restricts Data32
to at most 30-bit moduli and Data64 to at most 62-bit moduli. Consequently,
31-bit BabyBear/KoalaBear inputs in 32-bit storage cannot be presented as native
same-width GPU-NTT comparisons. Batch is also limited by CUDA grid.z to 65535.
Unsupported contracts retain an explicit reason, rather than an invented timing.

The September 20 matrix has 346 contracts. This adapter initially admits 268.
The other 78 exceed Data32's modulus range. Admission is not successful
correctness or performance evidence. The old fixed-modulus harness's logN 12..24
limit does not describe the clean upstream library used in this comparison.

Preflight then found a correctness defect in upstream Barrett multiplication for
the matrix's 62-bit modulus, `2391411402133733377`. A standalone kernel calling
the unmodified upstream multiplication returned 4096 mismatches in 4096 random
and boundary probes, while the 40/50/60-bit moduli each returned zero. The
upstream 128-bit right-shift operator does not handle the 65-bit shift requested
by this modulus. The exact failing modulus affects 54 contracts, including RNS
compositions containing it. They remain `blocked-baseline-correctness`, with raw
evidence, leaving 214 pairs to measure. This is not a cuButterfly speedup or an
assertion that cuButterfly's own 62-bit path failed.

The optional `--baseline-validation` argument records a pinned-revision primitive
failure and its evidence hashes. It excludes only the documented modulus from
paired timings. A corrected upstream version requires fresh validation and a
new comparison identity. Vendor kernels are not modified by this study.

A deterministic 14-contract pilot covers the available modes, directions, word
widths, modulus widths, transform sizes, batches and channel counts. Selection
uses only these parameters and input size, never observed performance. The full
run measures this pilot first, then the remaining compatible contracts.

Historical search journals provide exact-contract mapping proposals. A proposal
may come from another GPU UUID; its timing is never imported. The frozen proposal
is checked and measured again on the target. This run therefore measures existing
selected designs and does not claim a new optimum or a hardware migration result.

By default each library runs three trials for each of two input patterns, with
100 warmups and 100 timed repetitions. Headline speedup uses the median random
trial time of each library. Boundary patterns check modular arithmetic edge cases
and are retained separately. A ratio above one favors cuButterfly. Incomplete,
failed and unsupported cells never enter the geometric mean. Until the compatible
matrix finishes, the summary describes only the completed subset.

## Preparation and execution

Build the isolated external adapter using the instructions in
`benchmarks/gpuntt_crypto/README.md`. Use a clean pinned upstream checkout and an
out-of-tree build. The cuButterfly executable is the existing isolated
`benchmarks/crypto_ntt` consumer build.

Prepare contracts and mapping proposals without a GPU:

```sh
python scripts/run_crypto_external.py prepare \
  --output-dir results/crypto_external_20260929 \
  --mapping-source results/crypto_application_20260920/a100-80gb/acceptance/acceptance.json \
  --baseline-validation results/crypto_external_20260929/baseline_validation.json
```

Run with the physical GPU UUID and compiled external executable:

```sh
python scripts/run_crypto_external.py run \
  --output-dir results/crypto_external_20260929 \
  --gpu GPU-2971e500-d3f6-d7e6-e5f2-cb67a06fff02 \
  --gpuntt-source /tmp/cubutterfly-gpuntt-crypto-20260929 \
  --gpuntt-binary /tmp/cubutterfly-gpuntt-crypto-build-20260929/gpuntt_crypto_bench
```

Use `--scope pilot` for just the representative checks. A later run with
`--scope all` reuses those validated trials. The same command resumes interrupted
work; `--retry-failed` explicitly retries failed cells. Binary, source, matrix,
protocol and device identities must remain compatible. Changed inputs require a
new output directory. A clean binary snapshot is kept in the result directory.

The runner locks only the selected GPU and monitors for other clients. If another
client arrives, it stops its own invocation and leaves a resumable checkpoint.
It does not cancel other campaigns, occupy additional GPUs or wait indefinitely.

```sh
python scripts/run_crypto_external.py status --output-dir results/crypto_external_20260929
```

`journal.json` contains the complete per-trial records, including roots and mapping
provenance. `summary.json` contains progress and grouped ratios. Raw subprocess
output is retained in `logs/`. Known failed cells remain visible even when the
rest of the campaign continues.
