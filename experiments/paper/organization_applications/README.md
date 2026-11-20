# Offline organization and application figures

Run from this directory (Python with Matplotlib and NumPy):

```bash
python render.py --output-dir output
```

This renders the latest paper's Fig. 8 (`cross_operator_attribution`), Fig. 10 (`current_refresh_applications`), and Fig. 12 (`local_organization_profiles`) as PDF, SVG, and PNG. It exports the 150 attribution ratios, 18 profiler records, all application source rows, and figure summaries. No GPU, network, manuscript PDF, or TeX inputs are needed. The original physical sizes, fonts, point encodings, axis ranges, and estimators are retained.

Inputs are frozen numerical subsets of `motivation_organization_a100.csv`, `principle_core_organization.csv`, `resource_decisions_20261107.json`, `narrative_scaling_20261027.json`, the current-refresh snapshot, application closeout, fused QuaRot follow-up, and historical complete-call records. Original file and renderer SHA256 values are in `source_provenance.json`; these identify the original sources before sanitization. The runtime manifest separately hashes the actual sanitized input bundle. Machine identities, personal locations, and unused nested execution payloads are omitted. Source hash values remain provenance, not assertions that sanitized bytes equal original bytes.

Fig. 8 keeps independent fixed-kernel, FFT, and FWHT cohorts separate; the NTT endpoint identity is documented as one measured mapping, without a gain claim. Fig. 12 connects two kernel diagnostics by their observed min/max, not trial uncertainty. Barrier stalls are per active warp, and active warps are the named NCU metric.

Fig. 10 validates 36 distinct CKKS requests against `min(native_loop_ms, native_stage_batch_ms)` with shared cuButterfly times. The geometric mean is 1.131051975773907. Poisson/filter, bootstrap, historical FNO, closure, diagnostics, and fused QuaRot follow-up retain separate cohorts and aggregation rules. Archived invalid historical QuaRot observations remain explicitly excluded. The two application panels retain their different labeled ratio ranges.

This is offline reproduction of stored measurements. New GPU measurements use the artifact's existing runners and configuration, with their recorded hardware, precision, build, timing boundary, and validation controls. Running those entry points does not imply replay of the exact frozen current-paper matrix or recreation of historical builds. No GPU experiments are launched by this directory.

The same command also exports the Table II positive boundary control as `boundary_positive_summary_20261119.csv` (six process rows) and JSON (two batch summaries). It can run separately:

```bash
python export_boundary_positive.py --output-dir output
```

The minimal frozen banking timing, profile, and matching hashed identity subset preserves the original numerical controls: three processes per batch, seven trial intervals per mode, twenty captured complete calls, finite CPU checks, byte-identical Linear/XOR outputs, fixed global layout, and equal launch/register/shared-memory resources. The arithmetic core is fixed cuFFTDx Block with stage partition 8+12, EPT16/256 threads, and no overlap. The estimator is the median/min/max of three paired process ratios; the range is not a confidence interval. Local complex-FP32 tile slot XOR is not a direct physical-bank single-word theorem, and residual conflicts remain explicitly measured.

The boundary exporter validates the sanitized numerical subset and matching source/binary/library/configuration hashes. Original embedded raw text is deliberately omitted; runtime output explicitly reports that embedded raw hashes are **not** revalidated. The original unsanitized source SHA256 is separate in `source_provenance.json`, and the combined render manifest hashes the sanitized boundary input. Neither stale raw hashes from another sanitized snapshot nor GPU UUIDs are used to claim raw-byte validation.
