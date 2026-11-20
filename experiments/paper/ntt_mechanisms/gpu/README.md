# Exact Figure 11(a–d) GPU remeasurement route

This is the measured fixed-primitive harness, its exact local kernel/header dependencies, the campaign runner, analyzer, and exclusive-device guard. The harness and three local headers match the original campaign SHA256 entries. Only vendor-path configuration, compiler discovery, and pre-build source/license verification and an explicit fixed-length constraint were added to the runner. The task matrix, task order, arithmetic, CUDA Graph/event timing, correctness checks, and NCU invocation are unchanged. New measurements form a new cohort; no identical performance is promised.

Requirements: Linux, Python 3.9+, CUDA toolkit with `nvcc` supporting C++17 and sm_80, compatible host C++ compiler, and enough disk space for source archives/build/input/reference files. GPU execution additionally requires an A100, `nvidia-smi`, Nsight Compute (`ncu`) with hardware counter permissions, and exclusive access. Preparation compiles CUDA code and runs only its CPU reference/input preparation mode. No GPU is used in that phase.

External arithmetic sources are not redistributed here. Obtain these exact commit archives and retain their licenses:

| Name and archive filename | Pinned source | License |
| --- | --- | --- |
| GPU-NTT.tar.gz | [GPU-NTT commit](https://github.com/Alisah-Ozcan/GPU-NTT/tree/04b1103c35caa50455e17d932c0c1e59bd7e32d9) | Apache-2.0 |
| PhantomFHE.tar.gz | [PhantomFHE commit](https://github.com/encryptorion-lab/phantom-fhe/tree/1f4a198443b3af77118e51f53d5b8f332154b875) | GPL-3.0 |

`vendor-lock.json` records the exact gzip archive, license, and measured header SHA256 values. GitHub source archives at those commits can be saved under the filenames above. The resolver requires byte-identical archives; a differently packaged archive is rejected. Alternatively supply already extracted trees named `GPU-NTT` and `PhantomFHE` under one directory and verify their exact headers/licenses. The resolver never downloads. Do not remove upstream notices or treat external licenses as covered by the local artifact license. The local license is the artifact's anonymized license; the original and bundled hashes are distinguished in `source-provenance.json`.

From this `gpu/` directory, with both archives placed in `archives/`:

```sh
python resolve_vendors.py --archives archives --output vendors
python scripts/run_ntt_fixed_core_organization.py prepare \
  --vendors vendors --output campaign --nvcc nvcc --gpu-uuid "$GPU_UUID"
```

Set `GPU_UUID` to the UUID of the intended A100 before preparing. It is recorded but never queried during preparation. Existing source trees can instead be checked with `python resolve_vendors.py --verify --output VENDOR_DIRECTORY`, then passed using `--vendors VENDOR_DIRECTORY`. Preparation rejects an existing campaign; use a new output directory. Build logs and CPU metadata remain in that campaign. Prepared campaigns use absolute paths and hashes to freeze their environment, so prepare again after moving them.

Only when ready to use the GPU, set `DEADLINE` to an explicit future ISO timestamp with timezone and run:

```sh
python scripts/run_ntt_fixed_core_organization.py run \
  --campaign campaign/campaign.json --until "$DEADLINE"
python scripts/analyze_ntt_fixed_core_organization.py \
  --campaign campaign/campaign.json --output campaign/analysis
```

The runner executes 36 ordinary timing processes and 12 separate NCU captures under the included guard, preserving interrupted attempts. It waits for idle access, monitors foreign compute processes, and stops its own work on interference or deadline. It never terminates foreign jobs. Ordinary timing uses 100 warmups, 100 repeats per CUDA Graph sample, and seven samples per process. All requested values are checked exactly. NCU uses kernel replay, graph node profiling, and uncontrolled cache/clocks. A permission-blocked capture remains blocked and cannot make a complete Figure 11(a–d) analysis; ordinary performance is never inferred from profiler timings. These guard checks do not replace the site's scheduling/reservation policy.

To redraw after a complete analysis, from the parent directory:

```sh
python render_ntt_mechanism_profiles.py \
  --fixed-analysis gpu/campaign/analysis/analysis.json --output-dir new-cohort
```

This explicitly combines newly measured panels a–d with the frozen bundled local-layout data in e–f. Use `--layout-analysis` to supply a separately measured compatible layout analysis. Output source hashes identify both inputs. `verify.py` checks the frozen original export only; a new cohort is expected to differ. There is no GPU remeasurement runner for e–f in this package.

Validation performed for this bundle: compiler build passed, CPU preparation passed, and every prepared file (including input, twiddles, Shoup factors, exact reference, parameters, and metadata) matched the measured campaign's hashes. See `cpu-validation.json`. No GPU measurement was executed. Generated local campaigns contain device UUIDs and local paths: sanitize them before anonymous redistribution.
