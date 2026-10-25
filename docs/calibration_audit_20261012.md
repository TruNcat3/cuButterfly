# A100 校准与重测入口审计

本审计针对 2026-10-12 工作树，未执行 GPU 测量，也没有覆盖旧结果。

## 当前硬件身份

| UUID | 型号 | 显存 | 现有证据 |
| --- | --- | ---: | --- |
| `GPU-2b759faa-3e58-7106-10aa-ea6e387c3f8e` | NVIDIA A100-PCIE-40GB | 40 GiB | `results/local_hardware_profile_a100_gpu0_v09`，旧 build，不能用于当前 build 的时序 |
| `GPU-2971e500-d3f6-d7e6-e5f2-cb67a06fff02` | NVIDIA A100 80GB PCIe | 80 GiB | `results/parallel_resume_20261011_gpu1/r2/calibration`，research 模式，旧 build 身份 |
| `GPU-0eae5f07-33de-321b-70a9-ffeae49309e6` | NVIDIA A100 80GB PCIe | 80 GiB | `results/plan_batch_trace_20261010/recalibration_gpu2_finalized_bound_20261011`，research 模式，旧 build 身份 |

当前 `build-a100` 是 SM80 的新静态 build，默认编译策略为 `precompiled`，包含新的 planner/kernel 修改，但不包含 `libcuntt.so`，且 cuFFTDx 与 xD cuFFTDx 均关闭，不能代表完整核心集合。当前两个 80GB profile 使用的是 `build-e13-cufftdx`、research 模式和旧 binary/library fingerprint，因此只能复用语义/映射描述，不能复用 timing、stage service、composition 或 cost-model 系数。完整核心的重测需先重新构建共享的 `build-e13-cufftdx`，再冻结新的身份。

本日首次 reduced-core 校准保存在 `results/current_calibration_20261012_gpu1`，已停止，未完成 selector 推广或横评。原始数据保留，不导入完整核心的新校准。

## 必须重新测量的范围

需要在同一 UUID、同一 build 和同一 `CUBUTTERFLY_COMPILE_MODE` 下重新测量：

1. hardware capability probe and device identity
2. scheduling and pipeline scheduling
3. 1D stage service curves for the confirmed mapping population
4. operator mappings used by AUTO selection
5. composition calibration and local cost model

历史校准可作为 mapping seed 或 workload 清单输入，但不能作为 timing 或模型系数输入。旧的 GPU1/GPU2 calibration manifests 均为 `calibrated-local-serial`，composition 的 `inventory_complete=false`，且只覆盖有限 confirmed mappings。

xD 的公共 `MEASURE` 与 user-cache 重放是另一条现有校准接口。`calibrate_local_hardware.py` 的 workload 仍以 `logN` 为轴，不能用这条命令证明新 xD physical-cost 系数已标定或已被安装模型消费。xD 需要保存本轮完整 plan 的 MEASURE 结果，并验证 DEFAULT 命中同一 user cache；两种证据分开报告。

## 全库存入口参考（不是本轮有限任务）

以下全库存命令可能长时间运行，且只覆盖配置文件中的 1D 语义。
本轮采用独立的有限 workload 清单，补入短 FP16 FWHT，再对确认的候选补齐 stage/composition 和 selector 重放；不将有限任务标为全库存完成。

GPU 空闲后选择一个 UUID，建议先用 A100 80GB GPU1：

```bash
cd /home/wt/git/cuButterfly
export CUDA_VISIBLE_DEVICES=GPU-2971e500-d3f6-d7e6-e5f2-cb67a06fff02
export CUBUTTERFLY_COMPILE_MODE=research
export PYTHONPATH=/home/wt/git/cuButterfly/scripts
export CAL_ROOT=/home/wt/git/cuButterfly/results/local_hardware_profile_a10080_20261012_v1
export CBPY=/home/wt/yes/envs/cubutterfly/bin/python
export CUBUTTERFLY_REGISTRY="$CAL_ROOT/registry.json"

"$CBPY" scripts/calibrate_local_hardware.py \
  --build-dir /home/wt/git/cuButterfly/build-e13-cufftdx \
  --profile-dir "$CAL_ROOT/hardware" \
  --output-dir "$CAL_ROOT/calibration" \
  --force \
  --trials 3 \
  --operator-trials 3 \
  --operator-warmup 100 \
  --operator-repeat 100 \
  --cost-model staged \
  --stage-calibration full \
  --stage-point-budget 0 \
  --search-workloads /home/wt/git/cuButterfly/config/install_search_workloads.json \
  --search-budget 64 \
  --search-finalists 3 \
  --seed-budget 8 \
  --search-stage-projection full \
  --search-strategy model
```

GPU2 可用时只替换 `CUDA_VISIBLE_DEVICES` 和输出根目录，例如 `local_hardware_profile_a10080_gpu2_20261012_v1`。不要把两个 UUID 的结果写到同一目录。

校准完成后先检查：

```bash
"$CBPY" scripts/check_hardware_profile.py "$CAL_ROOT/hardware/hardware_profile.json"
CAL_ROOT="$CAL_ROOT" "$CBPY" - <<'PY'
import json, pathlib
import os
p = pathlib.Path(os.environ["CAL_ROOT"]) / "calibration/calibration_manifest.json"
d = json.loads(p.read_text())
assert d["status"] == "ready-with-measured-mappings"
assert d["stage_service_calibration_status"] in {"complete", "complete-bounded", "complete-sampled"}
assert d["stage_inventory_complete"] is True
assert d["device"] == "NVIDIA A100 80GB PCIe"
print(p)
PY
```

## 重新测性能的最小集合

校准 manifest 通过后，只把其 registry 和 profile 绑定到同一 build/UUID 的新输出目录：

* E1：统一算子横评，使用 manifest 中记录的实际 registry 和新的 `cost_model.json`。
* E2：五个固定 profile case，重新记录 trace/NCU，不复用旧 binary 的计数器数据。
* E3：FNO、QuaRot、一个 FFT 图像滤波应用，使用 batch 模式；端到端结果与 operator kernel 时间分开记录。
* xD：coverage、scale、C2R copy-elision 三组，必须保留 native 与 adapted 两列，并计入转换开销。

`build-a100` 当前只有静态 `libcuntt.a`。依赖 `--library build-*/libcuntt.so` 的旧 xD/C API 入口不能直接指向它；应先生成与当前工作树完全一致的 shared SM80 build，或使用当前 build 中的静态 benchmark executable。任何 shared-library build 都必须让 calibration identity 重新生成，不能复制旧 `.so`。

## 主要风险

* `check_hardware_profile.py` 按 GPU 名称和可见设备检查，显存容量由 calibration script 再次比对；不同容量必须使用不同输出目录。
* `CUBUTTERFLY_COMPILE_MODE` 会进入 measurement identity。校准和重测都固定为 `research`，不要让 build 默认的 `precompiled` 与环境变量混用。
* 当前旧 profile 的 composition 覆盖不完整，不能宣称完整并发重叠校准。
* 新 xD cost fields 只在当前工作树重新测量后才有硬件系数；旧 cost model 仅可作为格式参考。
