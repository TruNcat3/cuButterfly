# QuaRot 性能归因审计（2026-11-07）

本次仅 CPU 读取已有证据，未跑 GPU、未访问外部数据盘。完整证据 SHA256、逐进程计划及原始报告定位见 [派生 JSON](quarot_attribution_20261107.json)。结论针对 Nov2 `fac10905…` 构建；不能自动解释 Nov5 安装库的性能。

## 可以确认的结果

| B / process | baseline wall ms | treatment wall ms | speedup | GPU event 增量 ms |
| --- | ---: | ---: | ---: | ---: |
| 1 / 1 | 154.200125 | 159.228199 | 0.968422 | 5.071884 |
| 1 / 2 | 154.167583 | 159.196640 | 0.968410 | 5.036026 |
| 4 / 1 | 177.165389 | 185.186769 | 0.956685 | 8.020981 |
| 4 / 2 | 177.594403 | 185.147971 | 0.959203 | 7.554047 |

来源是 `results/a100_closure_20261102/applications/quarot_b{1,4}_p128d8_process{1,2}/release/report.json` 的 `summary.end_to_end` 与两变体的 `cuda_event_ms.end_to_end` 中位数。两进程交换变体顺序仍有倒退。GPU event 同样增加，因此不能仅归因于 Python/FFI 提交成本。

`paper/experiments/e13/run_quarot_random_end_to_end.py:253` 的 `_timed` 同时记录 wall 与 GPU event，开始前同步，末尾再次同步；`:704` 用 **wall 中位数** 计算主表。`:352` 的 timed_replay 在 Graph replay 前执行两个静态输入 `copy_`，之后 clone 全部九个 logits。`:320` 捕获一次完整 prefill 加八次 decode；`:365` 的 capture/setup、计划创建和验证在计时外。host wall 包含 Python bookkeeping、event 建立与同步；GPU event 包含输入 copy、Graph 及输出 clone，也可能包含 host 提交间隙。它不是仅 FWHT 时间，也不是纯 Graph replay 时间。

## 实际替换合同与主横评不匹配

架构定义见 runner `:109`（4096 hidden、11008 intermediate、32 层、32 heads）。上游 `results/e13_assets/sources/QuaRot/e2e/quantized_llama/modeling_llama.py:82` 在 head 轴转置后调用 attention Hadamard，`:99` 定义 N=32；`:112` 定义 MLP Hadamard。`quarot/functional/hadamard.py:7` 将 11008 分解成 K=172 × N=64。

| phase | N | B1 flattened batch | B4 flattened batch | full normalization dimension | 每序列调用数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| prefill attention | 32 | 16384 | 65536 | 32 | 32 |
| prefill MLP | 64 | 22016 | 88064 | 11008 | 32 |
| decode attention | 32 | 128 | 512 | 32 | 256 |
| decode MLP | 64 | 172 | 688 | 11008 | 256 |

batch 与 N 直接来自报告 `adapter_setup.plans[*].descriptor.semantics`。调用数是 32 层 ×（1 prefill 或 8 decode）静态推导，非 trace 实测；不含未替换的 KV-cache Dao 调用。各变体仍执行相同的 KV-cache/量化/矩阵乘等工作。

主横评 FP32 Dao 的十个配对是 `[256]/B1`、`[4096]/B32`、`[32768]/B1` 各正逆向，以及 `[128,128]/B4`、`[32,32,32]/B4`、`[8,8,8,8]/B4`、`[2,2,2,2,2,2,2,2]/B16`。另有 `[65536]/B4` 但 Dao 不支持，未进入十点均值。来源是 `results/full_current_20261025/measurement/results.json` 所索引逐行文件，路径及 hash 已收录 JSON。**没有与应用实际 rank-one N32/64 和 batch 匹配的请求**；多维 shape 含 32 并不等于应用 N32。1.474× 是这个不同集合的几何平均，不能赋给应用核心。

主横评 runner `scripts/semantic_gpu_pipeline.py:221` 对 out-of-place 路径捕获20次执行，七个 event 样本各除20；与应用一次完整序列的 wall 边界不同。

## 已证实的适配成本，以及尚未量化的贡献

baseline 是 QuaRot 复合公式：`quarot/functional/hadamard.py:92` 调用 native Dao 对连续 FP16 输入执行 FWHT 并传入 `1/sqrt(full_n)`，K>1 后执行 `hadK @ input`。它不是把整个 QuaRot 复合操作都替换为裸 Dao 的单算子对比。

Nov2 treatment 明确为 `fp32_composed`。`paper/experiments/e13/quarot_adapter.py:435` 先保证 contiguous，再 `.float()`；`:450` 执行未归一化 FP32 FWHT；`:459` 单独 `mul_`；`:467` 通过 `copy_` 窄化到复用 FP16 输出；`:503` 在 K=172 路径继续同样的 hadK matmul。attention 转置后的连续化在双方都存在；不能把全部布局代价算成 treatment 新增成本。K 矩阵 `.to()` 若 device/dtype 已匹配可以无复制，未测不能按真实复制计数。输出复用减少分配但不会消除上述类型转换和缩放的数据访问。

全部八个实际合同选择 `temporal-tile-radix4` 或 `temporal-tile-radix8`，mapping 为 rank-one packed、natural order，未显示额外边界置换。逐进程选择不同，见 JSON。报告的 `selection_reason.kernel_ms` 是计划搜索时记录的单侧候选测量，**不是 matched Graph 分项 trace**，不能将其乘调用数作为已证实的应用核心占比。也没有证据确认执行时额外通用 pack/permute 的次数或耗时。

当前读取的应用报告和历史诊断没有可对应上述四类合同、同一构建/Graph/归一化/布局、同时覆盖双方的分项 trace。因此可确认转换与独立缩放存在，尚不能判断它们、短 N 核心效率、布局或其他交互各解释多少毫秒。Amdahl 只能限制局部收益的传播；仅说“核心占比低”不能解释整体倒退的正增量，更不能替代 matched 测量。历史 `docs/e2e_framework_hotpath_20261021.md:11` 对固定开销的宽泛解释不能作为本次负结果的定量证据。

## 公共接口已有融合能力；最小后续点

当前 `src/c_api.cpp:2962` 已允许 standard rank-one FP16 storage、FP32 compute、FINAL_STORAGE rounding 和 output_scale；`:4681` 把 epilogue 参数传入 common Butterfly 配置。E13 adapter 已有 `public_fp32_final`：`:294` 保留 FP16 storage 并设 FP32 compute，`:322` 设置最终舍入和输出缩放；`:451` 直接返回核心结果，跳过外部 cast/mul/narrow。runner CLI `:470` 已允许此合同。**无需先新增 kernel**；Nov2 queue 仍明确使用旧 `fp32_composed`，当前接口能力不能追溯视为旧库已经融合。

建议只做两个 isolated matched 点：B1 prefill MLP `(batch=22016,N=64,scale=1/sqrt(11008))`，B1 decode attention `(batch=128,N=32,scale=1/sqrt(32))`。固定原 FP16 输入及原 attention 非连续布局，分别对 native Dao、旧 composed、现有 public_fp32_final 捕获相同 Graph，报告裸核心/完整适配的 event 时间、实际 kernel 数及每段 copy/cast/scale/排列时间，K=172 点保留相同 hadK matmul。明确 GPU/库/runner hash，先验证输出误差与完整调用语义，再判定是否需要将新隔离运行参数切到融合分支；保留旧报告。

工程已证实：现有应用走旧 composed；公共接口和 adapter 已具备融合接入。工程可疑：短 N 的额外带宽和 launch 成本、当前选择的短 N 核心效率。无法判定：各分项占比以及融合接入能否回收全部约5–8ms。没有发现需要本审计直接修 kernel 的已证实缺陷。

## 当前安装库的八点隔离诊断尝试

2026-11-07 使用 `scripts/benchmark_quarot_contracts.py` 与 `configs/experiments/quarot_contracts_20261107.json`，固定当前安装库 `results/a100_fp16_acceptance_20261105/build/libcuntt.so`。CPU 合同、稠密 Hadamard oracle 和拒绝覆盖测试共3项通过，`--help` 与语法检查通过。脚本复现 attention 的非连续 `[B,S,128,32]` 输入，比较 Dao、旧 composed 与已有 public fused 路径；Graph event 只覆盖完整变换子路径，包含需要的 contiguous/cast/scale，不含 hadK matmul、输入 copy、输出 clone 或7B模型。

有界隔离运行命令如下；`--until` 是本次记录的绝对截止时间，后续重放须给新的显式期限及新结果目录：

```bash
env -u CUBUTTERFLY_HARDWARE_MODEL -u CUBUTTERFLY_PROFILE_ROOT \
  -u CUBUTTERFLY_REGISTRY -u CUBUTTERFLY_PLAN_CACHE \
  -u CUBUTTERFLY_PLATFORM_TUNING -u CUBUTTERFLY_JIT_CACHE \
  -u CUBUTTERFLY_E13_ALGORITHM_POLICY CUBUTTERFLY_COMPILE_MODE=research \
  /home/wt/yes/envs/cubutterfly/bin/python scripts/run_isolated_gpu.py \
  --gpu-uuid GPU-2971e500-d3f6-d7e6-e5f2-cb67a06fff02 \
  --report-dir results/quarot_contracts_20261107_80g --idle-seconds 10 \
  --until 2026-11-07T11:13:58.520711+00:00 -- \
  /home/wt/yes/envs/cubutterfly/bin/python scripts/benchmark_quarot_contracts.py \
  --library results/a100_fp16_acceptance_20261105/build/libcuntt.so
```

显式继承的硬件/profile/registry/cache/platform变量已清除，保留正常默认路径。默认 registry 有分开的80GB/40GB目标；首 attempt 的 runner 日志确认实际选择 `config/hardware/NVIDIA-A100-80GB-PCIe-sm80-84974239744B/hardware_model.json`。

最终结果：guard 退出码2，状态 `expired`，没有 `release`。首 attempt 于19:08:26启动，19:09:24因新增外部 PID3631472被 guard 中断并隔离，此后持续等待直至截止。状态写于19:14:12（guard等待轮询可能略晚于19:13:58绝对截止）。没有可接纳的八点时间、kernel counts或数值误差，不能更新为融合已回收多少成本。隔离记录与工具/配置 SHA256已追加到派生JSON；现有约5–8ms归因边界保持不变。
