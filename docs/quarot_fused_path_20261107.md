# QuaRot fused route and performance reporting, November 7

## Matched operator paths

Eight predefined application requests were measured on A100 80GB PCIe with
FP16 input/output, FP32 arithmetic and final FP16 storage. Each CUDA graph
contains 20 complete transform calls, with seven rounds of 100 replays.
Initialization, compilation, CPU reference and profiler runs are outside timing.
All 24 variant outputs were finite and exactly matched the tested CPU FP32
reference after final FP16 storage; no floating acceptance threshold was added.
This is a finite functional observation, not a universal rounding-equivalence proof.

| Request | Dao (us) | Old composed (us) | Public fused (us) | Dao/fused |
|---|---:|---:|---:|---:|
| b1_prefill_n32 | 20.504 | 31.312 | 10.644 | 1.926x |
| b1_prefill_n64 | 19.230 | 36.657 | 6.091 | 3.157x |
| b1_decode_n32 | 4.198 | 12.218 | 4.091 | 1.026x |
| b1_decode_n64 | 2.126 | 10.202 | 1.854 | 1.147x |
| b4_prefill_n32 | 69.322 | 93.538 | 32.073 | 2.161x |
| b4_prefill_n64 | 68.652 | 148.422 | 18.762 | 3.659x |
| b4_decode_n32 | 4.728 | 13.027 | 4.405 | 1.073x |
| b4_decode_n64 | 2.437 | 10.814 | 1.984 | 1.228x |

The fused mapping is one stage, tile 32 or 64, 128 threads, temporal-tile radix-2,
FP32 computation with final FP16 rounding and the actual full-dimension scale.
The old composed route selects temporal-tile radix-4 and adds separate widen,
scale and narrow operations. These are public selected-path comparisons, not
a fixed-core fusion ablation: the speed difference cannot be assigned solely
to removed launches. Both attention variants include their input contiguous copy.

Profiler kernel names identify two versus five transform-related kernels per
N32 call (Dao/fused versus composed), and one versus four per N64 call.
Every profile also contains two `FillFunctor<long>` activities per graph whose
origin was not independently identified; the raw 42/102/42 and 22/82/22 counts
remain intact. They are not silently removed from total profile time.

The first single-call-graph diagnostic remains archived: host submission gaps
limited its short-request timings, and inference-mode mismatch broke its profiler.
The revised runner preserves inference mode, records calls per graph and writes
partial checkpoints. Partial or interrupted attempts do not become accepted releases.

The complete matched report, including all rounds, mappings and profiler events,
is [frozen here](evidence/quarot_matched_20261107.json). It excludes the K=172
matrix multiply, full model, input refresh and output clones. Full-model results
are recorded separately; this table is not an application speedup table.

## Other performance contracts

The [operator audit](performance_contract_audit_20261107_operators.md) reproduces
the main 49 external pairs, cyclic54 and FP16 nine-pair aggregates. The
[application audit](performance_contract_audit_20261107_applications.md) reproduces
CKKS batch aggregates and the six fixed-core controls. No headline ratio was
changed by this audit. Exported rows now retain complete semantics, adapter
operations and individual timing protocols.

CKKS and bootstrap use device-event sample means; FNO/QuaRot use wall medians,
and filtering/Poisson use event medians. There is no pooled application mean.
The main operator table now counts native, adapted and axis-composed comparisons.
Main FWHT's 1.474x includes copying Dao's returned tensor into a caller-supplied
output; it does not establish a bare-core gain. At this November 7 audit checkpoint, a returned-tensor comparison
for those same ten requests was unmeasured. The subsequent
[FWHT native-return repair and public DEFAULT check](fwht_native_repair_20261107.md)
closes that gap: three-process matched replay gives native-return GM
1.013–1.031x and supplied-output GM 1.320–1.330x; the separate single-process
public DEFAULT check gives 1.009x/1.324x. These later FP32 ten-request results
retain their own build and contract and do not replace the FP16-storage QuaRot
subrequests or full-model measurements. The matched QuaRot subrequest
diagnostic above already consumes Dao's returned tensor directly.

## Relation to the full model

A diagnostic projection uses the architecture's 32 layers and eight decode
steps, without fitting a parameter: sum 32 times each prefill subrequest and
256 times each decode subrequest. This predicts old composed-path excess
of **5.024 ms (B1)** and **7.597 ms (B4)** over direct Dao, and fused-path
savings of **0.833 ms (B1)** and **2.987 ms (B4)**. These are standalone-time
projections, not measured whole-model attribution. Historical application GPU
increments of roughly 5–8 ms are on a different build and are only a consistency
check; they cannot establish exact causal shares.

The predicted fused savings are small compared with complete 7B calls of
roughly 154–178 ms. The Hadamard K-factor multiplication, quantization,
attention, projections and other model operations remain in both variants.
Their unchanged cost limits propagation of the local improvements.
[Projection inputs and arithmetic](evidence/quarot_subpath_projection_20261107.json)
are recorded separately from measured application rows.

## 完整 7B 最终结果（2026-11-07）

四个独立 clean 进程已经完成。来源为 [生成证据](../experiments/snapshots/current_refresh/quarot_fused_e2e.json)，其中记录逐进程原始报告路径、SHA256、库与 runner 身份、全部 wall/event 样本和数值误差。CPU 核验逐份原始报告 hash，并从每变体各7个样本重算两时钟中位数及其比值；下表与生成证据一致。

A100 80GB PCIe，UUID `GPU-2971e500-d3f6-d7e6-e5f2-cb67a06fff02`；库 SHA256 `775a85c8f76e44bbfee886fd90489e7c309bd097b60e968183e446aa006177b2`。固定32层随机权重 Llama-2-7B、seed20261003、prefill128/decode8、SDPA prefill、验证预热4次、计时预热2次、每变体7次完整 Graph 调用。边界为设备输入 refresh、完整 prefill/decode、独立 logits clone 和同步，wall 包含 host 调用；setup、计划创建与 Graph capture 排除。替换的 FWHT 路径采用 FP16 storage、FP32 compute、final FP16 writeback（并非宣称整个模型使用 FP32 运算）；实际走 `public_fp32_final`。该路径已成为默认值，旧 native/composed/eager 可显式重放。

| B / 顺序 | wall 基线 / 自方 ms | wall 比值 | event 基线 / 自方 ms | event 比值 |
| --- | ---: | ---: | ---: | ---: |
| 1 / baseline-first | 154.198675 / 153.553649 | 1.004201 | 154.169342 / 153.520126 | 1.004229 |
| 1 / treatment-first | 154.220778 / 153.519231 | 1.004570 | 154.175491 / 153.489410 | 1.004470 |
| 4 / baseline-first | 177.825060 / 174.839853 | 1.017074 | 177.782791 / 174.736389 | 1.017434 |
| 4 / treatment-first | 177.559008 / 174.942173 | 1.014958 | 177.528839 / 174.910461 | 1.014970 |

四进程各9组跨后端 prefill/decode 输出均 finite，最大绝对误差和相对 L2 误差均0；同后端 Graph 原输入/改输入/恢复及输出生命周期检查通过。未指定应用误差阈值，状态保留 `pass_timing_numerical_unassessed`；随机权重测量不提供模型质量证据。四个 clean 进程含第四任务的显式后续窗口，原窗口中断 attempt 仍是历史中断证据，不计作第五个样本；最终不存在待运行任务。

B1 wall 比值为1.004201–1.004570，B4为1.014958–1.017074；每批次保留两种顺序的范围，不挑最快进程、不与旧 cohort 池化，不声称统计显著。八点组件对照更大的局部收益只部分传至完整模型：组件计时不含 K=172 matmul、完整模型、输入 refresh 或输出 clone，不能称作同边界的端到端结果。旧 composed 选择 radix4/radix8，public fused 选择 radix2；这是公共选中路径的工程改进，改变了局部核心及外围操作，不能声称纯融合或固定核心因果消融。

旧 Nov2 composed 负结果仍保留：B1约0.9684×，B4约0.9567–0.9592×；它们属于不同构建与 cohort。新结果证明当前公共路径的完整应用表现，不能把跨构建变化定量拆成5–8ms的纯融合贡献。无需扩展本轮矩阵或追加 GPU 任务。
