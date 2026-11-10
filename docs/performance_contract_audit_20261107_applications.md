# 应用与固定核心性能合同审计（2026-11-07）

限定 CPU 读取持久证据及重算；未运行 GPU、未改原始记录、初轮未审计 QuaRot；文末追加四个完整7B clean进程的CPU证据核验。依据 AGENTS.md 和 research_principles.md 中同语义、同核心、匹配调用及有限候选原则。结果不能作为重新验证数值正确性或源代码与历史二进制完全对应的声明。机器可读重算在同名 JSON。

## 确定发现

1. **初轮发现的 CKKS mean/median 标签问题已修正。** 初轮 `experiments/snapshots/current_refresh/snapshot.json:aggregation.ratio` 写 complete-call median；`experiments/workloads/heongpu_ckks_e2e_bench.cu:819–845,898–902` 实际逐重复累加 sample_ms，CSV 写 `times/reference repetitions`，比值为样本算术均值之比。导出器 `export_current_refresh.py:325–345` 直接读 CSV，不转换为中位数。数字未因此算错。最终 snapshot 已改为每行匹配统计量、时钟和边界，并注明 NTT5 的 paired median；CKKS 均值不再被全局称为 median。历史二进制对应关系仍限制这项源码解释。
2. **FNO 正收益只在 Graph wall time。** `results/e2e_unified_20261013/fno_hotpath_summary.json` 两独立进程、B4、64²/128² 的 Graph wall ratio 为 1.0299315353/1.0564188968；eager 为 0.9554150066/0.9573577328。两者都完整包含设备输入 copy、forward/replay、独立输出 clone、同步；setup 排除。`export_application_closeout.py:39–50` 分开导出，`application_cohorts.tex` 只显示 Graph，故文字必须显式限定 Graph。不能把它解释为 eager 改善、训练性能或模型质量。旧 `application_historical_complete_call.json` 的 B8 64² 和 B4 128² 为另一个历史 cohort，不能拼接为同批次的 before/after。
3. **bootstrap 不是安全参数生产性能。** Nov2 `application_closeout.json` 的 bootstrap B1/B8 为 1.0333320794/1.0315936768，记录 `sec_level=none` 和 `source_binary_correspondence=not_verified`。CKKS runner:434 显式用 none；`exact_ciphertext=1` 证明替换输出一致，不证明安全性或安全参数性能。当前 refresh 的两点约 1.030/1.032 与 Nov2 是不同来源，不宜混算单一刷新均值。

## 各应用合同

| 应用 | 匹配及开销 | 判断 |
| --- | --- | --- |
| FNO | `run_fno_graph_integration_20261013.py:127–156` 同 FP32 输入、deepcopy 同权重，完整 model，双方 copy/clone；摘要还核对权重不变、Graph 对本后端 eager、输出和 mapping hash | 未见单侧省略输出物化；随机权重仅是性能输入 |
| FFT filter | `run_fft_filter_end_to_end.py:460–510` 双方固定共享空间滤波器、谱预缓存排除；自方 forward normalization 补偿一次折入滤波谱，CuPy 默认 norm；同 FP32 输入/complex64，same crop | 支持固定滤波器合同，不支持每调用更新滤波器的外推 |
| FFT filter Graph | 同文件:212–277 双方设备输入 refresh（包括固定 kernel 的冗余 copy）、Graph replay、crop 结果 clone/copy；Graph capture 排除 | 输出所有权和适配开销两侧一致 |
| FFT filter padding | 自方 power-of-two/指定 padding，CuPy `next_fast_len(...,real=True)` 原生 padding | 同卷积结果，不是相同 FFT 内部 shape；对原生应用基线合理，不能解释成同尺寸 FFT 算子加速。torch same-padding 明确 diagnostic，不计外部结果 |
| Poisson | `run_periodic_poisson_graph_20261013.py:145–190` 同 CPU FP32 RHS、同频率 multiplier、相同 norm、完整 rfftn→乘子→irfftn，双方输入 refresh 和独立输出 copy | Graph 对 Graph；export_current_refresh 取 CUDA-event median，不能和 FNO wall median 当同一种绝对延迟 |
| CKKS | runner:667–677 检查 stage batching 多项式工作数；native loop、native stage batch、cu auto 各自 Graph；:819–845 轮转测量顺序 | loop 包括批提交方式差异；matched stage batch 更能隔离集成方式。不能把两列当 72 个独立 workload |

CKKS `ckks_table.tex` 每 batch 12 个请求，共 36 个；每请求两基线，共 72 个比值。CPU 从 snapshot 的 baseline_ms/own_ms 独立重算几何均值：

| B | 请求数 | Native loop | Native stage batch | 每请求更快 native |
| --- | ---: | ---: | ---: | ---: |
| 1 | 12 | 1.0357833503 | 1.2901971025 | 1.0357833503 |
| 8 | 12 | 1.6478245379 | 1.1834041765 | 1.1816571244 |
| 32 | 12 | 1.7053619160 | 1.1821900091 | 1.1821900091 |

与生成表四舍五入一致。更快 native 是每请求取最小 baseline latency 后聚合，不是两列几何均值再取 min。未发现该分母计算错误。此重算验证导出算术，未逐一重建所有 Graph 节点语义。

## 固定核心六组

直接审计 `results/full_current_20261025/mechanism/timing.json` 全六 comparison 的 baseline/treatment resolved_mapping。FFT/FWHT 仅 `stage_partition` 和随分段变化的 `boundaries` 不同；NTT 仅 stage_partition 不同。核心字段没有换核证据。双方同 workload、warmup/repeat，三 paired trials 且顺序交替，是有限手工组织对照，未发生不同搜索预算的自动 winner 归因。

独立计算 paired ratio 的 median 为 FFT 11.545817/21.406784、FWHT 13.674326/24.096421、NTT 11.056669/19.835448，均与当前 snapshot 一致。第一点 ratio of median times 实际为 11.527190，不能混作 paired median；原始 comparison 的 `median_treatment_over_baseline` 命名/数值存在历史混淆，当前导出正确地从 pairs 重算。

资源记录包含 `compiler_resources_known=false`、`shared_components_known=false` 和 preflight `performance_accepted=false`，而最终 comparison 为 measured/performance_accepted=true。实测成功不能补全静态资源证明；本次未发现明确非法候选，但也不能认证全部 lowerings/候选合法或全局最优。stagewise 是极细粒度同核心控制，11–24× 是对此内部基线的组织收益，不是对最佳端点、cuFFT 或最优组织的加速。未审计所有历史 factor_graph 扩展实验，不外推本六组结论。

## 本轮验证与未闭合项

使用 Python 标准库重算 36 请求/72 CKKS 比值的三批次几何均值，以及六组共 18 paired trial；未运行生产脚本或 GPU。没有发现应用表数值的确定性除法/分母错误，初轮 CKKS mean/median 标签问题已由最终导出修正，timing/security/build 边界仍须保留。历史二进制与源码对应、CKKS 所有输出物化细节、完整候选资源合法性仍需独立证据，未在本轮声称闭合。

## 完整 7B 最终结果（2026-11-07）

四个独立 clean 进程已经完成。来源为 [生成证据](../experiments/snapshots/current_refresh/quarot_fused_e2e.json)，其中记录逐进程原始报告路径、SHA256、库与 runner 身份、全部 wall/event 样本和数值误差。CPU 核验逐份原始报告 hash，并从每变体各7个样本重算两时钟中位数及其比值；下表与生成证据一致。

A100 80GB PCIe，UUID `GPU-2971e500-d3f6-d7e6-e5f2-cb67a06fff02`；库 SHA256 `775a85c8f76e44bbfee886fd90489e7c309bd097b60e968183e446aa006177b2`。固定32层随机权重 Llama-2-7B、seed20261003、prefill128/decode8、SDPA prefill、验证预热4次、计时预热2次、每变体7次完整 Graph 调用。边界为设备输入 refresh、完整 prefill/decode、独立 logits clone 和同步，wall 包含 host 调用；setup、计划创建与 Graph capture 排除。输入输出 FP16 storage、FP32 compute、final FP16 writeback；实际走 `public_fp32_final`。该路径已成为默认值，旧 native/composed/eager 可显式重放。

| B / 顺序 | wall 基线 / 自方 ms | wall 比值 | event 基线 / 自方 ms | event 比值 |
| --- | ---: | ---: | ---: | ---: |
| 1 / baseline-first | 154.198675 / 153.553649 | 1.004201 | 154.169342 / 153.520126 | 1.004229 |
| 1 / treatment-first | 154.220778 / 153.519231 | 1.004570 | 154.175491 / 153.489410 | 1.004470 |
| 4 / baseline-first | 177.825060 / 174.839853 | 1.017074 | 177.782791 / 174.736389 | 1.017434 |
| 4 / treatment-first | 177.559008 / 174.942173 | 1.014958 | 177.528839 / 174.910461 | 1.014970 |

四进程各9组跨后端 prefill/decode 输出均 finite，最大绝对误差和相对 L2 误差均0；同后端 Graph 原输入/改输入/恢复及输出生命周期检查通过。未指定应用误差阈值，状态保留 `pass_timing_numerical_unassessed`；随机权重测量不提供模型质量证据。四个 clean 进程含第四任务的显式后续窗口，原窗口中断 attempt 仍是历史中断证据，不计作第五个样本；最终不存在待运行任务。

B1 wall 比值为1.004201–1.004570，B4为1.014958–1.017074；每批次保留两种顺序的范围，不挑最快进程、不与旧 cohort 池化，不声称统计显著。八点组件对照更大的局部收益只部分传至完整模型：组件计时不含 K=172 matmul、完整模型、输入 refresh 或输出 clone，不能称作同边界的端到端结果。旧 composed 选择 radix4/radix8，public fused 选择 radix2；这是公共选中路径的工程改进，改变了局部核心及外围操作，不能声称纯融合或固定核心因果消融。

旧 Nov2 composed 负结果仍保留：B1约0.9684×，B4约0.9567–0.9592×；它们属于不同构建与 cohort。新结果证明当前公共路径的完整应用表现，不能把跨构建变化定量拆成5–8ms的纯融合贡献。无需扩展本轮矩阵或追加 GPU 任务。

## 历史四点的最终资格核对

初次核对时，`experiments/snapshots/application_historical_complete_call.json` 仅有四舍五入延迟和比值，不含 source/hash/数值状态。最终数据文件已保留四行原顺序，补充 source/hash 与 performance_eligible：两旧 FNO 为 true（仅历史构建），两旧 QuaRot 为 false（历史归档，不用于有效性能证据）。FNO 已改用原始精确中位数及比值；两旧 QuaRot 数字原样保留。`experiments/tools/render_application_historical.py:20–25` 硬编码这些数字。以下仅核对这四点；原始文件未修改。

| 点 | 原始报告（相对仓库） | 原始报告 SHA256 | 资格 |
| --- | --- | --- | --- |
| QuaRot 512/32 | `results/paper_profile_rc2/end_to_end/quarot_random_prefill512_decode32_sdpa.json` | `b739603dd20b2e10a56ea6db163362edbb7142f718b9896142b494d042392f45` | 排除有效应用性能证据；保留历史归档 |
| QuaRot 2048/128 | `results/paper_profile_rc2/end_to_end/quarot_random_prefill2048_decode128_sdpa.json` | `21c561ccca1247b6b4063ee3dd9c20db0bc8e15e95ed68c050b2e79487581827` | 排除有效应用性能证据；保留历史归档 |
| FNO 64²/B8 | `results/paper_profile_rc2/end_to_end/fno_darcy_64_v9.json` | `447e417e301083c0c64d7a6124736d799ec8dc78b54a41846d8b0d40d11d8bd0` | 可保留为历史构建负结果 |
| FNO 128²/B4 | `results/paper_profile_rc2/end_to_end/fno_darcy_128_v9.json` | `6e227576bd97041026c3d86fc26b67d4af802588585064f100efb1447671699a` | 可保留为历史构建负结果 |

四份 hash 均与 `experiments/provenance/application_snapshot_20261009.json` 匹配。旧 QuaRot 原始中位数比值确为1.0080077398/1.0028010053，算术没有挽救证据资格：raw 仅有 `status=pass` 和同 seed 声明，没有 state hash/summary、finite/prefill/decode output checks、runner/adapter hash。`README_quarot.md` 的修复前历史段明确声明 invalid random state；`paper/experiments/targeted_evaluation_followup_20261012.md:133–136` 进一步说明旧 Linear4bit 零 scale 与 cache-position 兼容问题使其不具备应用性能证据资格。因此这两个点应退出有效性能证据图，原始数字保留为失效历史记录。确证的是来源、缺少检查和项目修复记录；无法从这些 raw 文件确定每个输出当时是否 NaN，也不能重建缺失的 runner/adapter 源码身份。

旧 QuaRot library 为 `90a63df0b568aa0d76231ebd76cecbfcbd16656c06c8ab4e39587069787132e4`。修复后的 `results/paper_targeted_20261012/quarot/full512_validation.json` 是另一身份：library `24486975585fa2ab6399a0cfeae4b9449c398490cb76c5ea5a73a3560abe8c66`、`fp32_composed`、双方 state SHA `2ea312e2a11a3b032cbac425fe23cf2dee9a890696926b9ab4831465d5dfa638`，1,359,872个 scale 全部正且 finite，状态 `pass_validation`。后来的修复验证不能追溯认证旧两个时间点。

两旧 FNO 原始 `pass=true`、`quality.gate_before_timing=true`，冻结 gate 为 maxabs/relative-L2≤1e-4；实测 maxabs 均8.3446503e-7，relative-L2分别1.0560814e-7/1.3240302e-7。有 library `90a63df0…`、runner `bfaa1170…`、adapter `a6581447…`、checkpoint/input hash。CPU 从各三进程 raw samples 重建 process median 再取中位数：64² 为3.0065239407/3.1516216695ms、ratio0.9539609306；128² 为2.9897741042/3.1403121538ms、ratio0.9520627115。历史生成 JSON 先把延迟 round 到三位再相除，数值略偏但三位显示仍0.954/0.952。应直接使用精确原始中位数并保留 source/hash 与历史构建标签；这两点不证明当前构建性能。
