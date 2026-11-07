# 算子性能合同 CPU 审计（2026-11-07）

范围：主 66 请求、独立 cyclic 90 请求／54 外部配对、FP16 九点公共 DEFAULT。仅读取本地 JSON 与代码并重算；未运行 GPU、校准或全工程测试，未访问 sdb，未改 paper 或生产 runner。遵循 `AGENTS.md` 与 `docs/research_principles.md` 第 2–4 节的语义配对、完整调用和有限覆盖原则。

## 已确认

通过动态导入 `paper/tools/export_current_refresh.py`，CPU 重建 `Exporter.operators()` 与 `Exporter.cyclic()`，分别与快照对应对象逐字段完全一致；这同时执行原始 row/index/payload 哈希及 cyclic 补测来源、descriptor、GPU UUID、protocol 等检查。未重新读取庞大误差 NPZ 或输出 NPY，数值正确性沿用其原始报告。

主队列有 66 个唯一请求、49 个数值外部配对；不是 66 个全部已有外部基线。以下 GM 为逐请求 baseline complete-call median / own complete-call median 的几何均值；不同 baseline 对同一请求各成一对，不当作独立请求。

|算子|精度|基线|对数|GM|名义更快|
|---|---|---|---:|---:|---:|
|fft|fp32|cufft|11|1.297913164|9|
|fft|fp32|vkfft|11|1.190290199|5|
|fft|fp64|cufft|6|1.366218227|5|
|fft|fp64|vkfft|6|1.353940859|6|
|fwht|fp32|daofht|10|1.474332030|9|
|ntt|word64|gpu-ntt|5|1.188155691|4|

以上与 `paper/generated/current_refresh/operator_table.tex` 三位小数一致。NTT 五对的 source 都是 `measurement/rows/<rowid>.json`，cohort 为 main66；`paper/tools/render_current_refresh.py:71` 仅筛 main66，没有用 ntt5 覆盖。初步怀疑 repair 替换经核实排除。JSON 的 `main_ntt_source_mapping` 给出逐点定位；`ntt5` 与 `fft_grouped2` 独立存储，不进入主表。

## 完整调用与分组

`scripts/run_semantic_campaign.py:697` 起构建语义匹配的 vendor 对照，`:755` 起登记 adapter operations，`:777` 起 execute 包含 gather、核心、缩放和 scatter。高 rank FFT 的轴组合包含重排拷贝、逐组 vendor 调用及逆归一化；DaoFHT 包括其返回分配与写回；GPU-NTT 包括位反转和逆 N 合同。setup、planning、编译、reference、warmup 在稳态计时之外。Graph 不计 Python 调用与图构建开销，因此“完整调用”应限定为所列 GPU 工作的稳态 Graph 调用。合法适配结果应保留。

|队列|算子/精度|基线|调用类别|对数|GM|
|---|---|---|---|---:|---:|
|cyclic90|ntt/word32|gpu-ntt|adapter|18|1.400189816|
|cyclic90|ntt/word64|gpu-ntt|adapter|36|1.264793524|
|fft_grouped2|fft/fp32|cufft|external-axis-composition|2|2.300077034|
|fft_grouped2|fft/fp32|vkfft|external-axis-composition|2|2.219478984|
|fp16_public_default9_20261105|fft/fp16|cufft|native-storage/vendor-arithmetic-unspecified|9|0.853583245|
|main66|fft/fp32|cufft|adapter|4|1.289540663|
|main66|fft/fp32|cufft|external-axis-composition|2|2.319050345|
|main66|fft/fp32|cufft|native|5|1.034351856|
|main66|fft/fp32|vkfft|external-axis-composition|2|2.225437277|
|main66|fft/fp32|vkfft|native|9|1.035764903|
|main66|fft/fp64|cufft|adapter|3|1.523998991|
|main66|fft/fp64|cufft|native|3|1.224772625|
|main66|fft/fp64|vkfft|native|6|1.353940859|
|main66|fwht/fp32|daofht|adapter|10|1.474332030|
|main66|ntt/word64|gpu-ntt|adapter|5|1.188155691|
|ntt5|ntt/word64|gpu-ntt|adapter|5|1.175440223|

## 独立 cyclic 与 FP16

cyclic：`snapshot.json#/cyclic/summary` 重算 54 对 GM=1.308404594、51 胜／3 负；word32 18 对、word64 36 对。其余 36 请求为 unsupported-semantic，无数值对且不填 0。原始和补测来源分别为 `results/full_current_20261025/recovery_followups_20261029/cyclic_mapping/measurement/results.json` 与 `recovery_gpuntt_baseline_20261102/results.json`；`paper/tools/export_current_refresh.py:213` 起校验补测未覆盖已有基线、逐点 descriptor 与原 own timing 一致。两时段来源分开保留，不把补测说成交错复测。`numbers.tex` 的 CurrentCyclicNTT=1.308 与其一致。

FP16：九点 GM=0.853583245、1/9 名义更快，全部 forward、out-of-place、自然顺序、shape/batch 原样；storage=4、compute=0，输出尺度 1，逆归一化关闭。逐点核对 `results/a100_fp16_acceptance_20261105/public/runs/<case>/default/result.json` 与 `external/<case>-cufft/result.json` 的文件哈希、实际 mapping 合同、shape/batch、GPU/library 身份、准备输入／reference 哈希、七次 Graph median 和比例均通过。`paper/tools/load_fp16_acceptance.py:127` 起还规定相同准备数据，DEFAULT、MEASURE、installed 分离；本轮未调用完整 loader 重新哈希所有大数组。cuFFT 明确低存储精度接口（`scripts/cufft_low_precision_baseline.py:53`），vendor 内部乘法／累加不可确认为逐指令相同。BF16 0 对是 pending，不是失败或零速比。

## 身份、计时与重复样本

主队列与 cyclic 的 GPU：NVIDIA A100 80GB PCIe，84974239744 bytes，UUID `GPU-2971e500-d3f6-d7e6-e5f2-cb67a06fff02`。主 library SHA256 `fac10905e8c7d88a1c42f669545df6182e988409a297ee99f55d00e6d5e57778`，research 模式；FP16 独立构建 `775a85c8f76e44bbfee886fd90489e7c309bd097b60e968183e446aa006177b2`。不可把两队列称为同构建新实测。完整主身份还在原 `measurement/results.json#/identity`，包括源 hash、broker、profile、model 和 registry；compact snapshot 的 identities 仅保留子集。

主66全部 own timing 是 `20 complete calls per CUDA graph; 7 replays; 10 warmup`；没有 in-place event 分支混入。`scripts/semantic_gpu_pipeline.py:221` 与 `run_semantic_campaign.py:504` 用 CUDA events 包住 Graph replay，再除 20；Graph 与 event 不是互斥概念。所有有 timing 的主 own／baseline 共 115 项 median 重算通过。65/66 own 行出现相同 replay 数值；这符合有限事件分辨率或稳定执行，但不能据此证明统计独立性或虚构精度。每点仍只有七个 replay 样本，不能把 20×7 称作 140 个独立测量。

## 有问题与可修复建议

1. 主表混合 native、adapter、external-axis-composition，类别信息被隐藏。保留所有合法完整调用对，在表注中披露类别与计入操作，或附本审计分组。FFT FP32 对 cuFFT 的 native 子集 GM=1.034351856（5 对），混合 GM=1.297913164（11 对）；整体均值不可表述为纯 native core 收益。确切混合值见 JSON，避免从四舍五入子组重算。
2. `paper/tools/export_current_refresh.py:100` 的 compact semantic_fields 仅保留 storage、shape、batch、inverse；compute、orders、normalization、placement、strides、word modulus 等需回溯 raw descriptor。建议导出完整 semantics 与 baseline adapter_operations／source identity 引用，并校验 baseline 的实际支持合同；当前依靠同 row 原始合同及 runner 实现，不能仅用 compact pairs 证明全部语义。
3. 表题的 66 和 cyclic 90 指请求覆盖；对应外部配对分母必须同时列明。没有匹配基线的算子和精度应明确为 gap。快照 coverage 的 operator_track_complete=True 限定为有限队列，不代表 full_refresh_complete。

## 错误口径与无法确认

浮点一律 error-report-only／accuracy-review-required，历史 atol/rtol 不是接受门槛，nominal faster 不是显著性结论。主原始 row 保留 numeric_diagnostics、correctness 分布、lossless error_artifact；FP16 保留 finite/nonfinite、L2、max/RMS 和分位数，两侧使用同 CPU reference，acceptance_threshold=null。整数 pass 是精确语义校验；本轮未重新计算所有数值 reference 或误差数组。

无法由本 CPU 审计确认：测量时独占 GPU、温度／频率稳定性、源代码与二进制完全对应、闭源库内部算术，以及跨独立 trial 的置信区间。已有 SHA256 是历史身份凭据，不自动证明当前 dirty HEAD 就是该构建。本报告不把这些限制变成新增 GPU 门禁。

伴随文件：`docs/performance_contract_audit_20261107_operators.json` 保存重算值、调用类别分组、源哈希与 NTT rowid 映射。

## 补充：FWHT supplied-output 公平性边界

主表十对 Dao 都是 out-of-place、自然顺序，实际 `adapter_operations` 包含 backend allocation、FWHT/fused scale、`copy_backend_result_to_supplied_output`。`scripts/semantic_dao_baseline.py` 的 `_Core.__call__` 在取得原生返回 tensor 后写回统一 runner 的预分配输出。`placement=0` 本身只规定输入输出分离，没有单独声明 caller-provided output ownership；项目 C API `include/cubutterfly/cubutterfly.h:301` 则显式接收输出指针。现有测量因此是 supplied-output API 合同下的完整适配调用。

若允许返回 backend 自有 tensor，Dao 返回值已可提供独立输出，这次 copy 并非 FWHT 数学或独立 ownership 必需。主表 1.474× 应限定为含写回的完整适配调用比较，不能解释为裸 Dao 核心或纯组织收益。FFT native 直接绑定输出，没有这种无条件返回 tensor 写回；cuFFT adapter 的 scale 满足请求幅值，但独立 multiply kernel 是实现选择。高 rank 轴组合 pack/unpack 满足当前组合实现的逻辑布局，不代表所有实现均需相同 copy。

后续已补：同十请求、相同CPU参考在775a构建上的三独立进程对照完成。指定输出合同的GM为1.236356–1.241750，每轮10/10名义领先；原生返回合同GM为0.935586–0.937841，每轮仅1/10领先。原生返回Graph为每侧保留20个独立结果，指定输出Graph重复写入一个调用方缓冲区，因而两组之间还包含输出工作集/缓存差异，不能把差额全部归因copy。历史1.474332仍仅属原构建/原合同，不覆盖这批新结果。完整逐请求、误差和身份见[后续证据](a100_evidence_followup_20261107.md)与[便携数据](../paper/generated/fwht_return_contract_20261107.json)。
