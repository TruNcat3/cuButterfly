# A100 综合实验与论文更新核查

日期：2026-11-02。结果来自 cuButterfly 持久化实验记录；本文核查与改稿不重新
执行测量。加速比统一为对照完整调用延迟除以 cuButterfly 延迟，组均值为
每请求一个权重的几何平均。不同研究集合、竞品和调用边界不合并成一个总均值。

## 完成范围

| 集合 | 已完成 | 数值与性能口径 |
| --- | --- | --- |
| 主算子横评 | 66/66 请求，12 个算子/精度组 | 29 个整数精确通过；37 个浮点完成有限性与误差分布报告，未新增验收阈值 |
| 独立 cyclic NTT | 90/90 请求 | 全部精确通过；54 个 GPU-NTT 配对，36 个对照模数范围限制 |
| 有限域组合 | 138/138 合同，2 输入 × 2 变体，共 552 次测量 | 全部整数精确；内部 public-default 对比复用 cyclic mapping，GM 1.691×，不属外部横评 |
| 独立 CPU 对照补齐 | 34 请求，每项 2 个独立 CPU 进程，另 2 次诊断 | 24 整数精确、10 浮点误差报告；共享主机且与 GPU 计时边界不同，不并入 GPU 均值 |
| 固定核心阶段组织 | 6 对，每对 3 次交错 trial | 普通事件计时；对应 12 个实现变体均有 NCU/NSYS 捕获 |
| 当前应用 | CKKS 36、bootstrap 2、Poisson 3、固定滤波 2 请求 | 46 个外部应用配对（含 2 个单进程误差报告滤波结果），另有 36 个匹配批处理集成对照 |

主横评的 264 个基线格包括 49 个有计时的匹配配对、11 个不支持的请求和 204 个
算子域不适用格。49 个配对中 5 个是精确 NTT，44 个是浮点误差报告结果。
缺口不按失败算子或零性能计入统计。主横评与 cyclic 集合有两个不支持对照的
重复请求，因此不能把 66+90 写成 156 个唯一请求；有效外部配对无重复。

来源：[主横评](../../results/full_current_20261025/measurement/results.json)、
[cyclic 恢复](../../results/full_current_20261025/recovery_followups_20261029/cyclic_mapping/measurement/results.json)、
[基线补测](../../results/full_current_20261025/recovery_gpuntt_baseline_20261102/results.json)、
[组合独立审核](../../results/a100_closure_20261102/composition_summary.json)、
[组合测量](../../results/full_current_20261025/composed_138/searched_graph/results.json)、
[组合清单](../../results/full_current_20261025/composed_138/searched_graph/manifest.json)、
[cyclic mapping 来源](../../results/full_current_20261025/composed_138/cyclic_mapping/mapping_source.json)。

组合的 552 次测量来自 138 请求 × random/boundary 两输入 × public-default/
searched-cyclic-cores 两变体。每请求先对两种输入的延迟比取几何平均，再等权汇总，
GM 为 1.690705×，请求范围 1.050505–2.995727×，138/138 观测比值大于一。
复用 mapping 来自已完成 cyclic 研究的 public MEASURE（预算 512），逐模数通道
回放；不是 joint RNS 搜索、跨模数融合、外部库优势或完整 FHE 应用加速。
核心选择和组织均可能改变，不能作固定核心因果归因。每个请求/输入/变体只有一个
进程 trial，默认变体先测；七次 Graph replay 和两种输入不提供独立进程置信区间。
原始仅 public-default 的 `composed_138/graph` 仍单列历史，不与本轮延迟混合。

## 算子对比

| 集合与算子 | 对照 | 配对数 | 几何平均加速比 | 测得领先 |
| --- | --- | ---: | ---: | ---: |
| 主横评 FFT FP32 | cuFFT | 11 | 1.298× | 9/11 |
| 主横评 FFT FP32 | VkFFT | 11 | 1.190× | 5/11 |
| 主横评 FFT FP64 | cuFFT | 6 | 1.366× | 5/6 |
| 主横评 FFT FP64 | VkFFT | 6 | 1.354× | 6/6 |
| 主横评 FWHT FP32 | DaoFHT | 10 | 1.474× | 9/10 |
| 主横评 NTT word64 | GPU-NTT | 5 | 1.188× | 4/5 |
| cyclic 扩展 NTT word32 | GPU-NTT | 18 | 1.400× | 18/18 |
| cyclic 扩展 NTT word64 | GPU-NTT | 36 | 1.265× | 33/36 |

“测得领先”只表示观测比值大于一，不是独立进程统计显著性检验。FFT FP32 相对
VkFFT 虽有平均收益，11 点仅 5 点领先，应保留其偏斜分布与负结果。主横评包含
原 36 请求和预先声明的 30 个多维/批量扩展请求；例如原 FP32 FFT 六点相对 cuFFT
为 1.111×，新增五点为 1.564×。合并后的 1.298× 表示该 66 请求研究的匹配集合，
不是同尺寸下从 1.111× 提升到 1.298× 的版本对比。

cyclic 54 个配对合计为 1.308×，范围 0.972–1.990×，51 点领先。32 位有效配对
覆盖 N=256/65536，64 位覆盖 N=65536/1048576，均有 B=1/7/16 和正逆方向。
18 对来自原始同时测量，36 对复用原 cuButterfly 延迟、单独补测同 GPU 同协议的
GPU-NTT。源文件、行、输入、mapping 和协议均经校验，近 1× 的差异不作显著性解释。
36 个未配对请求超出固定 GPU-NTT 版本的模数支持范围；cuButterfly 本身仍精确通过。
三个略慢请求均为 word64、B=1，最差延迟增加约 2.9%。

主横评没有低精度 FFT 配对。FWHT FP64、zeta、structured 和部分 xD NTT 有本库
执行证据，但当前 GPU 研究缺少相应 GPU 基线比值，不能写成已超越 GPU 对照。
独立 CPU 对照已补齐 34 请求：每项两个进程，共 68 次测量，另对两个不稳定点
各做一次诊断；24 个整数请求精确、10 个浮点请求报告完整误差，不新增阈值。
单线程固定 CPU 31 的共享主机不是独占环境；CPU 完整调用含调度、复制和转换，
GPU 列复用驻留设备 Graph 延迟且不含传输，两种边界不混入 GPU 横评统计。
两个小 zeta 请求的 CPU 更快，负结果与全部重复值均保留。来源：
[CPU 覆盖与限制](../../results/a100_closure_20261102/cpu_external/README.md)、
[CPU 汇总](../../results/a100_closure_20261102/cpu_external/summary.json)、
[34 请求配对](../../results/a100_closure_20261102/cpu_external/paired_34.csv)、
[身份审核](../../results/a100_closure_20261102/cpu_external/identity_audit.json)。
完整语义、原始延迟和实际 mapping 保存在
[论文数据快照](../generated/current_refresh/snapshot.json)，派生表为
[operator_summary.csv](../generated/current_refresh/operator_summary.csv)。

## 应用完整调用

| 场景 | 类型 | 对照 | 几何平均加速比 |
| --- | --- | --- | ---: |
| Periodic Poisson，3 请求，1024²/2048²，B=4/16 | FP32 2D FFT | Torch/cuFFT Graph | 1.097× |
| 同上 | FP32 2D FFT | CuPy Graph | 1.106× |
| 固定 31² 滤波器，2 请求，B=16 | FP32 2D FFT | CuPy FFT convolution Graph | 1.057×，单进程误差报告 |
| CKKS key switching，9 请求 | NTT | HEonGPU 原生循环 | 1.395× |
| CKKS rotation，9 请求 | NTT | HEonGPU 原生循环 | 1.377× |
| CKKS square/relinearize/rescale，9 请求 | NTT | HEonGPU 原生循环 | 1.448× |
| CKKS square/rescale/key switching，9 请求 | NTT | HEonGPU 原生循环 | 1.494× |
| CKKS 上述 36 请求 | NTT | HEonGPU 原生循环 | 1.428× |
| CKKS 同批处理组织，36 请求 | NTT | 原生 NTT 核心匹配集成，受控对照 | 1.218× |
| CKKS 每请求更快的原生方案，36 请求 | NTT | 循环与匹配集成取更快者，受控对照 | 1.131× |
| 小参数 regular bootstrap，N=4096，B=1/8 | NTT | HEonGPU 原生 Graph | 1.031× |

CKKS 四类操作各交叉 N=16384 的 method I/II 与 N=65536 的 method I，再交叉
B=1/8/32。每个请求的双方密文逐元素一致。相对更快原生方案，三个 B=1 请求
落后；B=8 和 B=32 两组均约 1.182×。双方匹配集成不是上游库已有的批量 API，
也不是固定算术核心的组织消融。bootstrap 和短链 CKKS 使用 `sec_level=none`，
不能外推生产安全参数下的加速。

Poisson 与滤波保留同精度 CPU 误差、不同/恢复输入及输出生命周期验证，未新增
浮点验收阈值。滤波原始 `accepted=null`、`formal_performance_eligible=false`
保留在快照 `source_numerical_reports`，不把未设容差误作性能无资格；图中空心点
表示单进程证据，不代表数值阈值验收或独立进程统计结论。双方缓存滤波器，padding 为
1080²/2100²；与历史重复计算滤波器的九点集合分开。

补查纠正初次汇总遗漏：历史随机权重 FNO 在 64²/128²、B=4 时，Graph 为
1.029932×/1.056419×，eager 为 0.955415×/0.957358×。每尺寸两个独立进程、
每进程三组不同输入；计时包含输入复制、完整执行和独立输出复制。不能将它与
更早的训练模型质量或 eager 记录混成同一 cohort。

随机权重 QuaRot 的 B=1、128 prefill/8 decode eager 诊断为 0.982×，输出有限且
一致；只有两次计时且不含输入刷新/输出复制。10月18日原扩展的 Graph 探测确有
非有限输出，但10月19日修复 current-stream 后已有有效 Graph 记录：B=1/4 为
0.966889×/0.956544×，双方输出有限且逐元素一致，换输入、恢复输入和输出生命周期
检查均通过，计时包含输入刷新和输出复制。这是单进程历史负结果，不能再列为
完全缺失的 Graph 实验，也不冒充当前构建或独立进程重复。随机权重不证明任务质量。

来源：[应用派生记录](../generated/current_refresh/snapshot.json) 中每个 pair 的
`source` 指向原 CSV/JSON；[CKKS 原始目录](../../results/full_current_20261025/ckks)、
[Poisson](../../results/full_current_20261025/poisson)、
[滤波](../../results/full_current_20261025/fft_filters)、
[QuaRot eager](../../results/e2e_current_20261018/quarot_b1_p128_d8_eager_fp32.json)、
[QuaRot Graph baseline-first](../../results/e2e_current_20261018/quarot_debug_graph_p16_d1_baseline_first.json)。
后续有效记录：[FNO Graph/eager 汇总](../../results/e2e_unified_20261013/fno_hotpath_summary.json)、
[QuaRot Graph B1](../../results/e2e_alignment_20261019/quarot/full7b_b1_p128_d8_graph.json)、
[QuaRot Graph B4](../../results/e2e_alignment_20261019/quarot/full7b_b4_p128_d8_graph.json)。

## 机制归因与资源分析

已有两个 FFT 核心的独立实验显示，固定各自核心、改变组织分别获得约 1.217×
和 1.221×。这支持收益不完全来自换核心，但不证明所有负载和算术核心必然获益。

当前六点阶段组织实验固定 scalar radix-2，仅将每阶段一个分组改为 [8,4] 或
[8,8,2]。三次配对比值的中位数如下。

| 算子 | N=4096，B=4 | N=262144，B=1 |
| --- | ---: | ---: |
| FFT FP32 | 11.546× | 21.407× |
| FWHT FP32 | 13.674× | 24.096× |
| NTT word64 | 11.057× | 19.835× |

这些是相对刻意逐阶段组织的内部对照，不是相对 cuFFT/DaoFHT/GPU-NTT，也不是
同预算搜索出的强对照。NTT 预检精确；浮点预检只有该 benchmark 的最大误差，
不替代主横评的完整误差分布。对应 profile 有 12 个变体、无捕获失败。指定变换
kernel 数由 12→2 或 18→3，但捕获是 complete-process、kernel replay，不能把
其总 DRAM 字节或 profiler 时间当作完整调用流量和普通性能。shared bank-conflict
并未列入这批计数器。历史更强组织、过深驻留和流水负结果均仍在正文中。

来源：[普通计时](../../results/full_current_20261025/mechanism/timing.json)、
[profile 清单](../../results/full_current_20261025/mechanism/profile/manifest.json)、
[原固定核心与交叉控制说明](CORE_CROSSING_RESULTS_20261027.md)。

## 论文可写与尚不能写的结论

已更新摘要、引言、实验设置、匹配算子表、有限域扩展说明和 profile 论述。应用图、
CKKS 控制表与固定核心组织图从同一份冻结快照重绘。现有证据支持“在所测 A100
请求上，统一组织方法兼容不同蝶形算术，外部横评具有整体竞争力，组织收益能够
在固定核心对照中被观察，并能在若干完整应用中体现”。

仍不能宣称完整设计空间最优、所有算子/精度/负载逐点领先、全部端到端应用获益，
或已经证明跨架构迁移。当前完整研究清单仍缺 PRO 6000 实测、部分外部对照、
当前构建的 FWHT Graph 应用确认；更广强对照仅支持更强主张时需要。
searched-composition 的有限内部比较已完成，但不替代缺失的外部 GPU 比值或固定
核心归因。论文继续按四节组织实验，搜索保持工程角色。
