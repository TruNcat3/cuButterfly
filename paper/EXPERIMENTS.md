# 定稿前的有限实验清单

## 2026-10-12 按研究原则重组的主证据链

本轮以 [全蝶形研究原则](../docs/research_principles.md) 为准，正文实验按
下表组织。旧 E1--E14 是脚本与历史队列标识，不因章节重排改号或重跑。
历史矩阵的日期和范围在后文保留；它们不是当前全量任务的完成状态。

## 2026-10-17 已接纳的端到端补充

在 E13 的开源应用准备清单之外，已完成一个独立的 2D periodic spectral
Poisson endpoint，用来验证大 batch 的 xD lowering 和 Graph 重放。它固定
FP32、$N=2048$、$B=16$，把设备输入/输出拷贝、完整 RHS FFT/乘法/逆 FFT 和同步
计入时间；setup、编译、搜索和 capture 排除。A100 80GB 上 Graph 中位数为
cuButterfly 3.098 ms、Torch 3.590 ms、CuPy 3.674 ms，分别为 1.159$\times$ 和
1.186$\times$ 相对 Torch/CuPy。该 endpoint 使用 20 次单进程 steady-state 样本，
不是独立进程复现；它补充当前 application cohort，不替换原有 image-filter、FNO、
QuaRot 或 CKKS 记录。原始报告和 trace 见
[`docs/e2e_b16_trace_20261017.md`](../docs/e2e_b16_trace_20261017.md)，图表由
`render_principle_evaluation.py` 生成并记录在
`generated/principle_figure_metadata.json`。

| 论文问题 | 组织与展示 | 可复用证据 | 本轮补足重点 |
| --- | --- | --- | --- |
| RQ1 自动库是否高效 | 完整语义统一绝对延迟表；按算子/精度分组的分布图 | 当前 E1 36点；NTT 12点三进程独立批次 | 外部对照缺口及当前单进程覆盖，不能靠旧版填齐 |
| RQ2 机制如何选择 | 消融一：固定条件开关＋禁用组件后同预算重搜；联合延迟与硬件指标 | 已有布局/深度/流水控制、当前 FFT core×overlap | FWHT/NTT 同核心满 warp 的布局对照；受限搜索接口尚缺 |
| RQ3 收益是否来自运行模板 | 消融二：每种核心内比较传统/混合组织，再替换核心重复 | radix16 FFT 两核心×两组织；历史 FWHT/NTT 固定核心局部分段 | 其他蝶形的第二核心适配与完整交叉，不用 radix 标签代替核心身份 |
| RQ4 自动化和扩展是否成立 | 组合维度覆盖、独立池 regret、预算与编译/搜索成本 | 历史16候选池、生成见证及分层服务记录 | 记录实际模型消费路径，区分离线 replay 和在线搜索；不要求每个机制都获益 |
| RQ5 应用是否获益 | 相同 batch 的应用×后端绝对延迟/吞吐图表 | 图像过滤9点×3进程、已修复 QuaRot B1、独立 FNO 记录 | QuaRot batch 队列；CKKS完整应用仍缺，不以NTT单测代替 |

正文将当前算子筛选作为有边界的观测，保留独立的历史方法证据。RC2
大矩阵及旧 scaling 图仍在 artifact 中，不与当前36点混算。浮点未设新阈值，
历史通过条件按原 cohort 标注。外部库间均值分别计算，不用多个基线混合
提高总加速比。展示已有图、表的数据不足时保留缺测，不填预期性能数值。

复用审计见 [EXPERIMENT_REUSE_20261012.md](evidence/EXPERIMENT_REUSE_20261012.md)。
实际补测依赖和有限预算见
[principle_gap_campaign_20261012.md](experiments/principle_gap_campaign_20261012.md)
及 [机器任务清单](experiments/principle_gap_campaign_20261012.json)。其它缺测的
优先级、执行依赖和完成条件见
[后续任务清单](experiments/principle_gap_campaign_20261012_followup.md)。已完成的
历史实验不重复排队，准备完毕、正在测量和实现尚缺分别记录。

首批 FWHT/NTT 的四组布局配对已产生各三次进程计时，
[实际运行审计](evidence/PRINCIPLE_GAP_RUN_20261012.md) 保留初次编译 pending
以及仅补跑缺失配对的记录。NTT 通过完整 batch 精确比较；FWHT 当前只有内部
sanity 检查，完整误差分布接口和用户数值验收仍缺。该批只回答局部布局机制，
不替代消融二或禁用机制后重搜；四组中三个比值低于1，全部保留。

### 本轮图表与篇幅

- 一张当前算子绝对延迟大图，完整逐请求表放 artifact；NTT batch 扩展保持独立。
- 消融一给出组件及 hardware counters 的配对证据；未完成同预算重搜明确标记。
- 消融二用固定核心组织收益图突出纵向比较，避免只展示换核心后的最优值。
- 一张真实 batch 应用图，配一个其它应用绝对时间表；删除主文合成性能占位图。
- 主文保留回答研究问题所需的结论，具体源码、hash、CPU准备和队列状态放实验合同。

## 既有执行入口与历史清单

实验的统一配置与脚本入口见 [执行目录](experiments/README.md)，机器可读清单见
[plan.json](experiments/plan.json)。CPU 准备入口生成独立执行清单，不启动 GPU 任务。
已有原始结果继续复用；新结果
只有在硬件、构建、语义和计时协议匹配时才并入对应 cohort。正文 pending 标记
在相应实验完成并审核后删除。

历史实验已经包含同核心组织收益与核心替换收益，见
[HISTORICAL_EVIDENCE.md](HISTORICAL_EVIDENCE.md)。本清单中的“待补”指统一协议的
完整交叉验证、覆盖和生成/搜索评价，不表示此前没有效果。GPU 窗口限制影响当前
验收进度；历史证据不因此作废，也不拿来填充不同版本矩阵的缺项。

## 2026-09-21 实验证据的呈现顺序

算子和机制实验沿用 E1–E12。独立应用验证新增 E13，每类算子只选择一个开源应用，
其准备清单不改写现有 GPU 队列。借鉴用户指定的
[UnionSparse](https://arxiv.org/abs/2608.09291) 的分布、受控消融和机制指标组织方式，
按以下顺序接纳结果。

1. **总体效果。** E1 按算子、基线和精度画箱型图，同时保留每个配对点、样本数、
   中位数、四分位区间与几何均值。不同基线接口分组，in-tree 控制另列。
   正文历史分布图仍使用 71/72 的冻结快照；最新 RC2 算子结果集中在绝对吞吐箱型图
   和 `RC2_STATUS_20261002.md`，不与旧图数值合并。
2. **机制解释。** 先呈现 E2/E3 固定核心和固定组织的四格，再从已有 E4/E5
   配对中选择一个正收益、一个接近持平及一个负收益代表，核对下表指标。
   只复用或补齐这几个诊断点，不因解释需要重跑全矩阵。
3. **扩展范围。** E1/E11 依次按长度、batch、精度、方向分面。当前 FFT 图保留
   27 个 contiguous/out-of-place 点，另外 2 个 stride/placement 点不叠加进去。
   后续密码学位宽按实际模数、word/RNS/limb 表示分组，不与浮点精度混用。
4. **生成与选择成本。** E6/E9/E10/E12 报告排序、预算、覆盖和准备成本。
   correctness-only、profiler 诊断和匹配稳态性能使用不同证据类别。
5. **实际应用价值。** E13 直接测完整模型或密码学应用的执行，固定其余算子、
   权重/参数、输入与精度，仅替换蝶形后端。报告基线中的蝶形占比、调用侧适配开销、
   应用总时间与质量，不能把单算子加速比填成应用结果。

| 方法预测 | 配对控制 | 解释指标与判断边界 |
| --- | --- | --- |
| 驻留减少中间搬运 | 同核心、同语义，stagewise 与 resident | HBM/L2 读写字节、全局事务、kernel 数及无 profiler 延迟。按完整变换汇总，不能仅用带宽百分比解释收益 |
| 后继对齐改善交换 | 同组织和核心，仅改变 boundary layout | producer/consumer 事务、shared bank conflicts、指令量和寄存器。必须把两端代价一起计算 |
| 独立数据提供延迟隐藏机会 | 同 N/core/layout，改变 batch 或 tile batch | 实际 overlap、启动间隔、fill/drain、launch 数、事件等待及独立计时。存在 overlap 本身不等于加速 |
| 核心与组织相互制约 | strict core×organization 四格 | 核心指令/吞吐、registers/spills、occupancy、stall classes。occupancy 更高不能单独证明效率更高 |

当前正文已引用匹配 FFT 的 Nsight Systems launch/overlap 诊断。
上述资源 counters 尚无稳定匹配数据的部分继续标为 pending，不能从不同 workload、
不同 build 或不同 profiler replay 的记录拼成正收益解释。复用历史 trace 时保留其
正常计时与 profiler 计时的独立协议。对应绘图入口为
`tools/render_experiment_figures.py`，冻结数据为 `evidence/current_snapshot.json`。

## 实验问题与完成标准

| 优先级 / ID | 研究问题 | 设计 | 完成标准 |
| --- | --- | --- | --- |
| P0 / E1 | 当前库的综合表现如何？ | 继续各自 UUID 下的 72 单元验收，不重跑该记录中兼容的完成点 | 每点有正确性、选择映射、基线语义/不可用原因、重复 trial 和覆盖状态；全体完成才报告全矩阵几何均值 |
| P0 / E2 | 普通核心能否从组织改进中获益？ | 普通/优化核心 × 基础/混合组织；先 FFT FP32 N=2^8,2^12,2^18，B=1 和 max(1,2^22/N)，均 forward/contiguous/out-of-place | 同核心配对比较；核心替换配对比较；无法四格共用的形状显式缺失；保留负结果，不制造慢核心 |
| P0 / E3 | 上述收益能否跨算术复现？ | 在共同合法配置上加入 FWHT FP32 与 NTT word64；先 N=2^12，B=1,1024；NTT q=576460756061519873/natural output | 验证同精度、同模数、同归一化/顺序；外部库若未适配核心，只做外部基线，不宣称库本身被改进 |
| P1 / E4 | 哪个组织机制生效？ | E2/E3 中的共同合法映射，逐项改变 writer layout、驻留深度、data tiles、prefetch、overlap；不强求每 backend 都支持每轴 | 单因素记录实际资源、launch 数及延迟；需要 counters 的点在独立 profiling cohort 执行 |
| P1 / E5 | batch 是否提供预期 slack？ | 固定 N/core/layout，改变 B 与 batch tile；分别看小负载、接近饱和、饱和后 | 报告 fill/drain、实际并发/事件开销和稳态趋势；profile 时间不作为无 profiler 校准值 |
| P1 / E6 | 未见配置的排序如何？ | 延续独立 stage service，固定训练/验证集合；按 whole workload 留出 | top1/top2、pairwise、regret、median/P90 误差和 gaps 同时报告；不得用验证点拟合最终被验证的整图时间 |
| P1 / E7 | 不同硬件能否自动选到合适映射？ | V100 SM70、A10040/80 同语义的 E2/E3 子集，统一版本/协议；各自独立硬件校准 | 同时报告映射差异、性能、校准/编译/搜索时间、复用量；V100 async-copy 不可用按能力处理 |
| P2 / E8 | CT/GS、正逆和排序能否复用方法？ | 列出实际 backend 支持表；CPU 参考加公共 API 正逆及输出置换检查 | 结构推导与工程支持状态分开；不支持的变体留空/待实现，不能通过重命名冒充 |
| P1 / E9 | 模型和搜索是否减少寻找好点的成本？ | 在 E2/E3 的少量代表性负载上固定候选池与 core；比较服务模型、简单流量排序、随机/局部搜索；总预算包含 seed，可复用已测池先做离线重放 | 报告总预算 4/8/16 的 best-so-far、regret、实际编译/校准/测量成本；与既有 16 exploration + 最多 4 seed 的验收分别标注；离线重放不冒充在线耗时，小候选池 oracle 不冒充全空间最优 |
| P1 / E10 | 自动生成与扩展到底复用了什么？ | 审核现有语义适配、core 与共享构造路径，列合法/可生成/已测覆盖；在固定 leaf 阈值两侧检查逻辑构造，另选择一个明确留出的形状或真正新增的语义适配实例 | 区分解析/CPU 表达覆盖与 GPU 支持；记录共享代码、手写调度修改量、正确性与准备成本；留出形状称参数泛化，不称新算子扩展；不另启无界算子开发 |
| P1 / E11 | 精度、正逆和 CT/GS 形式能否复用同一组织？ | 对 FFT/FWHT/structured-2x2 扫描 FP16/BF16 native 与 FP32 accumulation、FP32/FP64；对 zeta/NTT 扫描整数正逆；显式记录 CT/GS 形式。当前 CT 走已有 lowering，GS 先作为 reference-only | 每个 GPU 点必须有完整批次正确性、方向/归一化和精度契约；GS 未完成 lowering 时只报 unsupported/reference，不用 CT 别名填充；外部库仅在匹配契约存在时比较 |
| P1 / E12 | 校准、编译和搜索的成本如何与库的 setup 成本比较？ | 将 cuButterfly 的编译、校准、候选搜索、首次调用和稳态 kernel 分开；对 cuFFT/VkFFT/Dao/GPU-NTT 记录 plan/setup/首次调用和稳态。固定版本、预算、正确性和 GPU cohort | 报告各阶段 wall time、可复用缓存、break-even 调用数；校准是工程准备成本，不能包装成方法新颖性；搜索成本另成系统评价 |
| P1 / E13 | 算子收益能否转化为完整应用收益？ | FFT/FWHT/NTT 各一个开源应用，版本、输入、模型/密码参数及原始后端固定；[独立准备方案](END_TO_END.md) | 全应用实际计时、配对重复、质量/解密检查、适配开销与 trace 占比；未集成和不支持项显式记录，不从 kernel 数字外推 |

## 交叉消融的固定规则

1. 普通核心使用已有正常优化的 scalar/radix2 实现；优化核心选择当前可执行的
   register/codelet 或导入组件。基础组织使用同算术核心的逐阶段/传统分组路径。
2. 首先在四格共同合法的子集固定映射进行比较；若某项接口不能保持相同映射，
   标明“不具备该配对”，不拿不同布局的结果充当纯核心替换。
3. 另报等预算调优后的系统收益：既有验收为 16 个 exploration attempts，加最多
   4 个 historical-seed attempts，最多 3 个 finalist；100 warmups、100 repeats、
   3 个确认 trial。四格比较固定总 screen 数并计入 seed，实际预算与省略数逐点报告。
4. 编译先做，GPU timing 前做 correctness；数值契约不因核心不同而放宽。
5. 以每点配对比值呈现普通核心的组织收益、优化核心的组织收益、两种组织下
   的核心收益。不将不同平台/协议的 trial 合并取中位数。

实验实施阶段先做共同合法性清单与预编译；这一步发现的缺少 lowering 是工程
任务，不靠论文文字补齐。准备脚本、CPU 检查和模块编译与 GPU 验收分别计数，
不因执行清单已经生成而将这些实验标为完成。

E9/E10 补齐 arch/sys 生成系统常见的评价问题，不另行复制 E1 全矩阵。
分层的可扩展表达可以通过构造与容量界论证，不要求先测尚不存在的硬件。
若报告当代 GPU 上的分层性能收益，则在 E4 的共同合法候选中限制/放开层间选择，
保持核心与预算可比。不得把“表达更多点”直接写成“所有点更快”。
2026-09-17 的组织工作新增统一执行计划、target 配置、受控配对入口和 CPU 汇总。
E1 复用独立检查点续跑；其他 GPU 实验按前置条件和设备独占状态执行。
严格四格的 A100 80GB/40GB 执行记录已归档；剩余缺项是更广的核心/算子覆盖、
E6 留出清单、E9 独立候选池的在线成本，以及 E10 代码复用审计。

本轮扩展将 E11/E12 保持为独立 cohort。E11 的机器可读准备合同是
`experiments/extended_semantics.json`，CPU 入口
`tools/prepare_extended_semantics.py` 会生成 46 个当前 CT GPU 命令和 46 个
reference-only GS 记录；它不把 GS 的设计空间枚举误写成已实现 lowering。E12
只比较准备和首次使用成本，不把校准时间混入 kernel latency，也不把较长准备时间
当作方法性能收益。

## 当前非迁移执行状态（2026-09-18）

本轮已完成并审核以下 A100 证据，均不涉及 V100 或硬件迁移：

* E1 使用 A100 80GB 的 71/72 完整匹配单元；一个 bit-reversed word64 NTT
  单元保留为明确不可用，不做插值或均值填充。
* E2 的固定算术叶 FFT 核心×组织消融在 A100 80GB 和同架构 A100 40GB 各完成 6 个
  workload、72 行，native/cufftdx-thread × stagewise/resident 四格全部正确；
  2026-10-12 按原始 CSV 复算纠正了旧 40GB 摘要的核心比值方向：native/cuFFTDx
  的几何均值为 1.038/1.043（stagewise/resident），不能将倒数写成总体退化。
  “同核心”固定的是算术叶，组织切换时 block 形状和 wrapper 可随之改变。
* E3 的跨算术组织控制在 A100 80GB 完成 24/24 单元：FWHT FP32 与自然序
  word64 NTT，$N=2^{12}$、batch 1/1024，固定 shared-iterative 核心比较
  leaf0/leaf2。结果见
  `results/paper_completion_20260918/e3_cross_operator/RESULTS.md`。
* E4/E5 的布局、深度、batch 和 profiler 证据，以及 E6/E9/E10 的模型、搜索
  和生成证据沿用已有归档；它们仍与迁移实验分开报告。

* E11 的 A100 80GB CT extension cohort 完成 46/46 个低精度、FP32/FP64、
  正逆和整数 zeta/NTT correctness points；包含 36 个浮点点、6 个 zeta 点和
  4 个 NTT 点。该 cohort 只有一个固定映射的 timing journal，作为语义覆盖检查，
  不替代三 trial 外部基线。GS 已进入 architecture-space manifest，但仍是
  reference-only；没有独立 lowering 就不计为 GPU 支持。

* E12 尚未完成。它将把校准作为工程准备成本，单独比较 compile/plan、calibration、
  search、first-use 和 steady-state kernel，并报告 break-even calls；这些时间不
  混入本文的 kernel latency 或方法新颖性论点。

E2/E3 的结果支持“组织收益可在固定核心上复现，并可跨算术语义复用”的限定
论点；不扩展为所有核心、所有算子或所有平台的最优性结论。E7（V100/A100
迁移）在本轮范围之外，仍按独立硬件 cohort 管理。
