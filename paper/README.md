# cuButterfly 论文工作稿

**cuButterfly: Unleashing Performance of Butterfly Computation with Hierarchical Hybrid Dataflow**

本目录是基于 APPT26 内部原稿重新组织的英文工作稿，采用用户提供的
ISCA2026 IEEE 模板。APPT 原稿尚未发表且未在审；未修改原稿，也未将其写成
已有正式发表成果。模板年份只表示所用格式，本工程不执行投稿。

## 2026-10-19 独立证据补充

实验正文已补入10月17日66请求同精度复核、FFT/FWHT/NTT预算8受限重搜与
当前组织控制，以及10月19日CKKS完整操作/短链的36配置、144执行路径。
CKKS保留原生循环与双方合批两种对照，报告24个B=8/32配置的收益和全部
B=1强制合批负收益；应用循环/合批自动选择、安装模型及完整bootstrap仍未完成。
这些独立cohort不与原36点筛选＋12点NTT扩展混算，既有图表及hash清单未变。
来源、局限及剩余事项见[当前证据补充](evidence/CURRENT_EVIDENCE_20261019.md)。

## 2026-10-17 端到端证据补充

新增一个独立的 2D periodic spectral Poisson endpoint：A100 80GB、FP32、
$N=2048$、$B=16$，计时包含设备输入/输出拷贝、完整 RHS FFT/乘法/逆 FFT 和同步，
排除 setup、编译、搜索与 Graph capture。Graph 中位数为 cuButterfly 3.098 ms、
Torch 3.590 ms、CuPy 3.674 ms。该数据来自 20 次单进程 steady-state 调用，图中作为
重复样本分布展示，不当作独立 process confidence interval。

Poisson 图和生成表由 `paper/tools/render_principle_evaluation.py` 直接读取
`results/e2e_current_20261017/poisson_2048_b16_adaptive_trace/report.json`，并写入
`paper/figures/principle_application_poisson_endpoint.*` 和
`paper/generated/principle_application_poisson_endpoint*`。它是独立 application cohort，
不覆盖原有 9 点 image-filter cohort，也不把新的 endpoint 反推为所有 2D/3D 形状的结论。
它对应 benchmark scope 中的 `end_to_end_fft_poisson_adaptive_revision` 补充 track，
其 manifest、report hash 和署名边界见 [研究报告与证据合同](../docs/reporting_contract.md)。

## 2026-10-12 按全蝶形研究原则重组实验

实验章节按自动库横评、机制与受限重搜、固定核心组织迁移、选择与准备成本、
完整应用五个问题组织。新增当前算子绝对延迟箱图、同核心组织收益图和真实 batch
应用吞吐/比值图；正文不再使用合成性能占位图。该轮沿用系统论文写作与科学证据
核查流程，保留负结果、误差政策和实际证据边界。

当前横评36点、独立NTT扩展12点和历史消融分开统计。
[证据复用审计](evidence/EXPERIMENT_REUSE_20261012.md) 核对原始来源，并纠正
历史 A100 40GB 核心替换比值的倒数解释。历史大矩阵仍保留，当前较小矩阵的
较高均值不被解释成同条件版本提升。浮点继续报告误差，不引入新验收阈值。

[实验组织](EXPERIMENTS.md) 给出每项论点的已知证据和剩余缺口；
[首批补测合同](experiments/principle_gap_campaign_20261012.md) 与
[后续任务](experiments/principle_gap_campaign_20261012_followup.md) 区分可运行
实验和需要先补接口的实验。队列运行与结果接纳分开，不能由进程退出判定实验完成。

图表可通过 `python paper/tools/render_principle_evaluation.py` 重绘，统计与原始文件
hash 在 `generated/principle_figure_metadata.json`。主文和实证稿均由
`make -C paper all evidence` 生成；当前页数和检查见 [验证记录](VALIDATION.md)。
下方各日期条目属于历史版本，其中旧图号、范围和待测状态不代表本轮状态。

## 2026-10-11 draw.io 图稿

正文的八张方法图已重构为原生 draw.io 图元，统一颜色、字号、箭头和分面。
打开 [八页编辑文件](figures/drawio/cubutterfly-methods.drawio) 即可编辑；也可单独修改
各图。论文直接使用 draw.io 导出的矢量 PDF，图内节点、连线和文字均可独立调整。
[编辑与导出说明](figures/drawio/README.md) 提供单图更新和整包回写入口。
实验统计图仍由冻结数据生成，分析图中的示例数值不作为性能测量。

## 2026-10-11 叙事与机制图修订

本轮把实验前的内容改成一条连续论述。Background 先定义蝶形依赖和算术变体，
随后接入相关工作，再由已有方法的组织对象引出局部复用与并行供给的矛盾，最后
进入方法章节。Related Work 不再等到实验之后才出现，读者在看到方法前即可知道
本文与变换生成、张量编译、层级数据流和 GPU 专用库的区别。

新增两张分析图：`organization_tradeoffs` 用同一 16 点 FWHT 对比逐阶段、两段和
整图驻留，给出可复核的闭合分量、传输和独立工作数；`semantic_transport` 用三位
反转、二维轴标签和 CT/逆 GS 局部形式说明可迁移的结构关系及其条件。流水图重绘为
A/B/C 的就绪屏障与缓冲区生命周期，Roofline 改为符号化边界和达到边界所需的条件。
这些图都明确标注为分析构造，不替代 GPU 性能证据。

当前稿件 `main.pdf` 为 13 页，其中正文到结论约 12 页，参考文献另计。增加的篇幅
来自定义、过渡和可复核样例，没有改变模板字号或页边距。`main-evidence.pdf` 同样
保留真实测量与分析图，排除带水印的合成性能布局。构建后应检查最终 PDF，而不是
只根据源文件长度判断版面。

## 2026-10-09 修订（历史记录）

该轮把创新收敛为两项：**由共同蝶形依赖导出的统一执行构造**，以及
**组合局部驻留与流式交接的分层混合数据流**。Related Work 独立讨论变换生成、
张量编译和层级数据流，并用八行方法表说明具体区别。SPIRAL/FFTX 等已有工作
已经研究统一变换描述，因此不使用未经证明的“首次统一所有变换”或“首次混合数据流”。

- 当时的[排版稿](build/main.pdf) 为 11 页，含三张明确带水印的合成数据图稿。
- 当时的[实证稿](build/main-evidence.pdf) 为 10 页，由 `make -C paper evidence` 生成，
  排除这些合成图。它仍是内部草稿，不表示实验或投稿验收已经完成。
- [图稿数据与替换规则](figures/DRAFT_LAYOUTS.md) 区分机制、规模/搜索与端到端布局。
  示例数值不参与性能结论，实测 RC2 箱型图和固定核心消融保留。
- [应用证据核查](evidence/APPLICATION_RECORDS_20261009.md) 补入已有历史 FNO
  和随机权重 QuaRot 测量。FNO 约慢 5%，QuaRot 的有限延迟测试近似持平，
  HEonGPU 完整 bootstrap 仍未测。不同构建不混算。
- [本轮改稿与检查](REVISION_20261009.md) 记录压缩、来源和未完成证据。

新版 `0.9.3.4` GPU 语义实验继续使用原冻结输入。本次改稿未修改内核、校准或队列。
以下较早叙事记录按各自日期理解，最新章节和图表状态以上述修订为准。

## 阅读与构建

- [主文件](main.tex) 引入各章；[PDF](build/main.pdf) 是构建输出，当前页数和检查
  记录见 [验证记录](VALIDATION.md)；[顶层阅读副本](main.pdf) 在构建后同步，避免打开
  不同版本。
- [构建说明](BUILD.md) 记录用户级 Tectonic 工具链；不依赖 CUDA，不修改公共 conda 环境。
- [实验来源](evidence/README.md) 和 [冻结清单](evidence/manifest.json) 对应正文结果表。
- [论点与证据](CLAIMS.md)、[后续实验](EXPERIMENTS.md) 区分完成内容和待补内容。
- [实验执行目录](experiments/README.md) 提供 E1–E12 的统一清单、硬件 target、
  CPU 准备/预编译、受控测量和证据汇总入口。
- [端到端方案](END_TO_END.md) 与 [E13 清单](experiments/end_to_end.json) 对 FFT、
  FWHT、NTT 各选择一个开源应用，固定替换边界、基线和质量合同；
  [执行记录](../results/paper_profile_rc2/end_to_end/README.md) 包含历史 FNO 与随机权重 QuaRot 测量。
  正文采用原始 JSON 复算统计，相关 README 的 QuaRot 短负载中位数错误见本轮核查。
- [参考文献核验](REFERENCES.md) 记录来源与访问边界。
- [相关工作对位](RELATED_WORK.md) 区分已有方法、本文研究问题及仍需的比较证据。
- [验证记录](VALIDATION.md) 记录 PDF、引用、数据来源及 CPU 数学示例检查。
- [下载的论文范本](literature/README.md)、[改稿方案](REVISION_PLAN.md) 与
  [方法构造底稿](METHOD_BLUEPRINT.md) 连接相关工作、修改依据和贯穿样例。
- [历史配对证据](HISTORICAL_EVIDENCE.md) 单独保留核心替换及同核心组织收益。
- [贡献收敛](CONTRIBUTION.md) 将中心观点收敛到依赖驱动的通用处理范式；
  [GPU 映射实例](GPU_MAPPING_CASE.md) 和 [服务组合实例](COST_MODEL_CASE.md) 对接真实记录。

在仓库根目录运行：

```bash
python paper/tools/check_theory.py --output paper/generated/theory_checks.json
python paper/tools/check_construction.py --output paper/generated/construction_checks.json
python paper/tools/export_evidence.py
python paper/tools/render_method_figures.py
python paper/tools/render_experiment_figures.py
make -C paper
```

两点 FWHT 组合算例的纯 CPU 重放命令见
[服务组合实例](COST_MODEL_CASE.md)；冻结输入包括实际资源需求系数与调度下界。

图表重绘需要 NumPy/Matplotlib，可直接使用现有 `cubutterfly` conda 环境。
`make` 只编译已保存的图表，不需要 Python 绘图依赖。实验表生成器读取冻结记录，
不随后台实验变化。显式接纳新记录使用实验绘图脚本的 `--refresh-snapshot`。
更新论文结果时，应审核新记录并
显式更新快照及正文统计；不能只替换图表而保留旧结论。

## 全文主线

2026-09-21 修订将正文收敛到 **dependency-closed subgraphs**、
**hierarchical hybrid dataflow** 和 **successor-aligned writeback**。
方法以 16 点 Hadamard 的行/列子图贯穿，保留少量必要关系，把位掩码、局部逆矩阵、
具体 GPU 参数和模型计算细节留在支撑文档。关键问题和发现适量加粗，正文尽量用
直接的因果句替代冒号、分号和长串并列说明。

实验借鉴用户指定的 [UnionSparse](https://arxiv.org/abs/2608.09291) 的证据组织。
当前主图先用同一 RC2 采样矩阵上的绝对 transforms/s 和 effective butterfly-pairs/s 箱型分布对比高性能库，
并保留逐 workload 的配对图作为细节，再看匹配比值、固定核心/组织消融与 trace，随后检查长度、batch、
精度和方向的覆盖，最后讨论生成、搜索和迁移。新绘图仍只读取既有冻结快照。
正在运行的新版综合实验不会被部分合入本轮正文或取代旧构建的缺测项。

全文的证据链以两张总图连接。Introduction 的 `overview` 从共同依赖、stage/data
子图组织推进到硬件层级。生成章节的 `system_architecture` 将相同输入接入候选、
lowering、验证和选择，并区分准备与运行。算子分布回答效率，固定核心实验和 trace
回答原因，E13 最后回答实际应用价值。FNO 已产生阶段性结果，其余应用仍待测；
2026-10-09 正文已合入单独标注的历史端到端表，不能将算子加速当作应用加速。

Introduction 面向体系结构/系统读者：先说明蝶形计算在频域处理、精确多项式
算术及结构化变换中的重要性，再提出跨算子、规模和 GPU 配置的实现问题。
核心挑战是依赖、复用和并行供给的耦合；核心发现是这些问题可由共同的
stage-bit 子图及其交集关系构造，再映射到 stage/data 的分层时空组织。
贡献集中于依赖驱动的处理范式、分层执行方法、生成选择和证据；解耦的是语义、
负载域与硬件实现的描述，不是它们的性能关系。

Background 借主图的共同依赖面板和算子列表说明“蝶形计算是什么、有哪些”，随后介绍已有 GPU 库和
局部计算核心。Motivation 延续 APPT 的 Roofline 起点：从计算量、物化流量与强度
推导复用机会，再说明资源限制、历史观察和 stage/data 的时空选择。分层还负责
表达随规模扩展的逻辑划分，不仅是当代硬件上的性能选项。方法章完成组织与依赖
构造，随后才描述子图执行的 GPU 落地。

三个术语层次固定如下：

| 层次 | 正文术语 | 含义 |
| --- | --- | --- |
| 组织思想与方法 | hierarchical hybrid dataflow | 根据依赖与资源，在可扩展逻辑层次选择 stage/data 的时空组织 |
| 执行实现 | block-level subgraph execution / resident subgraph execution | 将所选组织落实为核心调用、驻留、交换和同步 |
| 系统 | cuButterfly | 适配核心、lowering、校准、搜索、生成与重放 |

图块不是 CUDA CTA 的别名。数据驻留与任务/worker 驻留也不是一回事。
正文中的 logical group、physical group、kernel launch 分开使用。
逻辑层深、物化段数与物理存储层数分别记录。算子语义模板、负载参数、硬件能力是
输入，映射是联合选择的输出。性能模型服务于排序，不承担方法正确性或最优性证明。

## 相对 APPT 的继承和扩展

| APPT 原稿 | 新论文处理 |
| --- | --- |
| FHE/HHE 的多尺寸 NTT 动机 | 扩展为多种蝶形计算的重要性与跨平台组织问题；不保留大段专用密码背景 |
| stage/data 两维时空展开 | 保留思想，并给出寄存器、warp、CTA、跨 kernel 的层级映射 |
| Npart/Spart/p 和特定 NTTU | 转为图块、实际核心、资源及执行边界；不固定分段数 |
| URAM bank/offset | 继承写回服务后续读取的原则；按 GPU 事务重新约束和证明 |
| 独立 FPGA pipeline 的 II=1 | 改为带共享资源、依赖、队列和填充排空的条件性吞吐模型 |
| 整体中间量可放 URAM | 对每个存储层计算峰值活跃集；必要的 global 交换显式计入 |
| 特定配置的 2N/Npart 迭代 | 不直接推广为任意阶段划分的通式，使用实际图块/执行组计数 |
| 单 FPGA 及文献性能比较 | 使用 V100/A100 原始记录，规划匹配核心、组织和协议的消融 |

## 结构对应的口径

用户确认的“GA”指 **GS / CT 等蝶形变体**。正文给出位反转的邻接对应、
CT/GS 局部矩阵逆关系，以及全图逆运算需要反转因子顺序的说明。
拓扑可对应不代表算术标签、内存连续性、浮点舍入或运行时间相同。
zeta 使用全蝶形的依赖子图；任意 structured 2×2 的逆要求矩阵可逆。
搜索器另称 evolutionary neighborhood search，不将现有实现包装为完整遗传算法。

## 图与实验的当前状态

总览图合并原蝶形依赖图和混合组织图的作用，独立架构图展示自动实现及运行边界。
保留定性 Roofline、N=16 子图/写回、bank/offset 和流水时序图。
APPT 原图和先前草图保留供溯源。图源、继承关系和重绘命令见 [图稿说明](figures/README.md)。
定性图和分析样例明确标注，不冒充测量结果；实验图使用独立数据生成器。

V100、A100 40GB 和 A100 80GB 的不同实验 cohort 分开保存。V100 旧版本性能
可用于呈现已有方法证据，但不等于当前版本已在 V100 重新完成部署验收。
A100 80GB 旧版图表快照为 71/72 complete：29 FFT、13 NTT、12 FWHT、
5 subset-zeta、5 superset-zeta、6 structured-2x2 和 1 个 legacy alias。
剩余 NTT 位反转输出项属于原构建缺口，独立修复不混入这一快照。
旧版图表结果见 [生成汇总](generated/current_results_summary.json) 和正文图表；
三次试验先取中位数，再计算匹配性能比的几何均值。原有
`generated/results_table.tex` 保留历史 cohort，不作为当前主矩阵汇总。

历史记录已观察到同核心的组织收益、核心替换收益及 batch 交叉点，已写入动机。
A100 80GB 上的严格 FFT 核心/组织四格和 FWHT/word64 NTT 跨算术控制已完成，
但仍是有限 cohort，不能外推为所有核心、算子或平台的最优性。广泛未见配置的
模型泛化、同版本 V100 和迁移成本仍按各自进度标注。只有实际改接并测量某外部库，
才能声称该库也得到提升；待补项不预设正结果。

冻结 RC2 的独立汇总见 [RC2 状态](RC2_STATUS_20261002.md) 和
[机器可读快照](evidence/rc2_snapshot_20261002.json)。正文把它作为已接纳的算子 cohort，
同时保留旧版图表和严格消融作为历史/机制证据，避免不同源码、设备和协议混合。
完整研究目标、端到端性能、同版本 V100 验收及投稿定稿尚未完成。

## 参考文献维护

正文使用 `\cite{key}`，文献集中放在 `references.bib`，由
`\bibliographystyle{IEEEtran}` 和 `\bibliography{references}` 生成参考文献。
不要手写编号或维护另一份 `thebibliography`。论文条目和对应软件条目分开；
出版年份依据正式发表记录，预印本和软件访问日期不能冒充会议年份。
新增引用应同时更新核验说明和相关工作对位，明确哪些比较来自全文、哪些仅有
摘要或元数据支持。增加引用不会改变冻结实验的覆盖范围。
