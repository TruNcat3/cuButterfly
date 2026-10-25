# 相关工作对位与论证边界

本轮核验日期：2026-10-09（复用 2026-09-17 来源审计）。正文统一使用 `\cite{key}`，条目集中保存在
[`references.bib`](references.bib)，由 IEEEtran BibTeX 样式生成编号。
本表说明论文应如何使用来源；文献中的性能数字不并入 cuButterfly 的实验结果。

本轮正文按变换生成/专用执行、tensor 编译/tile 调度、层级数据流三类重写，
并在 [generated/related_work_table.tex](generated/related_work_table.tex) 加入
8 行、4 列的方法比较。各行写已报道的优化对象与组织方法，最后一列写本文具体研究
的构造，不使用支持/不支持勾叉，也不将软件与同名论文算作两种新算法。
VkFFT 表中仅写已核验的 runtime generation，详细内部调度的核验限制留在来源记录。
逐项来源及可访问性见
[本轮记录](literature/review_20261009/related_work_sources.md)。

创新性集中于从 labeled butterfly dependencies 得到 dependency-closed subgraphs，
由相邻 index sets 的交集推导通信及就绪，再用 successor-aligned writeback 衔接子图。
方法思想为 hierarchical hybrid dataflow，GPU 落地为 block-level / resident subgraph
execution。语义、负载与硬件的分离指输入及构造分离，核心、布局、驻留和物理映射的
性能仍需联合优化。不主张 tiling、hybrid、核心替换或 evolutionary search 首创，
不以有限候选搜索主张全局最优。APPT 是内部未发表草稿，本轮已移除正文中将其作为
正式先行工作的定位。

用户强调的两点贡献据此写为正向且可验证的口径：统一跨蝶形算术的依赖导出执行范式，
并定义局部驻留与子图间流式交接组成的层级混合数据流，包含通信、就绪与后继布局规则。
已有 SPIRAL/FFTX 的统一变换表示/生成和 Welder 的层级 tile 数据流意味着，当前来源
不足以支持“无人统一相似计算”或“无人定义混合数据流”的宽泛排他断言。

可执行的外部比较对象统一登记在
[`../config/external_comparison_catalog.json`](../config/external_comparison_catalog.json)，
并由 [`../scripts/validate_comparison_catalog.py`](../scripts/validate_comparison_catalog.py)
检查。该目录把直接基线、可嵌入的局部计算核心、端到端上下文和方法参考分开，
从而避免把不同抽象层的工作混入同一个吞吐率平均值。

## 与本文最相关的研究问题

1. **组织与核心的联合优化已有先例。** 不能把已有研究概括为“只优化算术核心”。
   本文要说明如何将这些选择纳入多个蝶形算子共用、可实际 lowering 的空间。
2. **统一表示本身不足以证明创新。** 必须交代表示中的 stage/data、核心归属、
   边界布局和生命周期如何改变可生成的候选、搜索质量及迁移代价。
3. **阶段性性能已有具体支持。** 历史中已有保持局部核心的组织配对和核心替换
   观察，见 [HISTORICAL_EVIDENCE.md](HISTORICAL_EVIDENCE.md)。统一交叉消融、
   未见配置排序和匹配版本跨架构实验进一步确定交互与泛化范围。

## 文献对位

表中“范围”指核验文献实际讨论的对象，不表示该工作不可能扩展到其他对象。

| BibTeX key | 已有贡献及本文用途 | cuButterfly 需要展示的区别或证据 |
| --- | --- | --- |
| `fftw2005`, `spiral2005` | 变换规划、codelet、结构化表示和实现搜索的基础工作 | 不能声称首创变换生成或多算子表示；需展示 GPU 子图边界、核心接口与实测服务组合的具体作用 |
| `fftx2018` | 组合 FFT 计划、用户计算内联，以及 SPIRAL 后端的源代码生成与数据搬运优化 | 是组合生成的直接近邻；需展示蝶形边界布局、生命周期、核心契约及独立阶段服务在统一搜索中的作用 |
| `vkfft2023`, `vkfft` | 可移植 GPU FFT 库与运行时生成；论文与实现分别引用 | 完整 FFT 基线；不能从公开接口猜测其内部调度，也不能把其覆盖范围等同于本文当前的幂二长度实验 |
| `gpuntt2023`, `gpuntt2025`, `gpuntt` | Merge/Four-Step 的阶段划分、共享内存复用及参数优化；另有 2025 年发表研究 | 必须承认 NTT 专用库已经做组织优化；区分论文算术、当前分支与实测 adapter，不能据此宣称已证明跨算子生成收益 |
| `hadacore2024` | Tensor Core 处理局部 Hadamard 块，较大变换通过共享内存交换 | 说明高效核心与组织共同重要；论文中的低精度结果不能直接代替当前 FP32 FWHT 基线 |
| `flashfftconv2024` | 面向长卷积的 FFT 矩阵分解、Tensor Core 和融合；§3.1–3.2 联合分析分解、容量与搬运 | 需解释同类组织选择如何适用于不同蝶形算术与可替换核心；卷积端到端/半精度结果不能直接与当前单个 FP32 FFT 配对 |
| `quipsharp2024` | 随机 Hadamard 变换用于低比特 LLM 量化；为应用动机提供实例 | 应用重要性的来源，不是已有 cuButterfly 端到端量化提速证据 |
| `tvm2018`, `triton2019` | 通用 tensor 编译及 tiled computation；TVM 包含计算/调度分离和图级、算子级优化 | 不声称通用编译器不能表达蝶形；本文应具体说明 stage 依赖及核心/边界契约给生成过程带来的约束和收益 |
| `tensorir2023` | 核心 block 签名与外层变换分离，tensorization 匹配硬件 intrinsic | 是核心可替换性的重要先例；本文要说明 stage-bit 闭合子图、边界布局及物理服务怎样形成蝶形候选，而不能只强调接口解耦 |
| `eyeriss2016` | 数据流分类、原语到物理映射，以及统一资源条件下的层次搬运/能耗比较 | 借鉴资源公平与机制分析；不将 ASIC 能耗结论等同于 GPU 吞吐 |
| `ansor2020` | 分层 sketch、evolutionary search 与学习型 cost model | 搜索策略本身不是新颖性依据；评估独立物理服务组合的样本复用、未见配置 regret 和调优成本 |
| `roller2022` | 硬件对齐的 tile 构造和 micro-performance model，考虑事务和 bank 等约束 | 层次、布局和快速建模均有先例；本文应给出蝶形跨阶段驻留和实测服务的适用范围 |
| `welder2023` | tile-graph 表达不同存储层的复用，graph connecting/subgraph tiling 使用流量和容量估计；生成时处理布局及 liveness | 是组织抽象的直接近邻；不能声称首创 tile dataflow 或中间量复用，需展示精确蝶形依赖、可替换核心和校准服务的共同作用 |
| `timeloop2019` | §IV–VI 将工作负载、体系结构和映射分开，以层次分块和时空映射评估数据流 | 层次数据流思想有先例；本文需处理蝶形阶段依赖、实际 GPU lowering 及编译/实测资源，不能仅将术语改名 |

## 引用与比较规则

- 论文引用描述方法，软件引用标识实现，两者可以同时出现。正式发表的条目使用
  真实会议/期刊；只有预印本来源时使用 `@misc`、arXiv 标识和 URL。
- 依据原文的正面能力进行比较，不由“摘要没写”推断“不支持”。对专有库不猜测
  内部 kernel 组织，基线依据其公开 API 与实际计时协议。
- FlashFFTConv 的预印本时间为 2023，官方作者仓库给出的会议为 ICLR 2024；
  本稿采用会议年份。应用论文与算子库不是同一层级的性能基线。
- 现有 BibTeX 条目及相关段落不意味着所有外部软件已安装、适配或测试。
  新增外部基线前，先核对算子、类型、方向、归一化、布局和分配/计时边界。

## 本轮新增来源的核验入口

- FlashFFTConv：[作者论文全文](https://arxiv.org/html/2311.05908v1)，
  [官方代码与会议 BibTeX](https://github.com/HazyResearch/flash-fft-conv)。
- QuIP#：[PMLR 论文页面及 BibTeX](https://proceedings.mlr.press/v235/tseng24a.html)。
- Timeloop：[NVIDIA 研究页面](https://research.nvidia.com/publication/2019-03_timeloop-systematic-approach-dnn-accelerator-evaluation)，
  [作者全文](https://parashar.org/ispass19.pdf)，
  [出版 DOI](https://doi.org/10.1109/ISPASS.2019.00042)。
- FFTX：[作者全文](https://www.spiral.net/doc/papers/hipc_2018.pdf)。
- VkFFT：[出版 DOI](https://doi.org/10.1109/ACCESS.2023.3242240)、
  [官方实现](https://github.com/DTolm/VkFFT)。本轮未取得 IEEE 全文，比较仅使用已核验范围。
- GPU-NTT：[2023 ePrint](https://eprint.iacr.org/2023/1410)、
  [2025 发表记录](https://doi.org/10.1109/ACCESS.2025.3570024)。
- HadaCore：[作者论文](https://arxiv.org/abs/2412.08832)。
- Triton：[出版 DOI](https://doi.org/10.1145/3315508.3329973)。本轮只核验元数据和宽泛定位。
- TVM、Ansor、ROLLER、Welder：分别使用 USENIX 的
  [OSDI 2018](https://www.usenix.org/conference/osdi18/presentation/chen)、
  [OSDI 2020](https://www.usenix.org/conference/osdi20/presentation/zheng)、
  [OSDI 2022](https://www.usenix.org/conference/osdi22/presentation/zhu)、
  [OSDI 2023](https://www.usenix.org/conference/osdi23/presentation/shi) 论文页及全文。

## 现有实验如何回答最近邻比较

- E2/E3 的核心 × 组织交叉消融检验组织收益及其跨算术复现，不把核心加速归给组织。
- E4/E5 的布局、驻留和 batch 实验解释何时应减少物化、何时应分段，包含资源饱和反例。
- E6/E7 以未见组合排序和匹配版本跨 GPU 测试评价服务复用与迁移成本。
- 文献对位不自动启动新 GPU 矩阵。FFTX/通用编译器的可比 workload 应先确认生成
  能力和语义；FlashFFTConv/HadaCore 若进入计时对比，先建立与其应用/精度一致的合同。
  对位表中的能力描述不是已完成的外部库适配或测量。

完整核验范围见 [REFERENCES.md](REFERENCES.md)；待测内容继续由
[EXPERIMENTS.md](EXPERIMENTS.md) 管理，不因增加引用而标为完成。

下载的全文、范本阅读位置和修改方法见 [literature/README.md](literature/README.md)
与 [REVISION_PLAN.md](REVISION_PLAN.md)。
