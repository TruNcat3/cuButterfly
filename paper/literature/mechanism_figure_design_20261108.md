# 机制统计图的设计依据与统计口径

署名：TruNcat3 · 项目：cuButterfly · 2026-11-08

本轮仅重绘冻结结果，不追加GPU实验，也不修改原始数据。主图保持一个图号，
用两行三列将同核心性能、通信/供数观测和资源代价对应起来。

## 参考对象

已检查官方论文全文及图所在PDF页面，以下只作为表达方式参考，不作为本项目实验证据。

| 来源 | 查看位置 | 借鉴的组织方式 | 本图中的落实 |
| --- | --- | --- | --- |
| [Welder，OSDI 2023](https://www.usenix.org/system/files/osdi23-shi.pdf) | Fig.13，印刷页712，PDF第13页 | 用同组受控变体并排展示延迟、kernel数、访存事务和中间结果容量，将机制变化与执行结果对应 | 同核心组织对照与完整调用通信量同图，明确5→4 kernel，并列寄存器代价 |
| [ROLLER，OSDI 2022](https://www.usenix.org/system/files/osdi22-zhu.pdf) | Fig.17–18，印刷页243，PDF第12页 | 显式列出不同优化配置；不同资源服务指标使用分开的坐标轴 | 保留t1d0强端点、t4d0控制、d1收益和d2退化；shared与活跃warp用上下对齐的独立轴 |

原文PDF已在项目literature清单中归档，本轮校核官方USENIX条目与原PDF。
没有复制原文图形、数据或文字，也没有把两篇论文的实验结果迁移到本项目。

## 主图面板与论述

- **(a) 同核心收益**。FP32驻留prefix相对同核心逐步骤外部交换。柱高为
  `median(T_external) / median(T_resident)`，虚线1表示无变化。共同排列及后续步骤计入。
- **(b) 通信变化**。分别展示驻留后DRAM与L2完整调用读写量占各自同核心基线的比例，
  虚线100%为基线。DRAM和L2不是两份可相加的通信，二者有不同分母。
  原始GiB和比例均保存在CSV中。
- **(c) 局部资源代价**。只对比被替换的前两个步骤和驻留prefix的寄存器数；
  共同排列和不变后缀不属于这个局部资源对照，但始终包含在(a)(b)完整调用范围内。
- **(d) 强端点对比**。FP64所有四种组织都以各自核心t1d0作参照，d1相对较弱t4d0
  的受控预取收益单独标注。避免只展示1.925/1.869×而掩盖强端点下1.152/1.117×。
- **(e) 供数等待**。两个factor kernel分别画点；横向偏移只用于区分类别。
- **(f) 资源约束**。上轴为每计算块分配shared，两个核心和factor的值相同才合并柱；
  下轴是各factor实际活跃warp百分比。两轴都从0起，不共用不同单位的y轴。

关键负例是d2的long-scoreboard继续下降，完整执行却慢于d1。只能说容量、
并发和等待共同变化，与资源约束权衡相符；不能据此量化各项独立贡献。
不能把这些指标堆成100%的“归因份额”，也不能把两种机制的收益简单相乘。

## 统计与可追溯性

- 12个变体各3个独立进程，保留全部36条普通计时。内部100次replay不当作100个独立样本。
- 加速比柱使用“中位数之比”；点及须线使用3个进程内配对比值及其最小–最大范围。
  不把它们写成“配对比值的中位数”，不提供小样本显著性或置信区间。
- NCU每配置只有一次完整capture。按kernel保留百分比，无profile重复样本的误差棒。
  图中同配置的不同factor标记不是统计误差范围。普通计时与profile耗时不混用。
- 数据分母、换算、完整来源行、输入哈希和CSV哈希由绘图脚本写入metadata。
  当前NCU采用未锁缓存/时钟的kernel replay，观测不保证逐次普通运行的精确流量。
- 图导出PDF、PNG和可编辑SVG。论文双栏宽7.05英寸设计，入文缩放0.98；
  刻度和主体标签约7pt，辅助注释约6.0–6.5pt。不以增加DPI代替最终尺寸检查。

文件：

- [统计主图 PDF](../figures/representative_profiles_20261108.pdf)
- [图数据 CSV](../generated/representative_profiles_chart_data_20261108.csv)
- [完整来源与绘图口径](../generated/representative_profiles_figure_metadata.json)
- [原始结果及误差报告](../evidence/REPRESENTATIVE_PROFILES_20261108.md)

## 绘图工具与方法来源

绘图使用Matplotlib与项目脚本，PDF采用Poppler进行入文目视检查。
本轮采用Scientific Agent Skills的scientific-visualization流程检查分母、
物理单位、缺失值、重复样本和最终尺寸，另以systems-paper-writing检查图文对应。
工具流程来源记录为：Timothy Kassis, Vinayak Agarwal, Yuhuan He, Darshil Patel,
and Aubrey M. Brueckner. *Scientific Agent Skills: A Library of Procedural Knowledge
for Research Agents*. 2026. [arXiv:2609.00065](https://arxiv.org/abs/2609.00065),
[DOI](https://doi.org/10.48550/arXiv.2609.00065)。2026-11-08核验最新记录为v2。
该引用属于绘图工具来源，不作为蝶形算法相关工作或性能论据。
