# 典型子图执行的 profile 补证方案

状态：2026-11-08 限定的12个变体全部完成，见[完整结果与归因](../evidence/REPRESENTATIVE_PROFILES_20261108.md)。
以下保留预先确定的范围与控制；实际入口、CPU预编译、三进程计时、NCU与NSYS均已完成。
这是针对机制解释的新有限补充，不重开已完成的 [A100 冻结清单](../evidence/A100_CLOSEOUT_20261107.md)。
目标是连接“组织干预 → 实际资源/通信变化 → 完整调用收益或退化”，不以计数器
好看、所有机制为正收益或胜过所有外部库作为结束条件。

## 固定范围

新增采集只含两个既有请求、共12个实现变体。原计划 A100 80GB
`GPU-2971e500-d3f6-d7e6-e5f2-cb67a06fff02` 忙碌，实际使用同型号空闲卡
`GPU-0eae5f07-33de-321b-70a9-ffeae49309e6` 建立独立新cohort，全部12变体在此卡重测。
已有40GB诊断及另一张80GB的历史计时保持其独立身份。
选择依据是已有可审计核心、明确干预和中大型工作量，未按新profile结果挑点。

| 编号与研究问题 | 请求、设计点及控制 | 本次工作与证据边界 |
| --- | --- | --- |
| P-S：子图执行具体改变什么？ | FFT FP32，N=65536，B=256，正向、不归一化、out-of-place；native register / cuFFTDx Thread 两种16点叶，各比较逐radix16步外部交换与256值resident prefix。前缀覆盖两个radix16步骤，后续步骤及请求不变 | 新采4格。追踪前缀、后续阶段及共同排列的完整范围。线程、所有权、wrapper、存储与launch共同变化，归因为完整子图组织，不能把全部收益独立归给驻留或少launch。原数据两核心分别约1.186/1.173×，只是选点先验，新计时独立报告 |
| P-P：连续tile执行与预取如何影响供数？ | FFT FP64，N=65536，B=16；两个256点factor，每组4个并行局部变换、64线程、EPT16、writer-aligned、recurrence；两个核心各测(t,d)=(1,0),(4,0),(4,1),(4,2) | 新采8格。t4d0→t4d1隔离预取配置，t1d0→t4d0隔离连续tile数；t1d0保留为更强同步端点，d2保留为退化点。各对校核其余实际mapping一致。编译后的寄存器/存储变化是干预后果，必须报告 |
| R-L：局部复用的收益与代价能否跨算术观察？ | 复用A10040，FFT/FWHT FP64与word64 NTT，N=2^20，B=16；两外部阶段带10+10固定，local0/2/4九格 | 直接复用已有普通计时和NCU/NSYS。NTT包含local2获益/local4退化；FWHT保留不同排序。两外部交换不变，不能解释成减少外存pass。原profile为单次冷采集，不能据其时长重排普通计时 |
| R-B：边界访问组织有什么实际效果？ | 复用独立A10040 FP16 N4096/B256，两阶段原/改进所有权与写回 | 原生packed算术指令数保持，store sectors/request 32→4、barrier stall降低，但部分shared冲突上升。它是所有权与写回的联合修复，不是纯swizzle开关；单段与cuFFT仍更快，不以修复坏路径代表相对强库的收益 |

P-S 来源：[实验配置](core_dataflow_ablation.json)、[配对汇总](../generated/principle_core_organization.csv)、
[执行证据](../evidence/SUBGRAPH_EXECUTION_20261108.md)。
P-P 来源：[固定核心组织报告](../../docs/factor_core_organization_experiment_20261025.md)、
[配置](../../configs/experiments/factor_graph_core_organization_20261025.json)。
R-L 来源：[匹配profile审计](../evidence/NARRATIVE_SCALING_PROFILE_20261027.md)。
R-B 来源：[11月4日低精度证据](../evidence/FFT_L_UPDATE_20261104.md)。

## 每个问题需要什么指标

| 机制问题 | 主要观测 | 不可替代的限制 |
| --- | --- | --- |
| 子图内部保留是否改变外层交换？ | 单次完整调用的kernel/拷贝数；分阶段DRAM读写字节、L2读写sector、global请求/sector、shared访问量；独立普通延迟 | 区分逻辑搬运、L2和DRAM，缓存可能使变化不同。捕获不得包含初始化、验证、warmup或重复回放累积。仅减少launch不能分解驻留贡献 |
| 边界交换是否更有效？ | producer和consumer分别的请求/sector、shared事务与bank conflicts、地址/搬运指令量、等待及普通延迟 | 同时计算写端与读端代价；sector/request下降不等于字节同比下降。未采到的项为缺失，不填0 |
| 预取是否改善tile供数？ | 异步复制实际指令、等待指令位置与相关stall、寄存器/shared/spill、issue/eligible warps、普通延迟 | NSYS能显示kernel/stream事件，不能直接看到单kernel内部tile流水。需要局部时间线时另行记录最小插桩诊断，不把其时间当普通性能 |
| 更深局部化为何有代价？ | 编译与NCU资源、active/eligible warps、barrier/scoreboard、普通计时的重复分布 | occupancy或stall百分比不是耗时分解，不能单指标证明瓶颈；不同kernel的百分比不直接平均 |

数值采集时查询当前NCU支持的metric，保存原始名称、单位和分母；不静默用另一
指标冒充缺失指标。完整调用只有绝对计数可在定义一致、节点范围明确时求和；
百分比、命中率与带宽应保留逐kernel值，或具备原始分子/分母后再加权。
原始NCU replay和NSYS时长均与无profiler计时分开。

## 预定执行约束（现已完成）

1. CPU侧先绑定请求、核心源码/二进制、输入及实际mapping，预编译全部12个变体。
   若旧二进制或模块不能匹配，则新建补证cohort，在相同新构建下重测全部对照；
   不把新profile附会到旧性能值。已知旧计时留作选点依据。
2. P-S已新增“准备和预热在外、捕获一次完整调用”的standalone入口，保留所有必要
   排列与边界kernel。旧`profile_control.py`按某类kernel过滤，未用来代替完整范围。
   P-P已核实Graph capture/实例化在范围外，范围内恰好一次完整replay；NSYS显式
   使用node粒度，原8份缺少节点的默认graph粒度报告保留为被替代记录。
   不以launch-skip猜测范围，不捕获整个benchmark进程后累加计数。
3. GPU侧在独占空闲窗口完成数值检查和普通计时。新补证采用3个独立进程，
   变体顺序轮换，统一warmup=100、repeat=100；全部过程保留，启动期不稳定则
   将本次标为需检查，不在结果出来后只丢慢轮。编译、搜索与profile均不计入。
4. 另行对12个变体各采一次单完整调用NCU及NSYS，记录重放策略、缓存/时钟设置。
   使用与该profile同构建、同设备和同mapping的普通计时，不使用profiler时间作加速比。
5. 浮点继续报告同精度参考及完整误差，不新增通过阈值；NTT复用数据保留原精确合同。
   输入、输出、映射或范围不一致时停止该配对归因，记录缺口。
6. 12个变体完成或有明确阻断记录后结束本补证。负收益或指标与假设不符也构成
   有效结果，不自动追加尺寸、调参、换核心或扩展到完整性能矩阵。

## 论文呈现

优先把现有分析图重构为机制对应的紧凑面板，而不是新增一串独立图：

- 子图执行面板：简化执行结构 + 普通完整延迟 + 外层交换绝对量。
- tile供数面板：连续tile数/预取深度 + 普通延迟 + 资源和等待；真实时序未采到时
  只放明确标注的示意，不把期望流水画成实测。
- 跨算术资源面板：FFT/FWHT/NTT的局部分组及资源代价，保留负例。
- 边界修复作为短例或补充材料，说明producer/consumer事务与等待的联合变化。

每个面板使用物理单位，异单位指标各自独立轴，避免把寄存器、百分比、字节等
混成一个无量纲颜色差值。控制实验负责方法归因；外部库profile负责描述实际
差距，两者分别标识。具体面板取舍待新增捕获完成，不预填有利结果，也不把
该计划当成已完成的A100证据。
