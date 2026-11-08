# 典型子图与预取的机制 profile

署名：TruNcat3 · 项目：cuButterfly · 日期：2026-11-08

状态：限定的两个请求、12个变体全部完成。6个普通计时进程给出36条结果，
12份NCU与12份有效NSYS覆盖34个kernel记录；没有扩充尺寸或搜索矩阵。
本补证不替代已冻结A100横评，也不表示PRO6000迁移已完成。

## 结论与实际设计点

固定算术叶时，子图组织改善了完整执行，而收益受资源约束。P-S直接补上
“闭合子图 → 内部交换 → 外部流量变化 → 完整调用收益”的证据链。
P-P说明连续tile遍历必须配合供数方式，不能把更多时间复用或更深预取本身当成收益。
两项都是内部机制控制，不是新增cuFFT横评，也不是自动搜索最优性的证明。

- **P-S**：正向FP32，N=65536，B=256，不归一化、独立输出。逐步组织为
  4个radix16计算阶段，每线程一个16点叶、每组256线程；驻留组织为
  3个计算阶段，首阶段每组16个256值子图、每子图16线程，内部执行两个radix16步，
  后续两个步骤不变。每计算阶段4096组，另有共同的输入排列kernel。
  两个核心分别是现有原生register叶和cuFFTDx Thread 16点叶。
- **P-P**：正向FP64，N=65536，B=16，不归一化、独立输出。两个256点factor，
  每组4个局部变换、64线程、每线程16值，writer-aligned、recurrence固定。
  连续tile数t=1时每阶段1024组，t=4时256组；预取深度d=0/1/2。
  两个核心是现有register-tile和cuFFTDx Block。全部变体保留两个kernel和一次Graph调用。

P-S改变了所有权、wrapper、存储和launch，固定叶不能把全部收益进一步独立归给
“少一次launch”或“纯驻留”。P-P在t固定时比较d，编译资源变化是该干预的后果。
不是把完整cuFFT调用当作局部核心，也没有改变输入精度或请求。

## 普通完整调用时间

单位为微秒。每个值来自三个独立进程，每进程warmup=100、repeat=100，变体顺序轮换。
中位数与最小–最大范围如下；不是置信区间，三次没有剔除慢轮。

| 请求 | 核心 | 组织 | 中位数 µs | 三进程范围 µs |
| --- | --- | --- | ---: | ---: |
| P-S FP32 | cufftdx-thread | resident-prefix-2 | 1202.575 | 1202.504–1202.616 |
| P-S FP32 | cufftdx-thread | stagewise-hbm | 1430.948 | 1430.180–1431.020 |
| P-S FP32 | native | resident-prefix-2 | 1194.435 | 1194.342–1194.629 |
| P-S FP32 | native | stagewise-hbm | 1413.827 | 1413.755–1413.898 |
| P-P FP64 | cufftdx | t1-d0 | 80.404 | 80.292–80.404 |
| P-P FP64 | cufftdx | t4-d0 | 134.554 | 134.513–134.574 |
| P-P FP64 | cufftdx | t4-d1 | 71.987 | 71.956–72.090 |
| P-P FP64 | cufftdx | t4-d2 | 93.143 | 93.082–93.225 |
| P-P FP64 | native | t1-d0 | 75.325 | 75.315–75.377 |
| P-P FP64 | native | t4-d0 | 125.901 | 125.809–125.921 |
| P-P FP64 | native | t4-d1 | 65.403 | 65.321–65.403 |
| P-P FP64 | native | t4-d2 | 80.118 | 80.067–80.138 |

P-S原生/cuFFTDx Thread组织收益为 **1.184/1.190×**。
P-P固定t4，d0→d1为 **1.925/1.869×**；相对更强t1d0端点仅为 **1.152/1.117×**。
d2相对d1慢22.5%/29.4%，且均慢于t1d0。三进程最大范围宽度约为中位数的0.185%。
这些新值不与此前另一张A10080上的1.034/1.093×拼接或声称为版本提升。

## Profile归因

| 观测 | 原生叶/核心 | cuFFTDx叶/核心 | 可以说明什么 |
| --- | --- | --- | --- |
| P-S完整DRAM读写量 GiB | 1.259→0.970 | 1.263→0.971 | 分别减少22.9%/23.1%，包含共同排列及后续步骤 |
| P-S完整L2读写sector量换算 GiB | 3.021→2.626 | 3.024→2.626 | 分别减少13.1%/13.2%，与DRAM和逻辑边界比例不同 |
| P-S寄存器/线程 | 分步48–51，驻留80 | 分步46–54，驻留83 | 减少交换同时增大局部资源需求 |
| P-P t4d0→t4d1长记分牌等待，两个factor分别列出 | 71.02/72.38%→13.79/20.63% | 63.27/67.68%→19.63/18.53% | 与异步供数改变一致，不是执行时间分解 |
| P-P t4d0→t4d1 issue active，两个factor分别列出 | 14.17/10.75%→29.49/24.34% | 11.46/8.09%→23.50/18.83% | 活跃warp比例低也能改善发射，不能只看occupancy |
| P-P t4d1→t4d2分配shared KiB/组 | 49→65 | 49→65 | 更深缓冲有容量代价；含分配粒度/驱动开销，实际请求48→64 KiB |
| P-P t4d1→t4d2活跃warp比例，两个factor分别列出 | 6.68/6.89%→5.76/5.69% | 7.08/7.10%→5.75/5.60% | 资源压力增加与退化同时出现，不能单指标量化因果份额 |

全部34个NCU kernel的local-memory读写sector计数为0，未观察到该次采集的本地内存流量。
SASS审计确认d0没有异步复制，d1/d2实际发出LDGSTS、commit及wait指令。
静态指令位置数不是动态执行次数，NSYS也不能直接看到单kernel内部tile重叠。
两个factor保持相同外部边界；P-P改善不能表述为减少外部pass。

NCU使用kernel replay、cache-control=none、clock-control=none。计数器没有跨重放
相加；只将一次逻辑调用内定义相同的绝对计数跨kernel相加。L2 sector按32字节换算，
百分比、带宽和occupancy保留逐kernel，不平均后冒充完整调用指标。
缓存与时钟未锁定，流量是本次profile观察，不是无profiler每次调用的精确字节保证。
普通加速比只使用前述独立CUDA-event计时。

## 数值与捕获范围

P-S全部256批次对同精度CPU FP32 radix2参考检查，最大复数绝对误差为
4.9591e-4/5.1880e-4，relative L2为3.9428e-7/3.9425e-7；同核心两组织统计一致。
保存的CPU参考另抽取批次0/128/255与SciPy complex64 FFT交叉检查，relative L2约4e-7。
这不是逐位正确性证明，也没有新增/放宽浮点通过阈值。
P-P全部16批次对CPU FP64检查，两个核心最大复数绝对误差分别9.2534e-13/1.0057e-12，
relative L2为7.2549e-16/7.1670e-16；四组织分别相同。全部输出有限。
精度与负载不同的两组不混合汇总；原始字段保留在冻结JSON。

所有捕获均在初始化、CPU校验、Graph创建和100次预热之后开始，只包含一次完整调用。
P-S NSYS验证5/4个kernel，P-P验证两个kernel、一次cudaGraphLaunch；全部单stream，
捕获内没有数据复制和内存分配API。NCU节点数与之对应。
最初8份P-P NSYS用默认graph粒度，缺少GPU节点活动表。保留原始报告为失效捕获，
只补采这8份，显式使用node粒度。普通计时和NCU没有因这一修复重跑或换数据。
原runner的NCU宽表缺失项解析不适用，冻结汇总从原始CSV重新核验26项指标均存在，
旧capture_scope字段保留但不作缺失项依据，解析代码已修复。

## 身份、复现与证据位置

- GPU：NVIDIA A100 80GB PCIe，UUID `GPU-0eae5f07-33de-321b-70a9-ffeae49309e6`。
  这是独立新cohort，原计划卡忙碌，使用同型号空闲卡；未与40GB或另一张80GB合并。
- 基础源码HEAD：`2eed1109cbfd9a8e35f87f9133770862386645ce`，本轮profile入口源码及
  二进制、输入、参考、模块、库与hardware model分别记录SHA256。
- 框架库SHA256：`e515f15ee65a1c61e14ca21edd4705eeed9e99fe9531284ec738a0b757eb314f`。
  所有变体先CPU预编译；GPU阶段只加载并测量，不进行搜索、编译或重新校准。
- 原始目录：`results/representative_profiles_20261108`。`campaign.json`是原始30项，
  `nsys_node_repair/campaign.json`明确绑定父campaign，仅修复8份时间线。
- [冻结数据](../generated/representative_profiles_20261108.json)保存36条普通计时、
  884个逐kernel指标、修复前后捕获身份和有效NSYS节点记录。
- [静态指令审计](../generated/representative_profiles_instruction_audit_20261108.json)
  记录8模块32个正逆kernel的编译信息；本实验只测正向16个factor kernel。
- [图](../figures/representative_profiles_20261108.pdf)复用论文现有机制图位置，
  [绘图脚本](../tools/render_representative_profiles.py)默认消费冻结JSON。
- 统计图按“性能—通信/供数—资源代价”重排；性能柱为中位数之比，点/范围为
  三进程内配对比值。DRAM/L2分别归一到同核心逐步执行，绝对GiB保留在
  [图数据CSV](../generated/representative_profiles_chart_data_20261108.csv)。
  [设计说明](../literature/mechanism_figure_design_20261108.md)记录Welder/ROLLER的参考位置、
  分母、各面板的捕获范围以及绘图工具来源。新增寄存器与活跃warp面板，不改变原始测量。

```bash
python scripts/summarize_representative_profiles.py \
  --campaign results/representative_profiles_20261108/campaign.json \
  --output results/representative_profiles_20261108/final_summary \
  --analyze-nsys \
  --nsys-repair-campaign results/representative_profiles_20261108/nsys_node_repair/campaign.json
python paper/tools/render_representative_profiles.py
```

路径按当前主机原始manifest保存。跨主机先准备对应参考、二进制和模块，再生成新身份，
不能把绝对路径改掉后仍冒用原始campaign哈希。重绘论文图不依赖GPU或原始二进制。
现有A10040跨FFT/FWHT/NTT局部分组和FP16边界诊断继续复用其原cohort，
不与本轮求统一均值。本轮不是外部库profile、跨架构验证或所有机制的独立因果分解。
