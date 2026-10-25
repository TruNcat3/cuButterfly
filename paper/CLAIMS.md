# 论点—理论—实现—证据对应表

本表将拟表达的贡献与实际证据分开。性能数字的 cohort 和哈希见 evidence/。

2026-10-12 的实证状态以 [复用审计](evidence/EXPERIMENT_REUSE_20261012.md)
及 [五问题实验组织](EXPERIMENTS.md) 为准。统一方法不要求每个可选机制对
每个算子都加速；需要分别验证自动选择、机制适用范围、同核心组织收益与
跨核心迁移，以及完整应用收益。旧模型检查、离线候选池与新实测不混合。

2026-09-21 正文调整以构造规则和贯穿样例为主。下表中的 bit reversal、CT/GS
矩阵、intersection 和容量界等详细推导保留在 [构造底稿](METHOD_BLUEPRINT.md)；
正文保留它们的理由和适用条件，不再要求逐式阅读。硬件映射实例和服务组合数值
仍由对应支撑材料记录。表中的历史测量状态属于其原始 cohort，不能代替后台新构建
的综合验收。

| ID | 论文论点 | 支撑与位置 | 已完成检查 | 仍需的证据/边界 |
| --- | --- | --- | --- | --- |
| M1 | 多类蝶形可共用组织表示 | Background、Method；`OperatorGraph` 与语义独立映射 | 当前接口/IR 核对；六类数学应用列表 | 不代表六类拥有完全相同标量 DAG 或所有 lowering 组合 |
| M2 | CT/GS 的通信骨架可按位反转对应 | Method 变体对应；METHOD_BLUEPRINT.md 位反转关系 | L=1..8 共 3586 条邻接对应 | 不自动保留布局连续性或性能 |
| M3 | 正逆可复用组织方法，但须替换局部逆并反转顺序 | Method 局部 CT/GS、zeta/structured 说明；METHOD_BLUEPRINT.md 逆矩阵 | 24 次浮点正逆检查、20 次精确 NTT、96 个局部逆、12 个 zeta roundtrip | 小规模 CPU 不是所有 GPU 变体的支持证明；低精度另有舍入契约 |
| M4 | 合法块级组合保持语义 | Method 的 Preserving the requested computation 拓扑归纳 | 明确四类前提；与实际边界/事件语义核对 | 条件性正确性论证，不是形式化验证整个 C++/CUDA 工程 |
| M5 | 驻留收益须满足依赖和容量 | Method 活跃集；Execution 有限缓冲生命周期 | 单个大 FFT 的字节计数；buffer 语义核对 | 编译寄存器和真实 spills 仍需测量；不能汇总所有 SM SRAM 当单 CTA 容量 |
| M6 | 写回可对齐后继读取 | Execution layout 契约及单词 XOR 例子 | 5 种 bank 数的双射、行/列访问；非 swizzle 反例 | 不推广为任意多字节/向量访存无冲突；global 合并另测 |
| M7 | 增加独立数据可扩大 overlap 机会 | Execution 稳态 II、资源/依赖下界 | 推导条件；已有 27 点组合实验 | 不保证饱和资源仍可隐藏；FWHT 最大 40.287% 误差未消失 |
| S1 | 已实现语义—映射—lowering—生成—重放链路 | Search；`docs/unified_planner.md` | 代码/文档接口对应 | 特定 abstract schedule 仍可能 needs-lowering；不声称通用任意 CUDA 程序生成 |
| S2 | 校准对象是物理服务，便于复用 | Search；stage service/composition | 当前独立测量、插值和组合机制核对 | 未见编译实例仍需覆盖判断，不能保证零补测 |
| S3 | 将蝶形语义、核心接口、层次组织和物理服务接入同一生成/搜索流程 | Introduction、Related Work；文献对位见 RELATED_WORK.md | 已明确变换生成、融合、布局与层次数据流的先行工作 | 不能把这些单项技术称为首创；流程优势仍需由生成覆盖、受控消融和迁移成本体现 |
| P1 | 在声明的 A100 80GB 合同上达到有边界的竞争力采样性能 | Evaluation 自动库横评；当前 E1 图与独立 NTT 扩展 | 当前 FFT FP32/FP64 各6点对 cuFFT 为 1.103/1.353；FWHT FP32 6点对 Dao 为1.667；独立NTT 12点三进程为1.285且全点精确；历史RC2 FFT36为1.031、VkFFT20为1.026 | 当前E1仅单进程；浮点先报告误差；五组缺匹配外部库；当前与RC2的范围/版本不同；这些是有边界的采样结果，不据此声称版本提升或全面领先 |
| P1b | 弱势区改善后仍需保留尾部边界 | Evaluation RC2 weak-FFT 段落；`evidence/rc2_snapshot_20261002.json` | FP64 weak-FFT10 中 6/10 胜出，最低比值 0.895 | 几何均值领先不等于逐点领先；仍需扩展未覆盖方向、布局和更大规模 |
| P2 | 固定局部核心时，组织优化可以带来收益 | Evaluation 消融二、固定核心组织收益图；历史E3 | FFT两种算术叶的组织收益几何均值为1.217/1.221，各6负载，80GB共72行；40GB独立复现；FWHT/NTT固定核心局部分段收益见E3 | 固定的是算术叶，wrapper/CTA形状可变；E3保持物理分组，仅改变组内交换，不是逐阶段HBM对比；其余算子双核心交叉及同预算重搜未完成 |
| P2b | 相同已声明路径下，更换核心可以改善完整变换 | Motivation；A100 register-tile accurate_core_control | 两点约 1.041x/1.023x 历史配对观察 | 端到端时间含核心资源变化的影响；不等于所有 core/组织组合都更快 |
| M8 | stage bit 导出闭合子图及后继连续布局 | Method component、Execution packed layout；METHOD_BLUEPRINT.md | N=16 的 stage 顺序/有序分段与独立稠密 Hadamard 对照，结果见 construction_checks.json | 数学构造不等于全部 bit permutation 已有 CUDA lowering；连续性不保证所有事务合并 |
| M9 | 子图交集确定边界交接、fan-in/out 和独立包 | Method intersection、Boundary packets；CONTRIBUTION.md | N=2..32 共 1,364 mask 对，66,429 次显式求交和 3,905 个图连通包检查 | 完整组件发布、全蝶形包络；单边图可能更稀疏，部分组件发布需要更细分析；相邻边界的包成员可能不同 |
| M10 | 组织可随域扩展，不绑定固定段数或逻辑层深 | Motivation 层次、Method Assigning reuse across the hierarchy；详细界在 METHOD_BLUEPRINT.md | 在完整本地组件、每 leaf 最多 k_max 位的假设下，L<=G*k_max；递归沿用已有闭合构造 | 两段 leaf 界不是所有两层表示的界；表达扩展不等于已实现所有 lowering、未来 GPU 速度或无通信开销 |
| P3 | 模型应以选择损失而非仅误差评估 | Search 候选确认；Evaluation 有限池排序 | 历史16候选池预算4/8/16的已知排序结果可复用；模型误差及局部holdout独立存档 | E9 stage-service当时零covered先验、使用结构fallback，不能称为校准模型验证；在线搜索成本与组件禁用后重搜仍缺 |
| P4 | 方法能适应不同 GPU | Evaluation 平台与映射 | V100 SM70 / A100 SM80 历史记录存在 | 当前统一版本的匹配迁移验收仍待补 |
| P5 | 蝶形组织收益能在部分完整应用中转化为实际应用收益 | Evaluation 应用吞吐/比值图及完整延迟表 | 当前filter 9点×3进程，3/9对CuPy获益；独立 Poisson endpoint 在 A100 80GB、FP32、N=2048、B=16 的 Graph 完整边界上相对 Torch/CuPy 为 1.159/1.186；QuaRot B1修复后完整调用仍约慢2%；独立FNO B4记录约慢5% | Poisson 是单进程 steady-state endpoint，不代表完整应用矩阵；尚不支持普遍应用加速；QuaRot batch扩展继续；CKKS完整RNS应用缺失；随机权重只验证实现一致性，不代表任务质量；不以单测填应用收益 |

## 本轮不作出的结论

- 所有蝶形算子、所有规模、所有数值类型均达到最优。
- 候选枚举、模块编译、曲线覆盖或硬件初始化任一单项等于完整迁移验收。
- 使用 cuFFTDx 局部核心的完整实现即 cuFFT，或直接调用 cuFFT 是本方法生成的性能。
- A100 40GB/80GB 的差别单独证明了跨架构泛化。
- 两个不同版本、不同 placement 或不同 normalization 的速度可直接拼接比较。
- 最近邻工作只优化算术核心，或仅凭统一术语即可证明方法优于既有生成器。

理论检查重现：`python paper/tools/check_theory.py`。详细结果保留于
`generated/theory_checks.json`；与 GPU correctness/performance 证据分别管理。
