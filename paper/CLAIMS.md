# 论点—理论—实现—证据对应表

**当前状态（2026-11-08）**：A100 冻结实验范围已经完成，足以支持下表的有界主张；不代表所有组合完成或论文跨架构目标完成。当前 FWHT 原生返回三进程配对 GM 为 **1.013–1.031×**，指定输出为 **1.320–1.330×**；另一次公共 DEFAULT 十请求检查为 **1.009×/1.324×**。QuaRot 完整随机权重 7B 四进程结果为 B1 **1.004–1.005×**、B4 **1.015–1.017×**。首次准备优化后三请求仍需 **890.207 秒**，不是轻量首次调用。PRO6000 待用户提供平台后执行。来源、无需追加的 A100 补测及论文边界见[本次收敛记录](evidence/PAPER_CONSOLIDATION_20261108.md)。

以下按日期保留历史状态；旧负结果和当时缺测不代表当前修复版状态。

2026-11-07 修复前后续[三项补证](../docs/a100_evidence_followup_20261107.md)完成：同十请求FWHT原生返回GM为0.936–0.938×；固定leaf强端点下FWHT两种核心为1.318/1.365×，NTT两种核心均保留原配置；[资源链](evidence/RESOURCE_DECISIONS_20261107.md)分别标明后验服务组合验证与前瞻实测选择。NTT数值服务预测及PRO6000跨架构证据未被本轮结果代替。

随后[FWHT布局修复](../docs/fwht_native_repair_20261107.md)在三新进程、相同旧AUTO映射的显式重放上，将原生返回Dao/cu GM从0.941–0.955提高为1.013–1.031，指定输出从1.226–1.229提高为1.320–1.330。相同mapping新旧60对输出逐byte一致。旧十请求的CPU reference实际为FP64，已更正标注并补同精度FP32误差，最大绝对误差0.00048828125、L2相对误差1.503e-7，无新阈值。该新结果不改写前轮快照或冒充所有构建/所有接口均领先。


2026-11-07 收尾：[A100完成范围与设备/构建边界](evidence/A100_CLOSEOUT_20261107.md)。冻结主66、cyclic90、组合138请求/552测量、固定核心及九进程应用队列均完成；旧应用不冒充最新安装构建；同日新增四进程 QuaRot fused 验证使用775a85构建，B1为1.004–1.005×、B4为1.015–1.017×，见[路径报告](../docs/quarot_fused_path_20261107.md)。独立40GB FP16十一请求安装后DEFAULT对cuFFT GM为0.8584099×，范围0.738866–1.08757×，不并入80GB均值。PRO6000 Server96GB仍待测；搜索是工程手段，不新增验收门槛。

本表将拟表达的贡献与实际证据分开。性能数字的 cohort 和哈希见 evidence/。

2026-11-06 同步：新增 [FP16 公共验收](evidence/FP16_PUBLIC_ACCEPTANCE_20261105.md)。16配置/334条阶段记录与12请求/32候选确认完成；9个FP16正向请求的DEFAULT、MEASURE和安装后DEFAULT均完成且选择一致。默认对cuFFT GM为0.853583×，1/9领先，范围0.733471–1.066737×。共享conda安装已验证。该cohort取代图表中的旧单点重放，但不改主66请求、cyclic NTT或应用均值。BF16新增自动横评按用户决定暂缓（cuFFT Xt支持BF16存储）；浮点仍只报告误差。

2026-11-04 同步：当前主横评与应用汇总见 [11月2日证据摘要](evidence/CURRENT_EVIDENCE_20261102.md)，
新增弱点确认见 [11月3日有限验收的便携资源记录](evidence/RESOURCE_DECISIONS_20261107.md)（原始目录 `results/a100_closure_20261103/final_acceptance/`），
FFT-L 作为精度维度进入统一横评，只有公共已选点重放进入性能表；固定控制单独归档，BF16新增实验按用户决定暂缓。证据见 [11月4日整理](evidence/FFT_L_UPDATE_20261104.md)。
三者保留各自精度、构建、硬件和计时身份；新修复版校准状态不等于新全量性能。

2026-10-12 的实证状态以 [复用审计](evidence/EXPERIMENT_REUSE_20261012.md)
及其原始实验账本为准。最新呈现采用 [四节实验组织](EXPERIMENTS.md)。统一方法不要求每个可选机制对
每个算子都加速；需要分别验证自动选择、机制适用范围、同核心组织收益与
跨核心迁移，以及完整应用收益。旧模型检查、离线候选池与新实测不混合。

2026-10-27 中心贡献进一步收敛为：**从蝶形依赖与时空组织导出性能表征，将
容量、通信、核心服务和就绪工作需求与硬件约束结合，实现可复用的组织优化。**
分层混合数据流组织这些可调选择；源码复用是实现证据，适配成本是补充收益。
对应补证见 [硬件约束下的选择闭环](experiments/hardware_characterization_closure_20261027.md)。
下表的 M 编号是既有论点编号，和叙事补测 M1–M3 的实验编号不同。

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
| M7 | 增加独立数据可扩大 overlap 机会，但收益取决于资源、依赖和编排成本 | Execution 稳态 II、资源/依赖下界；Evaluation mechanism controls | 解析条件成立；历史 A100 80GB pipeline/bulk 27 对中 pipeline 全部落后，N4096 B64 为45.636/29.266 µs；factor-streamed FFT 两核心开启 slice overlap 均退化 | 不保证更多 batch、更多缓冲或观察到 overlap 就有收益；历史27点 cohort 的 FWHT 最大40.287%误差保留其原始合同，不代表当前主横评全部FWHT的误差或验收状态 |
| S1 | 已实现语义—映射—lowering—生成—重放链路 | Search；`docs/unified_planner.md` | 代码/文档接口对应 | 特定 abstract schedule 仍可能 needs-lowering；不声称通用任意 CUDA 程序生成 |
| S2 | 校准对象是物理服务，便于复用 | Search；stage service/composition | 当前独立测量、插值和组合机制核对 | 未见编译实例仍需覆盖判断，不能保证零补测 |
| S3 | 用依赖导出的资源需求表征连接时空组织与硬件约束，复用同一优化流程 | Introduction、Method 资源需求表、Search；文献对位见 RELATED_WORK.md | 阶段/数据两轴、四项时空需求及固定算术预算关系已写入正文；M1资源profile提供有界佐证 | 不把层次映射等单项技术称为首创；已有有限三请求的服务/plan留出与公共调用验收（见P3），不等于完整表征已跨平台验证；资源链已逐项区分依赖需求、编译资源与独立服务测量，见RESOURCE_DECISIONS_20261107.md；NTT新增前瞻实测选择，仍无数值服务预测 |
| P1 | 在声明的 A100 80GB 合同上具有有边界的外部竞争力 | Evaluation 自动算子横评；CURRENT_EVIDENCE_20261102.md | 主横评66请求：FFT FP32/FP64 对cuFFT 为11/6对、GM 1.298/1.366（9/11、5/6领先），对VkFFT为1.190/1.354（5/11、6/6领先）；FWHT FP32 对DaoFHT 10对为1.474（9/10领先，包含Dao返回张量向指定输出缓冲区的复制；2026-11-07修复前原生返回GM为0.935586–0.937841、指定输出为1.236356–1.241750，保留为历史负结果；随后[布局修复](../docs/fwht_native_repair_20261107.md)三进程同mapping显式配对，原生返回GM为1.012506–1.030831（各进程7个名义领先），指定输出为1.320079–1.330490（各10/10领先）；2026-11-08单进程公共DEFAULT十请求分别为1.008741/1.324297，两类确认分开报告，两种接口输出工作集不同，不将差额全归因copy）；主NTT word64 5对为1.188；独立cyclic NTT 90请求全部精确，54个GPU-NTT配对GM 1.308、51/54领先，word32/64为1.400/1.265；历史RC2 FFT36为1.031、VkFFT20为1.026，仍仅属历史 | 主横评为单进程Graph screen，37浮点请求有限且报告误差但未设验收阈值，29整数精确；每个竞品使用各自匹配集合；缺失GPU基线不作为零性能，CPU边界另列；不合并主NTT与cyclic或把新增请求带来的均值变化称为版本提升；FFT-L另列P1b/P2，不并入FP32/FP64均值 |
| P1b | 平均收益不消除高精度与低精度FFT、FWHT的外部弱点 | Evaluation Where aggregate gains do not carry over；11月3日有限验收；FP16_PUBLIC_ACCEPTANCE_20261105.md | FP64 FFT N2^20 B16 对cuFFT 0.837907、FP32 FWHT N32768 B32 对DaoFHT 0.938867，均为原构建三独立进程确认；新FP16九点默认横评GM 0.853583、1/9领先，最大点N2^18 B32为1.066737；9点DEFAULT/MEASURE/安装后DEFAULT映射一致，校准和安装已完成 | 新FP16为有界正向请求集合，不替换主66构建，也不代表完整精度/方向矩阵；BF16新增公共横评按用户决定暂缓；浮点未设验收阈值，cuFFT内部算术不透明；旧22点固定控制和单点重放保留历史身份，不混入新均值 |
| P2 | 固定局部核心时，组织优化可跨蝶形带来有条件收益 | Evaluation 固定核心、local grouping与FFT-L控制；历史E3及2026-10-27 M1/M3 | 历史80GB FFT两种leaf各6负载、72行，GM 1.217/1.221；独立FP64强同步参考下prefetch收益缩为1.034/1.093，cuFFT仍更快；当前80GB六点scalar radix2逐阶段/分组控制11–24倍，非强优化对照；独立40GB M1 FFT/FWHT/NTT local2为1.196/1.106/1.199、local4为0.932/1.103/0.816，M3双核心16格均无中位数组织收益；11月7日80GB独立强端点控制在B16下，FWHT两leaf的三进程paired中位为1.318/1.365，NTT两leaf均选回原端点（同映射记1，不是新收益）；FFT-L N4096六条最终确认固定局部算术，B64 FP16/BF16为2.236/2.133、B256为3.089/2.969，B4为1.032/0.979，输出与各自旧参考位级一致 | 不合并80GB/40GB及各构建；固定leaf不固定wrapper/CTA，历史E3与M1是局部分组内交换而非逐阶段HBM，当前六点才是逐阶段外部交换控制；同逻辑HBM边界不等于实测流量相同；历史M3非两侧优化后的强对照；新端点控制两侧均允许调优，FWHT合法池2/6、NTT1/3且共同预算上限6；固定算术leaf不固定整个kernel，FWHT同时调整线程与列分组，不单独归因为列数；FFT-L是固定配置确认而非新校准AUTO，仍落后cuFFT且浮点未设新阈值；不要求每种机制每类均获益，也不声称搜索全局最优 |
| P2b | 相同已声明路径下，更换核心可以改善完整变换 | Motivation；A100 register-tile accurate_core_control | 两点约 1.041x/1.023x 历史配对观察 | 端到端时间含核心资源变化的影响；不等于所有 core/组织组合都更快 |
| M8 | stage bit 导出闭合子图及后继连续布局 | Method component、Execution packed layout；METHOD_BLUEPRINT.md | N=16 的 stage 顺序/有序分段与独立稠密 Hadamard 对照，结果见 construction_checks.json | 数学构造不等于全部 bit permutation 已有 CUDA lowering；连续性不保证所有事务合并 |
| M9 | 子图交集确定边界交接、fan-in/out 和独立包 | Method intersection、Boundary packets；CONTRIBUTION.md | N=2..32 共 1,364 mask 对，66,429 次显式求交和 3,905 个图连通包检查 | 完整组件发布、全蝶形包络；单边图可能更稀疏，部分组件发布需要更细分析；相邻边界的包成员可能不同 |
| M10 | 组织可随域扩展，不绑定固定段数或逻辑层深 | Motivation 层次、Method Assigning reuse across the hierarchy；详细界在 METHOD_BLUEPRINT.md | 在完整本地组件、每 leaf 最多 k_max 位的假设下，L<=G*k_max；递归沿用已有闭合构造 | 两段 leaf 界不是所有两层表示的界；表达扩展不等于已实现所有 lowering、未来 GPU 速度或无通信开销 |
| P3 | 有限候选生成、校准与实测确认是实例化组织空间的工程机制 | Search 候选确认；Evaluation mechanism controls；11月3日有限验收 | 历史16候选池预算4/8/16排序及预算8机制禁用重搜分别存档，合法候选数差异保留；11月3日FFT/FWHT两类、三请求范围完成34个认证service key、117 clean probes、68独立group holdout及15完整plan holdout；producer为calibrated-local-serial，三个请求cell配置赢家3/3，最终预测是事后计算的独立服务组合（覆盖预测不使用完整plan拟合的fallback参数），缺少这些预测先于确认的不可变receipt；11月7日FWHT/NTT冻结资源与候选、screen后冻结赢家并进行三独立进程确认，属于前瞻实测选择，NTT数值cost预测仍未覆盖；当前registry经公共DEFAULT和MEASURE消费，MEASURE接受24个scheduled候选并保留seed身份 | 搜索、模型排序或regret不是独立论文贡献；历史E9零covered先验、结构fallback不改写为校准验证；新有限验收不验证任意新尺寸、overlap或Graph预测，不替代跨平台证据；历史机制禁用重搜仍有候选接受数不等、差异小及独立确认不足，不能以新三cell验收补写为已完成该历史归因 |
| P4 | 同一性能表征与构造能用于不同 GPU 约束下的组织优化 | Evaluation 平台与映射 | A100 80GB 主平台横评与 A100 40GB 受控诊断各自独立；目前未完成所需跨架构实测 | 第二平台为 RTX PRO 6000 Blackwell Server Edition 96GB，匹配版本/语义的跨架构横评待测；历史 V100 仅归档，不作论文数据或论点支持 |
| P5 | 蝶形执行收益能在部分完整应用中转化为有界收益 | Evaluation 应用图及CKKS匹配集成表；CURRENT_EVIDENCE_20261102.md | 当前Poisson 3请求对Torch/cuFFT和CuPy GM为1.097/1.106；固定filter 2请求对CuPy为1.057；CKKS 36短链请求选loop/stage-batch后密文逐元素一致，对原生loop为1.428、匹配batch为1.218、每请求更快原生方案为1.131（33/36领先，3个B1落后）；小参数bootstrap N4096 B1/8为1.030/1.032；历史随机权重FNO两尺寸Graph为1.030/1.056、eager为0.955/0.957；修复current-stream后的历史QuaRot B1/B4 Graph为0.967/0.957且输出一致；更早重算filter九点仅3/9领先及旧QuaRot非有限探针独立归档 | 不混合固定/重算滤波合同及历史/当前构建；filter及旧composed QuaRot为单进程，最新fused QuaRot为四进程（每batch两种顺序各一次），FNO每尺寸两进程，浮点报告误差但未设新验收阈值；CKKS匹配batch由集成构造而非上游API，也不是固定核心归因；sec_level=none、短链与小bootstrap不证明生产安全参数性能；随机权重不证明任务质量，[2026-11-07新fused路径](../docs/quarot_fused_path_20261107.md)完整随机权重7B四进程，B1 wall为1.004201–1.004570、B4为1.014958–1.017074，跨后端输出误差为0；旧composed负结果不删除；不声称统计显著或普遍应用加速 |

## 本轮不作出的结论

- 所有蝶形算子、所有规模、所有数值类型均达到最优。
- 候选枚举、模块编译、曲线覆盖或硬件初始化任一单项等于完整迁移验收。
- 使用 cuFFTDx 局部核心的完整实现即 cuFFT，或直接调用 cuFFT 是本方法生成的性能。
- A100 40GB/80GB 的差别单独证明了跨架构泛化。
- 两个不同版本、不同 placement 或不同 normalization 的速度可直接拼接比较。
- 最近邻工作只优化算术核心，或仅凭统一术语即可证明方法优于既有生成器。

理论检查重现：`python paper/tools/check_theory.py`。详细结果保留于
`generated/theory_checks.json`；与 GPU correctness/performance 证据分别管理。
