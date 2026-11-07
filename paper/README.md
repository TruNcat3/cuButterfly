# cuButterfly 论文工作稿

**cuButterfly: Unleashing Performance of Butterfly Computation with Hierarchical Hybrid Dataflow**

本目录是基于 APPT26 内部原稿重新组织的英文工作稿，采用用户提供的
ISCA2026 IEEE 模板。APPT 原稿尚未发表且未在审；未修改原稿，也未将其写成
已有正式发表成果。模板年份只表示所用格式，本工程不执行投稿。

当前稿件的统一术语见 [TERMINOLOGY.md](TERMINOLOGY.md)。其中区分依赖闭合区域、
tile 子图、驻留 tile 子图、交接、混合数据流、性能表征和搜索工程，正文与图注应
优先使用这些词。

## 2026-11-07 数据同步与贡献收束

同日后续[三项补证](../docs/a100_evidence_followup_20261107.md)已完成。
同十个FWHT请求的新构建三进程对照，指定输出接口为1.236–1.242×，原生返回
接口为0.936–0.938×，后者Dao更快，不能用历史1.474×概括原生竞争力。
固定算术leaf并保留原端点的控制中，FWHT两leaf获得1.318/1.365×，NTT两leaf
均选回原配置。新增两张紧凑单栏表及可迁移JSON；资源链明确原三请求只有FFT/FWHT，
服务组合预测为事后验证，新FWHT/NTT为测前冻结候选后的实测选择，不冒称NTT数值预测。

同日性能合同复核确认原始均值和分母未算错，但修正了 CKKS/bootstrap 的
mean/median 标注。主表新增原生、适配和轴组合的配对数量，并明确 Dao FWHT
包含写回到调用者输出地址的成本，1.474× 不代表裸核心提升。
导出快照补全语义、计时和适配元数据，原测量值及几何均值保持不变。
QuaRot 的新路径与诊断见[验证记录](../docs/quarot_fused_path_20261107.md)。
四个完整 7B 进程已完成，B1 为 1.004–1.005×、B4 为 1.015–1.017×，
已加入应用图和表，保留旧 composed 负结果。两组均接近持平，未声称统计显著。
另将两条未完成随机状态/输出检查的早期 QuaRot 记录移出有效性能图，保留档案及
排除原因；旧 FNO 两点有数值检查，改用未四舍五入的原始中位数计算比值。

当前正文沿“共同依赖 → 组织的资源需求 → 分层混合数据流 → 固定核心归因与外部性能”
展开，搜索和校准仅作为实现手段。实验保持四节：设置、性能对比、消融、分析。
正文新增独立应用对照表，按场景、蝶形类型、对照和加速比展示；QuaRot 的负结果、
不同请求的历史结果，以及 Graph/eager 的差异均保留，不混合计算均值。

新增的应用快照包含 14 个队列对照行和四个 FNO 模式汇总行，来源哈希随快照保存。
主横评、cyclic NTT、CKKS 和 FP16 九请求均值不变。QuaRot 的应用适配路径审计见
[归因记录](../docs/quarot_attribution_20261107.md)，不能将主横评 FWHT 的均值直接当成
其实际短变换请求的收益。贡献范围与文字修订见[本轮记录](evidence/PAPER_REVISION_20261107.md)。

在另一台机器无需原始 `results/` 即可重绘当前三张数据图并构建 PDF：

```bash
make -C paper refresh-figures PYTHON=python
make -C paper all evidence
```

重绘需要 NumPy 和 Matplotlib；PDF 构建遵循本目录已有的 Tectonic 配置。
`refresh-data` 则仍需原始实验目录，用于显式更新证据，不能与离线重绘混用。
PRO6000 尚无实测数据，迁移流程见[目标平台指南](../docs/hardware/pro6000_server96.md)。

## 2026-11-06 FP16 公共选择与安装验收

统一横评图表已加入独立的九请求 FP16 DEFAULT cohort，替换旧单点重放。
新结果对 cuFFT 的几何平均为 **0.854×**，范围 **0.733–1.067×**，1/9 领先。
DEFAULT、MEASURE 和安装后 DEFAULT 的九点映射一致，三次计时分别保留；
没有按每点选取最快的一次。BF16 新增自动横评按用户决定暂缓，语义检查不替代性能矩阵。
主 66 请求、cyclic NTT 和应用统计不变。正文增加局部核心资源与组织收益的边界，
不将搜索或校准指标写成研究贡献。来源见 [验收证据](evidence/FP16_PUBLIC_ACCEPTANCE_20261105.md)。

`make refresh-data` 现通过 `FP16_ACCEPTANCE` 指定本次完整验收目录，导出时检查
父完成记录、输入、实际映射、计时和输出哈希；不完整记录会被拒绝。普通 PDF
构建仍只使用已有图表。两份 PDF 均为 14 页，已检查受影响页面。

## 2026-11-04 精度横评与论证更新

FFT-L已作为FP16/BF16精度维度并入统一算子横评图表，不再单列机制图。当前只有FP16 N8192 B32的公共已选点重放进入横评（18.12/12.90 μs，0.712×）；BF16新增实验按用户决定暂缓。22请求固定控制保留在证据包，不作为自动库结果。边界profile和单段/多段取舍收归通用分析。11月3独立确认的FP64 FFT/FWHT弱点继续保留；原66请求和cyclic NTT均值不变。

正文收紧理想强度与实际Roofline、依赖交集与物理无冲突布局的关系，明确单段和单位展开合法，并区分依赖推导、编译资源与实测服务。实验仍按四节组织，压缩部分应用历史和重复动机描述。

数据与限制见[FFT-L证据](evidence/FFT_L_UPDATE_20261104.md)、[研究总览](../docs/research_status_20261104.md)和[论文评审](evidence/PAPER_ASSESSMENT_20261104.md)。当前修复版校准保存212条有效记录后因GPU占用结束本轮等待，未声称完整校准、安装或新横评完成。

## 2026-11-02 综合结果与 cyclic NTT 补证

已复核 66 请求主横评、90 请求 cyclic NTT、138 个有限域组合合同、六组固定核心
机制对照及当前 Graph 应用结果。主横评与 NTT 扩展分别展示，后者 54 个有效配对的
几何平均加速比为 1.308×，其中 word32 为 1.400×、word64 为 1.265×。表格、摘要和
引言同步采用可追溯生成的数值；36 个真实基线模数限制与三个轻微负结果保留。

主横评的 FFT FP32/FP64 相对 cuFFT 为 1.298×/1.366×，FP32 FWHT 相对 DaoFHT
为 1.474×。浮点仍按用户选择报告误差，不引入通过阈值。CKKS 完整 Graph 请求相对
原生循环为 1.428×，相对逐请求更快的原生方案为 1.131×。有限域默认组合的 138/138
通过是执行覆盖证据，不是外部加速或自动搜索性能。未测跨平台与缺失对照保持明确。

补充 NTT 的 36 个基线测量复用原 cuButterfly 延迟，不冒充重新交错测量。后续补查
纠正了初次汇总遗漏：历史随机权重 FNO Graph 为 1.030×/1.056×；修复当前流后的
QuaRot Graph 为 0.967×/0.957×，输出有限且一致。早期非有限 Graph 探测和慢 1.8%
的 eager 诊断均保留各自身份，不能覆盖后来的有效记录；这些历史数据不混入当前
应用均值。正文还把匹配 profile 的阶段 kernel 数与普通计时分开解释。
详细的支持范围、负结果和待补项见[综合证据核查](evidence/CURRENT_EVIDENCE_20261102.md)。

## 2026-10-28 并行组织范式与论证修订

本轮将核心动机明确为“资源 Roofline 与实际执行之间的差距，以及差距随负载和
硬件变化的原因”，不再收窄成并行供给不足。单栏 Roofline 图用同一理想强度下
两个组织的示意执行点表达差距；图中没有实测百分比或虚构的跨卡数据。
正文加入 batch 增长、适用资源上界变化、局部容量越界三个解析情形，并用同一
小图的容量、分段和交换量推导说明如何作出组织选择。

已有 FFT 两核心×两组织消融前移，直接支持“收益不完全来自更换核心”。更强
组织对照回答剩余收益大小，不作为重新证明这条有限归因的门槛。实测例子复用
既有 FWHT batch 扩展和 NTT 局部分组对照，全体分布及负结果保留。
摘要、引言、方法和实验压缩了开发进度与任务安排式表述，正文聚焦论述、论据
和适用边界。待采集事项另记于研究文档。

来源、派生数值和解析条件见
[论证证据记录](evidence/ORGANIZATION_ARGUMENT_20261028.md)。数值政策和后台
任务不变；A100 40/80GB 不冒充跨架构验证。图稿工具的原有署名保留在
[图目录](figures/README.md)，没有加入本文研究贡献或相关工作。

## 2026-10-28 当前实验图表快照

正文的主横评已切换为 `full_current_20261025` 的 66 个请求，覆盖实际的
12 个算子/精度组合。新增或更新了算子延迟箱图、按精度配对的性能表、
Poisson/滤波/CKKS/小参数 bootstrap 的三面板应用图、CKKS batch 汇总表，
以及固定 radix-2 核心的阶段组织消融图。图表只使用已测值，PRO 6000 留空。

主横评、五点 NTT 补测、两点高维 FFT 补测、应用和历史消融分别保留来源。
浮点仍只报告误差，滤波点显式标为暂定结果。CKKS 同时展示原生循环与匹配
批处理两个对照，另外报告每个请求中更快的原生方案，不用弱对照掩盖损失。
旧 FNO、QuaRot 和重算滤波器实验的负结果仍在正文交代，没有改写成新测量。

从仓库根目录更新当前数据快照、图表和两份 PDF：

```bash
make -C paper refresh PYTHON=/home/wt/yes/envs/cubutterfly/bin/python
```

`refresh-data` 只更新快照及图表；普通 `all evidence` 构建使用已冻结的快照，
不会在后台实验运行时重新取数。来源 hash、实际设计点概要、全部配对延迟和
缺测状态保存在 [当前图表数据](generated/current_refresh/README.md)。最终数据到齐后
使用相同入口重绘，再核对正文和覆盖范围；新增 cohort 不自动混入 66 点均值。
本轮没有调整 GPU 任务、验收阈值或测量范围。

## 2026-10-27 硬件约束与研究重点

高效计算核心仍受目标硬件的访存、缓存、容量与并发能力约束。研究贡献聚焦于
把共同蝶形依赖与时空组织的资源需求联系起来，在这些约束下选择组织方式。
搜索是工程手段。正文区分默认静态选择、安装时模型引导探索、显式邻域调优和
xD 完整计划测量，不把它们描述成同一个遗传算法。旧排序竞赛与 regret 数据留档，
不再作为核心论据或必做实验；消融仍需公平调优、真实 mapping 和独立确认。

实验顺序为先完成 A100 已定义的有限完整清单，再做 PRO6000 迁移与匹配比较。
强制某机制的负收益与自动库相对外部库的性能分别报告，不相互替代。

## 2026-10-27 Roofline 动机与平台范围

图 3 后续压缩为单栏 480 × 184 示意图，只解释资源上界与就绪/驻留工作供给
的差距。原第二面板与实验章节的完整组织对照重复，而且吞吐比值不能单独证明
并行供给不足，因此不再在动机图中展示。正文保留跨算子的观察并指向完整实测图。

论文现只采用 **A100 实测与 RTX PRO 6000 Blackwell Server Edition 96GB 待测**
的口径。A100 80GB 是主横评平台，40GB 是独立诊断，二者不冒充跨架构比较。
正文、图注、引言中的间接结论及参考文献已移除 V100 相关内容，实验计划 E7 同步
改为 A100/PRO 6000。旧数据、旧图和修订记录仅留档，不进入当前论文或补测要求。

图 3 将资源上界与实际供给的差距画在同一符号图中，说明提高理想强度仍可能受
就绪工作、驻留资源和依赖限制。A100 40GB 的 FFT、FWHT、NTT 完整组织对照
仍在实验章节展示：18 请求、54 配置、162 次计时，保留退化点和完整计时范围。
原第二面板的 36 个相对比值、图源和 CSV 保留以便审计；它们不是实测 Roofline，
也不等于外部库加速比。图表工具来源继续记录于
[图稿说明](figures/README.md)，研究论据来自存档实验。

## 2026-10-27 图文压缩与合并

本轮将两份 PDF 从 **16 页压缩至 14 页**（均含参考文献），正文图从 14 张合并为
12 张，保留四节实验结构、全部已展示的实验点与负结果。没有缩小模板字号或页边距。

- 图 2 改为单栏，用阶段分组条带与资源计数替代重复蝶形网络，保留同一 16 点
  计算下的局部规模、独立分量与传输量比较。
- 行列交接、容量细分和独立的共享内存 bank 示例合为一张单栏图；两个地址映射
  仍明确分开，不暗示它们可直接组合。
- 两组固定核心组织实验合为双面板，分别保留 FP32/FP64 的平台、负载、重复次数、
  计时范围和对照口径，不合并统计。Poisson 图按单栏尺寸重排以保持标签可读。
- 计时、误差和证据范围的公共说明集中到实验设置，压缩重复图注及方法/实现的重复
  描述。实验章节词数由 3136 减至 2519（`texcount -sum`，约 20%）。

当前 [PDF](main.pdf)、[图源与重绘说明](figures/README.md) 和
[验证记录](VALIDATION.md) 已同步。页数仍是工作稿状态，不代表已满足最终投稿限制。

## 2026-10-27 研究主线与因果衔接修订

全文围绕“已有高性能专用库，为什么仍值得统一研究蝶形执行”展开。由共同依赖与
不同领域的优化方式出发，先区分 Roofline 揭示的资源机会和实际并行供给，再以
固定核心的组织作用提出可迁移的执行规律这一假设，引出分层混合数据流及其实现。
语义、负载和硬件分别约束什么，以及为什么需要共同构造，均在进入方法前交代。

早一轮动机使用的历史单算子对照已由上述 A100 跨算术组织分布替代。
图源、CSV 和 hash 见 [图表说明](figures/README.md)；历史与当前结果不混算。

摘要、引言、方法、实现、搜索、实验和结论已按同一问题衔接。进一步将贡献收敛为
**依赖结构 → 时空组织的资源需求 → 硬件约束下的优化**，方法中新增阶段/数据
两个维度的空间展开与时间复用需求表，并区分资源表征与具体运行时间预测。
实验正文现为四节：设置、性能比较（含应用）、消融、生成算子分析。主平台为
A100 80GB，第二平台为已确认的 **RTX PRO 6000 Blackwell Server Edition 96GB**，
第二平台结果仍待测。更完整的同预算选择、交叉控制和外部配对 profile 尚待补证据，
不因改稿而视为完成；有界任务及接口缺口见
[补证合同](experiments/hardware_characterization_closure_20261027.md)。
[叙事与证据计划](NARRATIVE_EVIDENCE_PLAN.md) 保留完整研究路径及 M1–M3 补足方案，
[验证记录](VALIDATION.md) 记录构建、图表来源与版面检查。

同日后续完成了以下补证，均与原自动库横评分开统计：

- [M1 扩展实验](evidence/NARRATIVE_SCALING_RESULTS_20261027.md)：A100 40GB 上
  FFT/FWHT/NTT 的 54 配置、162 次独立进程计时，以及九格[匹配 profile](evidence/NARRATIVE_SCALING_PROFILE_20261027.md)。
  固定标量算术原语时，local2 相对 shared-stage 对照的几何均值为
  1.196×/1.106×/1.199×；local4 为 0.932×/1.103×/0.816×。这些不是外部库加速比。
- [M2 复用审计](evidence/ORGANIZATION_REUSE_AUDIT_20261027.md)：贯穿共同构造、
  投影、约束和实际 GPU 模板，明确专用路径例外；参数留出和适配成本尚未完成。
- [M3 核心交叉](evidence/CORE_CROSSING_RESULTS_20261027.md)：FWHT/NTT 的 16 配置、
  48 次独立计时和 NTT 四格 profile。当前固定组织增强均无中位数收益；同核心
  仍可能跨越不同调度实现，这一归因限制保留在正文。

新增六面板实测图保留负结果与三次计时的全范围。NTT 均全量精确验证；本轮浮点
benchmark 只输出舍入后的最大误差，不替代完整误差分布和用户的验收决定。
该轮两份 PDF 均为 16 页（含参考文献）；同日后续图文压缩后的页数见上。
已有章节的重复说明已压缩，未缩小模板字号或页边距。

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

该轮稿件 `main.pdf` 为 13 页，其中正文到结论约 12 页，参考文献另计。增加的篇幅
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

## 早期叙事记录（2026-09-21 至 10-09）

本节保留此前的写作组织记录。最新正文主线、图表数据范围和未完成项以文首
2026-10-27 条目、[叙事与证据计划](NARRATIVE_EVIDENCE_PLAN.md) 及实际正文为准。

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
| 单 FPGA 及文献性能比较 | 使用 A100 原始记录，规划 PRO 6000 跨架构比较及匹配核心、组织和协议的消融 |

## 结构对应的口径

用户确认的“GA”指 **GS / CT 等蝶形变体**。正文给出位反转的邻接对应、
CT/GS 局部矩阵逆关系，以及全图逆运算需要反转因子顺序的说明。
拓扑可对应不代表算术标签、内存连续性、浮点舍入或运行时间相同。
zeta 使用全蝶形的依赖子图；任意 structured 2×2 的逆要求矩阵可逆。
搜索器另称 evolutionary neighborhood search，不将现有实现包装为完整遗传算法。

## 历史图与实验记录

总览图合并原蝶形依赖图和混合组织图的作用，独立架构图展示自动实现及运行边界。
保留定性 Roofline、N=16 子图/写回、bank/offset 和流水时序图。
APPT 原图和先前草图保留供溯源。图源、继承关系和重绘命令见 [图稿说明](figures/README.md)。
定性图和分析样例明确标注，不冒充测量结果；实验图使用独立数据生成器。

A100 40GB 和 A100 80GB 的不同实验 cohort 分开保存。其他旧平台记录仅供归档，
不再用于当前论文或列作后续必测任务。
A100 80GB 旧版图表快照为 71/72 complete：29 FFT、13 NTT、12 FWHT、
5 subset-zeta、5 superset-zeta、6 structured-2x2 和 1 个 legacy alias。
剩余 NTT 位反转输出项属于原构建缺口，独立修复不混入这一快照。
旧版图表结果见 [生成汇总](generated/current_results_summary.json) 和正文图表；
三次试验先取中位数，再计算匹配性能比的几何均值。原有
`generated/results_table.tex` 保留历史 cohort，不作为当前主矩阵汇总。

A100 记录已观察到同核心的组织收益、核心替换收益及随负载变化的组织差异。
A100 80GB 上的严格 FFT 核心/组织四格和 FWHT/word64 NTT 跨算术控制已完成，
但仍是有限 cohort，不能外推为所有核心、算子或平台的最优性。广泛未见配置的
模型泛化、PRO 6000 对照和迁移成本仍按各自进度标注。只有实际改接并测量某外部库，
才能声称该库也得到提升；待补项不预设正结果。

冻结 RC2 的独立汇总见 [RC2 状态](RC2_STATUS_20261002.md) 和
[机器可读快照](evidence/rc2_snapshot_20261002.json)。正文把它作为已接纳的算子 cohort，
同时保留旧版图表和严格消融作为历史/机制证据，避免不同源码、设备和协议混合。
完整研究目标、端到端性能、PRO 6000 验证及投稿定稿尚未完成。

## 参考文献维护

正文使用 `\cite{key}`，文献集中放在 `references.bib`，由
`\bibliographystyle{IEEEtran}` 和 `\bibliography{references}` 生成参考文献。
不要手写编号或维护另一份 `thebibliography`。论文条目和对应软件条目分开；
出版年份依据正式发表记录，预印本和软件访问日期不能冒充会议年份。
新增引用应同时更新核验说明和相关工作对位，明确哪些比较来自全文、哪些仅有
摘要或元数据支持。增加引用不会改变冻结实验的覆盖范围。
