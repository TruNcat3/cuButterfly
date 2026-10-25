# 论文导向实验补充与证据审计（2026-10-12）

这是对 [10 月 11 日冻结计划](targeted_evaluation_plan_20261011.md) 的补充，
不修改原始测量记录，也不将历史与当前构建的数据合并。当前优先顺序为
QuaRot 正确性修复、NTT 同口径复测、机制因果对照、CKKS 应用适配。
新增实验的成功条件是能解释结论并满足验证要求，不是必须测到加速。

用户要求收敛应用数量，但不限制每类只能一个。先覆盖经典 FFT 图像滤波、
CKKS 和 QuaRot；第二个应用须验证不同负载特征，避免堆叠同类应用。
FNO 保留历史结果与问题记录，不继续大量扩展。

## 1. NTT 的历史加速比为何不同

| 配对范围 | 点数 | GPU-NTT / cuButterfly 几何均值 |
| --- | ---: | ---: |
| RC2 完整密码学集合 | 214 | 1.4723× |
| RC2 word32 | 70 | 1.8050× |
| RC2 word64 | 144 | 1.3335× |
| RC2 inverse，跨位宽/模式 | 107 | 1.5716× |
| RC2 coset，跨方向/位宽 | 30 | 1.6561× |
| RC2 word64 cyclic，N=4096/65536，B=1 | 12 | 1.2715× |
| 当前 E1 word64 cyclic，N=4096/65536，B=1 | 4 | 1.1972× |

历史数据来自
`../cuButterfly-v0.9.1-rc2/results/integrated_repairs_20261001_replay_fix1/a100-80gb/ntt/{summary,journal,identity}.json`。
历史子集和当前四点的模数不同，不能从 1.2715× 与 1.1972× 判断同点退化。
当前四点皆逐元素精确通过。历史接近 1.6× 的数字不能代表当前四点。

另有 [历史六点表](../generated/current_results_pairs.csv) 使用相同的
`p=576460756061519873`。其中 N4096、B1、forward 的 cuButterfly/GPU-NTT
时间从 8.796/11.9091 us 变为 7.7312/9.4208 us，比值降低但双方均变快；
N65536 对应 18.668/16.8346 us 变为 13.9264/14.3872 us，比值反而提升。
这些点的旧/新计时协议仍不同，不能量化为单独的版本改进。

复测分两组，保持各自分母：

1. 当前四点各重复三个独立进程，保持当前模数、自然序、完整调用、冻结库、
   UUID 和 CUDA graph 协议。七次 replay 不是七个独立进程。
2. 增补固定 N=4096/65536、B=16/256、forward/inverse 的批量切片，共八点。
   B16/B256 与 RC2 的负载维度重合；模数先固定为当前值，单独报告。
   真正判断版本回退时，需旧/新二进制运行同一合同、同一计时入口。

当前横评只接入 GPU-NTT Merge。其 [上游](https://github.com/Alisah-Ozcan/GPU-NTT)
另有 4-Step，尚未完成同合同接入，不得称已比较该库所有实现。
没有匹配 GPU 基线时，可以补 [Intel HEXL](https://github.com/intel/hexl)
等优化 CPU 实现。HEXL 是 negacyclic NTT，不能直接冒充 cyclic 对照。
其 API 为 uint64，不自动填补所有 word32 或任意精度缺口。
先匹配环、根、顺序、模数和归一化；必须注明 CPU/ISA、线程数、常驻数据
边界，并另列含 H2D/D2H 的卸载耗时。CPU 比值不进入 GPU 比值均值。
现有 Python/C++ radix-2 函数是正确性 oracle，不代表高性能 CPU 库。

## 2. 机制实验要回答因果问题

设计依据是 [Welder §5.2 / Fig.13–14](https://www.usenix.org/conference/osdi23/presentation/shi)
将延迟、kernel 数和访存变化联动，
[FlashFFTConv §4.2 / Table 3](https://arxiv.org/abs/2311.05908)
设置 fusion-only 核心对照，以及
[Ansor §7.4–7.5](https://www.usenix.org/conference/osdi20/presentation/zheng)
分别评估测量预算、时间成本和排序质量。对应 BibTeX key 为
`welder2023`、`flashfftconv2024`、`ansor2020`，已在 `references.bib`。

| 问题 | 受控对照 | 主要观测 |
| --- | --- | --- |
| 数据流能否独立获益 | 固定算术核心，stagewise / 固定层驻留 / 分层混合 | 完整延迟、HBM/L2 字节、边界次数、访存等待 |
| 好核心与好组织能否叠加 | 简单核心/高性能核心 × stagewise/分层混合，共四格 | 各格时间、数据流收益、核心收益与交互，不将它们简单相加 |
| 地址对齐是否减少冲突 | 相同合法设计点，普通/偏移写回布局 | shared 冲突、访存指令、barrier、寄存器与占用 |
| 流水何时能掩盖延迟 | 固定 tile 和分段的串行/重叠配对，再扫 B/N | 完整关键路径、重叠区间、活跃 warp、等待和资源限制 |
| 搜索是否找到好点 | 相同候选池/预算，模型排序、随机顺序、现有搜索 | best-so-far、top-k recall、排序、相对已测最佳点 regret |

固定配置切换用于隔离机制；各受限空间重新调优用于比较可达性能，两类图分开。
如果关闭某策略后该映射不合法，记录不适用，不能偷偷换核心或分段。
所谓已测最佳只指冻结候选池，不是整个理论空间的全局最优。

第一轮选 FFT、FWHT、NTT 各一小一大负载作为六个锚点；FFT 另保留 FP64
精度对照。先枚举合法变体，再冻结清单。subset/superset/xor zeta 与
structured 2x2 以同语义 stagewise 对照检查泛化；没有对应核心组合时明确
不适用，不能将 FFT 的核心替换图当成所有算子的证明。
流水扩展先限 FFT/NTT 各两个 N × 三个 B（1/16/256），不无限追加矩阵。

所有时间来自普通执行、至少三个独立进程；NCU 数据仅解释机制。
预热后捕获明确的一次完整逻辑调用，标注每调用 kernel 数与各阶段归属。
NSYS 报告关键路径及区间并集，重叠 kernel 的时间和不能充当应用时间。
指标必须注明按 kernel、调用或时长加权；不得平均不同 launch 的百分比后
冒充全调用比例。当前小负载 L2 hit-rate 超过 100% 的报告不可直接画图。

现有 E2 四组 FFT 配对可复用其原始证据，但还未覆盖上述矩阵：
两个 prefetch 对照测到约 1.27×/1.17×，两个核心替换约 1.01×。
它们不能代替 resident、stage-overlap、NTT/FWHT 或外部库 profile。
保留无收益和负收益点，重点解释容量、并行度及流水启动/收尾的边界。

## 3. 端到端优先级和当前限制

### 经典 FFT 图像滤波（替代 FNO 的主 FFT 应用）

选完整二维线性卷积，参照开源
[CuPy fftconvolve](https://github.com/cupy/cupy/blob/main/cupyx/scipy/signal/_signaltools.py)
的 full/same 输出定义。第一轮限定 FP32、same 模式、图像边长
512/1024/2048 × 滤波器边长 31/127，共六点，输入和滤波器固定随机 seed。
这是完整滤波任务，不标成训练模型或下游视觉质量结果。

完整计时包括 padding、输入 FFT、滤波器 FFT、频域乘法、逆 FFT 和裁剪。
固定滤波器的缓存频谱可以另列 steady-filter 模式，双方同样缓存，不能
将只有我方预计算的结果与基线完整计算比较。setup、编译、搜索另报。
主图按相同线性卷积语义比较各自合法实现，允许不同 FFT padding 策略，
但不得漏计我方 power-of-two 扩展的成本。另一个同 padding 的内部对照仅
用于归因，不能把基线强行限制到较差 padding 后称其最佳应用性能。

预先用 SciPy FP64 oracle 验证边界、裁剪与误差；GPU 主对照采用 CuPy/cuFFT，
CPU SciPy 性能如报告则单列。目前 CuPy 尚未安装，本轮只冻结设计，未声称
已测量该应用。原有 FNO 的负结果不删除，也不与新应用混算平均值。

### FNO

当前 compact 路径 64/128 的基线/我方延迟分别为
2.7831/2.9458 ms 和 2.8101/2.9651 ms，约 0.945×/0.948×。
64 旧 split 基线为 2.9943 ms，换 compact 后基线快约 6.4%；我方变化很小。
这解释了胜负变化，但不能单独定位最新实现的瓶颈。

当前每 forward 有四对实数 FFT，复用 plan、输出和 workspace；稳态没有
逐次创建 plan，也没有 Python 侧 full-spectrum expansion。
inverse 底层为保护输入复制 compact spectrum，四次合计约 6.49/25.56 MB。
该复制是待量化的成本，尚无本轮阶段 trace 证明它是主要瓶颈。

有限诊断包括：捕获真实各层频谱，独立 clone 后比较 GPU split/compact
和 CPU FP64 oracle；记录输入是否被修改，单独排除共享输出 buffer 生命周期；
预热后捕获一轮 forward，拆分 FFT、复制、其余模型计算及 launch gap。
旧 128 split-vs-compact 的 1.38% 相对误差仍未定位。
`ifftn + irfft` 与 `irfftn` 同归一化下应等价，不能说是不同数学算子。

### QuaRot

保留它作为 FWHT 替换的适用边界，不扩展成 attention 优化研究。
先前随机模型中 Linear4bit 的零 scale 和 cache-position 兼容问题使其性能
数据不具备有效的应用证据资格。新 runner 使用固定 seed 的非退化 int4
权重、正 scale、完整状态 hash，并比较 prefill 和每步 decode 的有限性及误差。
随机模型只能支持实现一致性、性能和内存结论，不支持模型精度结论。
无已确定浮点容差时只报告分布，不自动放宽、不进入正式性能结果。
缺失 FlashAttention 仍阻断默认配置，显式 SDPA 仅标注为诊断配置。

修复后的 A100 80GB 短序列检查已运行完整 7B 结构，B1/prefill32/decode2。
两边状态 hash 相同，224 个量化模块的 1,359,872 个 scale 全部正且有限，
三次输出均有限。但 prefill 最大绝对误差为 7.0703、RMS 为 1.3878，
两次 decode 最大绝对误差为 5.9863/6.1914，尚不能接受为等价输出。
原始报告为
`results/paper_targeted_20261011/quarot_repair_20261012/validation32.json`。
该运行没有设容差，状态为 `numerical_tolerance_unassessed`，未测性能。
它使用更新分位数辅助函数之前已加载的代码；后续完整长序列分位数统计
已改为完整数组精确算法，九个 CPU 测试通过，不能反推该运行含新增 L2 字段。
下一步只做小层数局部诊断，区分同输入 FWHT 差异、原生重复稳定性与逐层
误差放大，不以重新设置宽容差处理这组大误差。

随后完成了四层、prefill8/decode1 的局部诊断，只构建一个随机模型，
不测性能、不改变数据类型。报告位于
`results/paper_targeted_20261011/quarot_repair_20261012/diagnostic4_guard/release/numerics.json`。
16 次相同输入的 Dao/我方 Hadamard 相对 L2 为 0.00057–0.00068，最大绝对
误差为 0.00195–0.00391。prefill 的首层 attention 相对 L2 为 0.0178，
首层 MLP 增至 0.2313，第四层 decoder 为 0.5632，支持量化网络放大小差异。
安装的 Dao 来源为 `external/fast-hadamard-transform`，其 CUDA 源码使用
float 累计并在输出转换前施加 scale；当前 adapter 使用 FP16 native 计算，
输出后再乘 scale，舍入路径确实不同。完整精度合同仍需对齐验证。

同一个模型的两次原生 prefill 完全一致，原生 decode 却在首层 attention
出现相对 L2=0.0781，最终 logits 为 0.5969。首层输入 norm 完全相同，
模型始末 hash 相同。因此 decode 混有原生 attention/缓存执行路径的不稳定，
尚不能定位具体 CUDA 原因，更不能归因于数据流替换。
该诊断不是原生模型或 7B 结果通过；正式 QuaRot 性能仍保持关闭。

如果 FWHT 原始占比为 p、其替换加速为 s，且其它时间不变，则理想端到端
加速为 1 / (1-p+p/s)。例如 p=10%、s=1.6 时也只有约 1.039×。
这只是示例，不是本轮测得的占比；有效新模型需实测 p 和适配成本。

### HEonGPU / CKKS

优先沿 [已固定 HEonGPU](heongpu_e13_build_status.md) 扩展，不再平行引入
多个同类应用。已完成构建及 [单模数正确性 probe](e13/README_heongpu_ntt_adapter_probe.md)，
后者含 host-assisted twist/reindex，不能作为全 GPU RNS 或 bootstrap 性能。
公共 API 已有顺序控制，但仍只有单模数设置，未暴露任意外部根与整组 RNS
合同；应用适配仍需严格处理逐 limb 模数、negacyclic twist、根的置换、
逆变换归一化、in-place、数据布局和 caller stream。

分三级推进，每级同时测原生/替换版本并记录覆盖率：

1. 完整 GPU RNS NTT/INTT 适配：逐元素精确验证，各 limb、方向、布局和
   stream 保持一致，转换全计时，禁止 host 搬运藏在计时外。
2. 固定密文运算链：multiply/relinearize/rescale 与 rotation，实测 NTT
   所占关键路径和适配成本；它是组件链，不能标成完整应用。
3. 完整 regular bootstrap：包含 nested key-switch 路径，记录 native-retained
   dispatch、总时间、峰值内存和解密误差。局部替换只能称局部替换。

先做可用的 N4096 smoke，原上游参数明确不安全，不进入代表性主结果。
原计划 N16384 只是资源候选，尚无有效主实验参数；另一个上游 v2 示例为
N65536，也不能仅凭 N 推断安全性。主结果需固定并验证 Q/P 链、深度、
scale、分布、安全参数与显存条件。setup/keygen/编译/搜索另列。
CKKS 有大量 NTT 是充分研究动机，但仍需 profile 验证实际占比，不能预先
保证 bootstrap 获得与 NTT 单测相同的加速。

## 4. 本轮完成边界

本补充冻结证据口径、有限实验设计和优先级，不声称上述新矩阵已运行。
当前修改交付 QuaRot 的随机状态与数值检查，并纠正 FNO 的数学等价性注释。
完整 CKKS 集成、机制矩阵执行、CPU 优化库接入和新 NTT 复测各自独立记录，
不能从旧数据推导为已经完成。

## 5. 后续执行更新（2026-10-12）

上述内容是执行前及早期失败记录，后续结果保留在独立目录：

* NTT：12 点完整 CPU oracle 就绪，三进程复测使用 GPU2 原有匹配校准，
  无重新校准；`results/paper_targeted_20261012/ntt_extension`。
* 机制：4 组 FWHT/NTT 驻留对照、FFT 核心×overlap 四格完成三进程计时和
  12 个单调用 NCU/NSYS 捕获。巨大驻留比值含单 stage CTA 严重欠填充，
  只能作内部诊断，不能冒充高性能 stagewise 或外部库比较。
  FFT overlap 在两个核心下均约慢 3.6%，负结果保留。
  详见 `results/paper_targeted_20261012/mechanism/README.md`。
* FFT 滤波：6 点×3 独立进程完成。原生 padding CuPy / cuButterfly 比值
  分别为 1.2658/1.1897/1.1087/1.0971/0.3839/0.3254，几何均值 0.7821。
  小图波动明显，大图负收益稳定；pow2 扩展增加工作量，同 padding
  Torch 对照也显示我方仍有额外成本。所有54输出有限，原始误差完整保留，
  未定容差时不能标正式通过。`results/paper_targeted_20261012/fft_filter/summary.md`。
* QuaRot：CPU FP64 oracle 和缓存隔离诊断后，显式增加 `fp32_composed`
  路径，对齐 Dao 的累计/缩放舍入，输入输出仍为 FP16，所有转换计时。
  四层所有局部/整网输出逐位一致；完整7B的512/32负载也在 A10040
  通过33组零容差检查。原生 repeat 不一致经完整预热后消除，不修改其
  优化算法或放宽阈值。A10080性能另测，不混用板卡。
  详见 [QuaRot 执行说明](e13/README_quarot.md)。

  A100 80GB 的两次独立进程完整调用计时已经完成。两次都使用
  `prefill=512/decode=32`、5 次完整序列预热、10 次测量、零容差门禁，
  并交换了 baseline/cuButterfly 的执行顺序。baseline/cuButterfly 中位数
  分别为：prefill 75.01/78.01 ms 与 75.39/79.07 ms，decode
  2385.42/2430.00 ms 与 2430.46/2480.06 ms，端到端
  2460.34/2509.74 ms 与 2505.40/2556.89 ms。端到端比值
  baseline/cuButterfly 为 0.9803 和 0.9799，说明当前兼容路径约慢 2%，
  不能声称端到端加速。FWHT 替换并非该任务的全部关键路径，FP32
  composition 的转换和缩放也计入测量；计划创建和首次使用仍在测量外。
  两轮均通过 33 组逐位输出检查，state hash 一致。两轮原始 JSON 位于
  `results/paper_targeted_20261012/quarot/full512_timing_p{1,2}.json`。

* 外部机制对照：GPU-NTT Merge 在 N65536/B16 的完整 cyclic word64
  调用为 78.13 us，固定 resident8 cuButterfly 对照为 108.54 us，
  比值 0.7198。该 cuButterfly 点是固定共享迭代映射，不是公共 selector
  的最快结果，因而只作机制诊断。FWHT 的 Dao 包不支持 N65536，批准的
  N32768/B16 对照两边均保留完整误差分布并标为
  `accuracy-review-required`。Dao/cuButterfly 中位数为 21.20/29.13 us，
  比值 0.7276；没有人为添加浮点阈值。profile 包含完整 NCU/NSYS，
  具体寄存器、shared 和 sector-derived 字节见
  `results/paper_targeted_20261012/mechanism/external/fwht_float_report/summary.json`。

应用范围没有扩大到更多项目；完整 CKKS device-resident RNS 集成仍未完成。
