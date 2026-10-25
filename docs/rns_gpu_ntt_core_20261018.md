# GPU-NTT 核心对齐记录

本记录固定说明 FHE 对照中实际使用的 NTT 核心，以及 cuButterfly 对其可复用部分的吸收边界。源码调查使用仓库内的 pinned checkout，不把端到端统计中的上下文收益误称为另一种 NTT 算法。

各表保留当时的构建和测量边界。后续带 CPU 精确校验的同进程批处理对照及
当前应用残差见 [放置与合批报告](../results/rns_placement_batch_20261019/README.md)。

## 来源和调用关系

`results/e13_assets/sources/HEonGPU` 的 pinned commit `1928a14bb002c5d7f1cb5534eca9fa19b555b4f2` 通过 gitlink 使用 GPU-NTT commit `8a4daf11c7cdea8dde45fa0f8f7183c138d42b3b`。HEonGPU 的 CKKS/BFV 路径调用 `gpuntt::GPU_NTT_Inplace` 或 `GPU_INTT_Inplace`，TFHE bootstrap 的局部 NTT 则调用自己的 N=1024 `SmallNTT` helper。缓存中没有第二套可执行的 FHE GPU NTT 实现。CPU HEXL 是独立的 CPU 对照，不能作为 GPU 核心来源。

GPU-NTT Merge 在 `thirdparty/GPU-NTT/src/include/gpuntt/ntt_merge/ntt.cuh` 和 `src/lib/ntt_merge/ntt.cu` 中定义。对 N=2^14 的 PerPolynomial RNS 配置，前向阶段是 local-log 5 加 9，几何为 `(grid=(32,1), block=(16,16), shared=512 words)` 和 `(grid=(1,32), block=(256,1), shared=512 words)`。N=2^16 使用 7 加 9。前向采用 CT，逆向采用 GS，根表按 bit-reverse 生成，正向结果原生为 bit-reversed。RNS 通过 `blockIdx.z % mod_count` 选择每个素数的 modulus 和 root table。

GPU-NTT 的 Data64 模乘使用 Barrett reduction，并用 `mul.lo.u64`、`mul.hi.u64` 和带借位减法实现 128 位中间量。其 4-Step 路径包含独立的分解、转置和 W 乘法，不能直接当作 Merge kernel 替换。`SmallNTT` 只覆盖设备内 N=1024 局部 helper。

## cuButterfly 的模板化接入

`RnsMapping::arithmetic` 现在把模乘 provider 作为搜索维度暴露：

* `Shoup` 保留原有 cuButterfly FusedCTGS 路径。
* `Barrett` 是 GPU-NTT-compatible FusedCTGS provider，使用相同的 CT/GS 阶段边界、negacyclic fused roots、bit-reversed boundary contract 和独立的每阶段 rows/threads。它只接受所有 modulus 小于 2^62 的请求，超出范围会显式拒绝。
* `GpuNttMerge` 是一个受约束的 RNS provider。它复用同一套 fused stage kernel，但固定为 radix2、Barrett、5+9/7+9 风格的 512-word tile，并要求每个输入和输出行按 N 对齐且互不重叠；`polynomials` 的逻辑顺序可以按 modulus 分组，domain 和物理行由每个 polynomial 的显式序列/offset 共同指定。在 `input == output` 且每行的 input/output offset 相同之时，它直接采用 GPU-NTT 的原地两阶段调用契约；有行置换或不同输入输出布局时仍使用 scratch，避免破坏输入或边界。

原位扩展后的两种地址模板都读取显式 physical offset，不能再把模板名字
解释为已经消除了偏移计算。直接原位执行还要求原生 CT/GS 边界顺序，
其他顺序通过 scratch 支持，详见 [RNS 接口](unified_planner.md#explicit-root-rns-ntt-integration)。
对 `N >= 2^16`，搜索器还会测量 GPU-NTT 1024-word 配置对应的 512-thread 候选，但不会预设它一定获胜。

搜索器会在合法的 FusedCTGS 候选中同时测量两个 provider，并按同一 CUDA Graph、同一输入和同一阶段几何选择。它没有把 HEonGPU 的 mod-raise、key-switch 或 stream 上的外围 kernel 混入 NTT 核心模板。

## 受控结果

在 A100 40GB（CUDA 12.4，sm80，N=16384，B=2，两个约 60-bit prime，forward 和 inverse 分开，同一进程交错 Graph，80 次/样本）上，固定 GPU-NTT 的 5+9 几何。每一行都经过逐元素精确校验，正向和逆向各重复十轮并交替首发顺序：

| 调用契约 | 方向 | FusedCTGS + Shoup | GpuNttMerge + Barrett | GpuNttMerge / FusedCTGS |
|---|---|---:|---:|---:|
| separate input/output | forward | 0.0126976 ms | 0.0148224 ms | 1.167 |
| separate input/output | inverse | 0.0187456 ms | 0.0222080 ms | 1.185 |
| input == output | forward | 0.0156160 ms | 0.0180992 ms | 1.159 |
| input == output | inverse | 0.0176832 ms | 0.0208640 ms | 1.180 |

该结果说明 GPU-NTT 的 Barrett 算术、连续交错布局和原地调用契约都可以被统一框架表达，但在这个 A100 核心对照上并没有超过 Shoup-CTGS。直接构建的 pinned GPU-NTT 仅作为 timing-only 外部控制，未做输出校验，独立进程的动态时钟测得 0.0120--0.0185 ms，不能与上表的同进程交错结果作精确倍数比较。因此不能用它声称某个 FHE 库拥有另一种更强的 NTT 核心。

## 当前结论和后续边界

本次吸收完成的是可复用的 arithmetic、root layout、CT/GS stage descriptor、RNS modulus selection、按行连续的系数访问、逻辑/物理行置换和原地调用契约。当前数据没有证明计算核心是 HEonGPU 端到端优势的根因。若要继续缩小 FHE 应用差距，应对 RNS interleaving、mod-raise/key-switch 周边 kernel、Graph 复用和批量规模做端到端 trace；不能把这些外围收益回填成 NTT 核心加速，也不能把通用任意偏移路径强行改成 GPU-NTT 专用布局。

## 适配器置换回归

HEonGPU 的 `Modulus_Ordered` 请求会先按 modulus 分组，因此
`RnsPolynomial` 的逻辑顺序可能是物理行的置换。此前 compact 地址使用
`poly_index*N`，会把这种合法布局错误地排除在 GPU-NTT provider 外。
现在 eligibility 检查的是输入和输出各自覆盖连续物理行，kernel 用每个
polynomial 的显式 offset 定位行，domain sequence 独立决定 modulus；只有
输入输出别名且每行 offset 相同才走原地两阶段路径。

在当前 libcuntt 上运行 pinned HEonGPU adapter regression（A100-PCIE-40GB，
N=4096/16384，连续、按 modulus、按 polynomial 三种顺序，正逆共 12 例）
全部通过独立 CPU 参考和 GPU-NTT 对照。适配器的 request-local tuner 在这些
请求中仍选择 FusedCTGS radix2/4，说明置换候选已可执行但没有被强行提升为
赢家；这与受控表中 Barrett/GpuNttMerge 慢于 Shoup 的结果一致。

## 原生核心控制

为区分“兼容 provider”与真实 GPU-NTT 核心，新增的
`benchmarks/gpuntt_native_compare.cu` 直接链接 pinned GPU-NTT 静态库，调用
`GPU_NTT_Inplace/GPU_INTT_Inplace` 的 RNS 入口，使用相同的两个 modulus、N
和 batch。A100 上得到的 timing-only 控制为：N=2^14、B=2 的 forward/inverse
分别为 0.0188800/0.0211456 ms，N=2^16、B=2 分别为
0.0167936/0.0185984 ms。它没有替代 cuButterfly 的 correctness regression，
也没有把不同的 Graph/外围调用开销混入结论；完整复现命令见
`benchmarks/README_gpuntt_native_compare.md`，原始数据见
`results/e2e_rns_workload_20261018/gpuntt_native_controls_20261019.csv`。

这个控制说明差距取决于阶段几何和线程组织：N=2^14 的等容量 provider 已经
达到 pinned 核心的测量范围，而 N=2^16 仍显示出明显差距。下一步应吸收其
二维 block ownership、阶段专用 geometry 和 warp-tail barrier 策略，而不是
继续扩大 Barrett 的普通 rows/threads 组合。

在相同两个 modulus、batch=2 和与 pinned 配置等容量的阶段几何下，直接控制与
兼容 provider 的事件时间如下。两者都不包含初始化；兼容 provider 的结果还
经过逐元素校验。N=2^14 的第一阶段使用 16 行 × 32 点，第二阶段使用 1 行
× 512 点；此前手工记录的 32 行第一阶段不是等容量配置，已从主对照中剔除。

| GPU / N | 方向 | pinned GPU-NTT | cuButterfly compatible | compatible / native |
|---|---|---:|---:|---:|
| A100 40GB / 2^14 | forward | 0.0188800 ms | 0.0219648 ms | 1.16 |
| A100 40GB / 2^14 | inverse | 0.0211456 ms | 0.0237440 ms | 1.12 |
| A100 80GB / 2^16 | forward | 0.0167936 ms | 0.0190464 ms | 1.13 |
| A100 80GB / 2^16 | inverse | 0.0185984 ms | 0.0194304 ms | 1.04 |

这组独立进程结果受 A100 时钟状态影响，不能作为最终论文主表；连续三次兼容
测量的中位数用于这里的保守诊断。当前 N=2^14 仍接近但没有稳定超过 pinned
核心，N=2^16 约有 13% 的前向和 4% 的逆向差距。后者不能再归因于“没有搜索
到某个 Barrett 点”，而应继续检查 7+9 阶段的线程/尾段实现，并在同一进程
交错 Graph 中复测。

## Barrett 算术的单变量复核

为排除 128-bit 中间量实现本身造成的误判，2026-10-19 在
`src/ntt_hybrid2d.cuh` 中将 Barrett 的宽乘和带借位减法改为与 pinned
GPU-NTT Data64 路径一致的显式 PTX（`mul.lo.u64`、`mul.hi.u64`、
`sub.cc.u64`、`subc.u64`）。其余 provider、stage 几何、布局和搜索空间均未
改变。正确性测试和 HEonGPU adapter 的 12 个正逆请求全部通过。

固定 5+9/7+9 风格几何的事件测量显示，PTX 只带来很小且不一致的变化。
N=2^16 前向从 0.0194048 ms 到 0.0190464 ms，逆向从 0.0196224 ms 到
0.0194304 ms。N=2^14 的首轮 PTX 记录使用了错误的 32 行第一阶段，未纳入
当前 canonical CSV；按 16 行等容量几何补测后的三次中位数为前向 0.0219648
ms、逆向 0.0237440 ms。原始记录见
`results/e2e_rns_workload_20261018/gpuntt_ptx_arithmetic_20261019.csv`。
因此当前差距的主因不是 Barrett 的乘法指令选择，而是二维线程所有权、阶段专用
几何和尾段同步策略；后续优化应优先接入这些结构，并重新在统一 CUDA Graph
控制下对照 pinned 核心。

随后对尾段同步做了单独的候选实验：仅对 dense `GpuNttMerge` 的 9-stage
第二 pass 去掉最后五个阶段之间的 CTA barrier，输入、根表、算术和阶段几何
均不变。RNS correctness 和 adapter regression 仍通过，但三次重复的固定映射
测量没有稳定收益，完整记录见
`results/e2e_rns_workload_20261018/gpuntt_tail_sync_20261019.csv`。N=2^14 的
forward/inverse 分别为 0.0218624/0.0236800 ms，与当前 barrier 中位数
0.0219648/0.0237440 ms 相近；N=2^16 forward/inverse 为
0.0189824/0.0194688 ms，也只在噪声范围内变化。因此该候选已撤回，不进入
默认搜索空间。原生核心的尾段优化依赖它自己的二维 ownership，
不能只通过在当前一维 stage kernel 中删除同步来复现。

## 历史 1.4x 的同口径复核

此前记录的约 1.4x 并非无依据，也不是新 `GpuNttMerge` 兼容 provider 的
结果。为确认它来自哪条路径，2026-10-19 在 A100 80GB PCIe 上重新运行了
旧的单模数 GPU-NTT Merge 控制和当前 `cuntt_bench` 的成熟 `hybrid2d/radix4`
路径。两者使用相同的 60-bit 模数 `576460756061519873`、正向、自然序输出，
均只统计 kernel event 时间；cuButterfly 每个点逐元素校验通过。GPU-NTT
控制是 pinned 的 timing-only 可执行文件，因此其输出未被当作正确性依据。

| N | batch | GPU-NTT ms | cuButterfly ms | GPU-NTT / cuButterfly |
|---:|---:|---:|---:|---:|
| 2^16 | 4 | 0.0282173 | 0.0250410 | 1.127x |
| 2^16 | 16 | 0.0773345 | 0.0632400 | 1.223x |
| 2^16 | 64 | 0.2903200 | 0.2129840 | 1.363x |
| 2^16 | 256 | 1.1160700 | 0.8018130 | 1.392x |
| 2^18 | 16 | 0.3379040 | 0.2629920 | 1.285x |
| 2^18 | 64 | 1.2993300 | 0.9923650 | 1.309x |
| 2^20 | 4 | 0.3693450 | 0.3738230 | 0.988x |
| 2^20 | 16 | 1.4246600 | 1.3135050 | 1.085x |

八个点的几何平均为 1.214x。中等 N 与中大 batch 的四个点为
1.285--1.392x，其中 `N=2^16, batch=64/256` 正是历史报告中约 1.4x 的
来源。`N=2^20, batch=4` 则略低于 1，说明该优势不是所有 N/B 组合都成立。
这组数据和 V100 上成熟 v0.6 Hybrid2D 的 1.408--1.438x 历史点一致地说明：
cuButterfly 的批量覆盖和层级数据流在合适的中大 batch 区域确实能超过
GPU-NTT；小 batch 或极大 N 的尾部仍需要专门的二维 ownership 和负载覆盖。

原始逐点结果见
`results/e2e_rns_workload_20261018/gpuntt_legacy_same_shape_a100_20261019.csv`。
它与前面的 RNS `GpuNttMerge` 表必须分开解读：前者是成熟单模数
Hybrid2D 与 pinned GPU-NTT 的历史路径复核，后者是新 RNS provider 的
适配性/核心吸收实验。不能用后者当前的 1.04--1.18x 比值否定前者的
1.4x 结果，也不能把前者的单模数优势直接宣称为新 provider 已经复现。

### 自动选择器的单点检查

同一请求使用 `--auto-select` 时，当前安装表选择了 `n1_log=6,
rows_per_block=4, threads=256`，kernel 为 0.248459 ms，选择理由仍是
`unmeasured-feasible platform prior`。已知的 `n1_log=8` 点为 0.212984 ms，
快约 16.7%。两者都逐元素正确，因此这不是 kernel 理论或算术错误，而是当前
安装表没有把本地测得的成熟设计点回填到排序器。结果见
`results/e2e_rns_workload_20261018/gpuntt_auto_selection_20261019.csv`。
后续应先让安装后的轻量校准覆盖 `n1_log/rows/threads` 这一层候选并记录
实测 winner，再扩大到新的二维 RNS provider；不能用固定 `n1_log=8` 的平台
特例掩盖校准缺口。

### 输出顺序拆分

在 `N=2^16, batch=64` 上又把 pinned GPU-NTT 的自然序转换单独关闭。GPU-NTT
不做转换时为 0.228796 ms；打开转换后为 0.290320 ms；当前 Hybrid2D 自然序
输出为 0.212984 ms。前者不能与自然序结果直接作为同一契约比较，但说明
GPU-NTT 的 bit-reversal materialization 在该点额外增加约 26.9% 的调用时间。
因此在完整自然序契约下，cuButterfly/GPU-NTT 的 1.363x 是真实的；只比较
NTT 核心本体时，两者已接近 1.074x，剩余差距主要转化为阶段几何和输出布局
的组织问题，而不是单纯的模乘指令问题。原始拆分记录见
`results/e2e_rns_workload_20261018/gpuntt_output_order_split_20261019.csv`。
