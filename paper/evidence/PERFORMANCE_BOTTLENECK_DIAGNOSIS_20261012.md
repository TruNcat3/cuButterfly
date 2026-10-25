# 落后场景诊断记录（2026-10-12）

这份记录只解释当前已经出现落后的场景。它把**应用总延迟**、**xD lowering**、**蝶形核心**、**选择器**和**主机调用边界**分开，避免用某一层的结果替代另一层的结论。除特别说明外，时间均为 A100 80GB 上排除编译、建计划和 warmup 后的稳态时间。

## 结论先行

目前没有证据表明混合数据流的理论本身无法工作。已经确认的主要问题是：

1. **xD FFT 的组合路径仍有不可忽略的边界成本**。实际调用包含实数边界、逐轴复数变换、紧凑 Hermitian 布局和归一化；单个轴的候选时间不能当作完整 xD 变换时间。
2. **QuaRot 的短 FWHT 路径过去把一次变换拆成了多个边界**。FP16 到 FP32、FWHT、显式缩放、FP32 到 FP16，以及某些 K>1 情况下的矩阵乘法，均可能独立产生转换或启动。融合候选已经证明这部分可以大幅缩短，但候选尚未注册到公共搜索器。
3. **FNO 的差距主要发生在应用调用边界**。kernel 总时间已经接近 Torch，eager 模式仍受到重复的 Python/ctypes 检查、reshape/contiguous 和多次小调用影响；CUDA Graph 消除主机间隙后，cuButterfly 反而略快。
4. **C2R 有一个已经证实的工程问题**：存在复数轴时先把紧凑谱复制到 scratch，再执行首个复数轴是冗余的。去掉这次 D2D copy 后，在旧 mapping 强制复现的控制实验中，2D/3D、B1/B4 全部保持精确输出，端到端 eager 时间下降 10.96%--23.21%。
5. **固定点 NCU 显示主要等待来自 scoreboard 和阶段间数据依赖，而不是简单的 occupancy 不足**。活跃 warp 比例为 90.0%--95.9%，但 long-scoreboard stall 仍为 19.1%--41.5%。因此“增加线程数”或“只改共享内存布局”不能自动解决大尺寸 FFT/NTT。
6. **部分历史弱点仍是选择器或实验 cohort 的问题**。例如 QuaRot 结果明确记录 `selection_source=static_model` 且没有 local timing estimate；融合 FWHT 和 C2R candidate 都标记为 `public_search_integrated=false`。这些结果不能被解释成公共搜索器已经找到并部署了候选。

## 逐类证据

### 1. xD FFT 图像滤波：同 padding 后仍落后

最新同 padding、固定 filter cache 的结果在
[`filter_normalization_optimization_v3/default_guard/release/result.json`](../../results/paper_targeted_20261012/filter_normalization_optimization_v3/default_guard/release/result.json)。代表性的 CUDA event 时间如下，左侧为 cuButterfly，右侧为同 padding 的 Torch/cuFFT：

| shape / batch | forward | inverse | inverse（filter cache） |
| --- | ---: | ---: | ---: |
| 1024, B1 | 0.326 / 0.311 ms | 0.301 / 0.270 ms | 0.224 / 0.201 ms |
| 1024, B4 | 1.142 / 1.086 ms | 1.045 / 0.973 ms | 0.973 / 0.912 ms |
| 2048, B1 | 1.448 / 1.339 ms | 1.286 / 1.148 ms | 0.961 / 0.868 ms |
| 2048, B4 | 4.442 / 4.334 ms | 4.035 / 3.867 ms | 3.703 / 3.575 ms |

过滤器 cache 只消除了 filter 的一次变换，不能消除 image 的 R2C、逐轴 C2C、乘法、C2R 和归一化。cache 后的剩余差距约为 3.6%--11.5%，所以原来的 padding 标签不能再作为主要解释。

当前 lowering 在 [`src/xd_real_fft.cu`](../../src/xd_real_fft.cu) 中明确包含：

- R2C/D2Z vendor boundary；
- 对紧凑复数谱逐轴执行 C2C 或自定义轴 kernel；
- C2R/Z2D vendor boundary；
- 必要时的 permutation、ping-pong workspace 和独立 scale kernel。

2048×2048、B4 的选择器确实执行了 runtime measured，但候选的轴核心时间只有约 0.365 ms；完整调用是数毫秒。这说明候选时间和完整计划之间存在多个没有被单轴候选覆盖的成本：

- 实数边界和紧凑谱的读写；
- 逐轴执行之间的 stream 顺序与 workspace 生命周期；
- 多个 prefix loop 对 batch 的重复启动；
- 独立的逐元素乘法或归一化；
- 复杂布局下 cuFFT 的 strided plan 与自定义轴核心之间的接口成本。

NCU 还没有对这组完整 xD 大调用做可加性的 kernel trace，因此暂时不能把剩余 3.6%--11.5% 归因到某一个轴核心。这个限制是证据边界，不是“理论已经失败”。

### 2. C2R：已经确认的冗余 copy

候选修改只改 inverse 分支：有复数轴时直接让首个 C2C axis 从 caller compact spectrum out-of-place 写入 scratch；只有 rank-1/no-axis 才保留原来的 D2D copy。最终对照在
[`c2r_copy_elision_gpu_v5/comparison.json`](../../results/paper_targeted_20261012/c2r_copy_elision_gpu_v5/comparison.json)。

在旧库 mapping 被强制复现的情况下，8 个点都满足：输入保持不变、输出精确一致。时间改善为：

| shape / batch | eager old → new | 降幅 |
| --- | ---: | ---: |
| 1024² / 1 | 28.621 → 22.221 ms | 22.36% |
| 1024² / 4 | 84.890 → 71.219 ms | 16.10% |
| 2048² / 1 | 88.320 → 78.643 ms | 10.96% |
| 2048² / 4 | 374.016 → 308.582 ms | 17.49% |
| 64³ / 1 | 17.869 → 13.722 ms | 23.21% |
| 64³ / 4 | 28.826 → 22.374 ms | 22.38% |
| 128³ / 1 | 40.755 → 31.744 ms | 22.11% |
| 128³ / 4 | 193.075 → 151.603 ms | 21.48% |

之前一次实验出现 128³ 的错误，是 old/new selector 选了不同轴序 `[1,0]` 与 `[0,1]`，不是 copy-elision 的数值错误。v5 将 old mapping replay 到 new library 后消除了这个混杂因素。CUDA Graph 结果不能使用，因为两边的 harness 都在 default stream 上 capture，失败原因是测试协议而不是算法。

这个修复还没有进入公共 candidate registry，也没有完成重新校准和综合矩阵，因此当前只能称为“已验证的局部工程修复”。

### 3. QuaRot：单算子可改善，端到端收益被边界稀释

短 FWHT 候选的独立实验在
[`fused_short_fwht_candidate/guard/release/result.json`](../../results/paper_targeted_20261012/fused_short_fwht_candidate/guard/release/result.json)。它使用一个 kernel、FP16 load/store、FP32 accumulation 和一次显式 scale，四个 N32/N64 点均与 Dao 及原 composed 路径逐元素一致：

| case | Dao | 原 composed | fused candidate |
| --- | ---: | ---: | ---: |
| N32, B128 | 17.90 µs | 71.68 µs | 24.46 µs |
| N32, B65536 | 51.91 µs | 83.20 µs | 24.18 µs |
| N64, B172 | 17.80 µs | 72.56 µs | 25.31 µs |
| N64, B88064 | 69.02 µs | 147.88 µs | 24.46 µs |

但 QuaRot adapter 的当前代码仍在一次调用中执行 `contiguous()`、可选的 FP16→FP32 cast、FWHT、`output.mul_(scale)` 和输出 narrowing。K>1 时还要把 `hadK` 转成目标 dtype 并执行矩阵乘法。对应源码是 [`quarot_adapter.py:393`](../../paper/experiments/e13/quarot_adapter.py:393)。因此端到端的主要问题有三个层次：

1. **核心边界**：原 composed 路径的多 kernel 和多次读写。
2. **适配边界**：布局整理、dtype 转换和单独 scale。
3. **应用占比**：FWHT 只占 QuaRot 的一部分，attention、线性层和其他 PyTorch 操作不会因为 FWHT 加速而消失。

完整模型在相同输入、零误差门禁下：

| batch | Dao | public default | explicit fused candidate |
| --- | ---: | ---: | ---: |
| 1 | 2482.40 ms | 2530.92 ms | 2461.87 ms |
| 4 | 2590.96 ms | 2644.61 ms | 2560.40 ms |

这个候选的结果仍不代表公共安装后的默认路径，因为结果明确标记 `public_search_integrated=false`，而且 plan metadata 的 `selection_source=static_model`、`selection_reason` 明确写着没有 local timing estimate。也就是说，这一轮同时暴露了实现边界和搜索集成缺口。

### 4. FNO：kernel 接近，eager 应用仍慢

FNO 的 kernel-only trace 在
[`application_diagnostics/fno_v1/release/fno_cubutterfly.profile.json`](../../results/paper_targeted_20261012/application_diagnostics/fno_v1/release/fno_cubutterfly.profile.json)。kernel duration sum 为：

| backend | 全部 kernel | FFT kernel |
| --- | ---: | ---: |
| cuButterfly | 594.044 µs | 73.758 µs |
| Torch/cuFFT | 604.610 µs | 74.015 µs |

但 eager whole-model 仍有 2.7%--6.2% 的落后，CUDA Graph replay 则领先约 1.9%--3.5%。这组现象排除了“蝶形计算核心完全慢”这一简单解释，剩余原因更符合调用边界：

- 每次 `rfftn/irfftn` 仍有 Python adapter、ctypes 调用和 shape/dtype/stream 检查；
- `contiguous().reshape(...)` 可能产生临时或至少一次额外视图检查；
- FNO 一次 forward 包含多个小 FFT，host launch gap 不能由单个 FFT 的 device 时间代表；
- graph replay 将两边的 host 间隙一起移除，所以 graph 结果不能直接替代 eager API 结果。

这也解释了为什么只把 FFT kernel 再优化一点，端到端不一定同比改善。要继续缩小 FNO 差距，需要稳定的直接 binding 或 graph-friendly 执行协议，最好把多个固定形状 FFT 与相邻 pointwise 操作合并进一次应用级 capture。

### 5. 固定点 FFT / FWHT / NTT：主要是数据依赖和阶段边界

NCU 成功采样在
[`performance_bottleneck_profile_20261012/guarded_v3/guard/release/stage_metrics.csv`](../../results/performance_bottleneck_profile_20261012/guarded_v3/guard/release/stage_metrics.csv)。同一个 N=2^20、B=4、256 threads 固定点的结果是：

| path | stage group 0 / 1 | registers | long-scoreboard stall |
| --- | ---: | --- | ---: |
| FWHT linear | 37.79 / 143.23 µs | 16 / 16 | 31.30% / 19.05% |
| FWHT writer-aligned | 86.34 / 146.53 µs | 27 / 30 | 12.03% / 18.26% |
| FFT linear | 131.33 / 318.72 µs | 29 / 30 | 38.43% / 36.33% |
| FFT writer-aligned | 148.61 / 316.42 µs | 32 / 30 | 25.11% / 33.03% |
| NTT linear | 163.33 / 453.47 µs | 30 / 29 | 26.08% / 25.40% |
| NTT writer-aligned | 202.50 / 452.90 µs | 30 / 32 | 16.84% / 24.02% |

这里有两个容易混淆的事实：

- writer-aligned 的 scoreboard 降低，并不代表总时间必然下降。FWHT 第一阶段从 37.79 增到 86.34 µs，寄存器从 16 增到 27，说明地址计算、共享内存交换和阶段收尾的代价超过了等待减少。
- active warp 比例为 90.0%--95.9%，所以当前证据不支持“只是 occupancy 不够”。FFT 的两个阶段 scoreboard 仍在 33%--38%；NTT 第二阶段的读写量约 129 MB，耗时约 453 µs，当前 writer layout 没有改变这一主导项。

因此大尺寸落后更像是**阶段间数据搬运和依赖链没有被完全隐藏**，而不是混合数据流不适用于这些算子。要验证更细的原因，必须在同一完整 xD/NTT 调用下继续对照 global load/store、barrier、地址指令、寄存器和实际 launch 间隙；不能拿固定点的 kernel 时间直接宣称整个库的全局最优。

### 6. 历史公共矩阵中仍可见的边界弱点

旧公共 selector cohort [`comprehensive_20260919/summary.json`](../../results/comprehensive_20260919/summary.json) 仍保留若干具有诊断价值的负例，不能与 2026-10-12 的新候选结果混为同一版本：

- A100 40GB 上 FP32 FFT 对 cuFFT 的 geomean 为 0.989，最低点约 0.666；FP64 FFT geomean 为 0.947，最低点约 0.670。
- A100 80GB 上 FP32 FFT 的 geomean 为 1.013，但最低点约 0.905；FP64 最低点约 0.871。
- bit-reversed NTT 是明显的语义边界弱点：A100 40GB/80GB 对 GPU-NTT 的单点比分别约 0.595/0.491。
- FWHT 的总体 geomean 较好，但仍有少数小 N、小 batch 点接近或略慢于 Dao，最低约 0.946/0.976。

这些旧点的共同特点是小工作量、单 batch、in-place 或非自然输出顺序。它们更容易被固定 kernel launch、顺序转换、bit-reversal 读写和 host API 占据。它们不能证明中大型自然顺序 NTT 或当前融合 FWHT 候选也同样落后；反过来，也不能因为中大型点领先就把这些语义边界删掉。

## 跨场景根因归类

| 层次 | 已看到的现象 | 具体原因 | 当前证据边界 |
| --- | --- | --- | --- |
| 语义/lowering | xD 完整调用比单轴候选大很多 | R2C/C2R、紧凑谱、逐轴执行、workspace、scale 和 permutation 组成了多个阶段 | 已知源码路径，尚缺完整 xD kernel 可加性 trace |
| 核心 kernel | FFT/NTT scoreboard 高，writer layout 仍可能更慢 | global dependency、阶段 handoff、地址/交换开销与寄存器压力共同作用 | NCU 已证实等待结构，尚未找到全局最佳 mapping |
| 选择器 | candidate standalone 很好，默认应用未必用到 | candidate 未注册，或使用 static model/first executable fallback | metadata 已明确记录，需集成后重新校准 |
| 适配层 | FNO/QuaRot eager 落后而 graph/融合改善 | Python/ctypes、contiguous、dtype cast、scale、多个小调用 | adapter 源码和受控结果已支持 |
| 应用占比 | 单算子加速未线性传到端到端 | FFT/FWHT 只占完整模型的一部分 | full-model 结果支持，但需逐层 breakdown 才能量化每项 |
| 语义边界 | bit-reversed、in-place、小 N 明显弱 | 额外顺序转换、别名约束和 launch latency | 旧矩阵有负例，当前版本还需重新覆盖这些组合 |

## 目前不能下的结论

- 不能把 FNO 的 eager 差距写成 cuButterfly FFT kernel 慢；kernel trace 已接近持平。
- 不能把 C2R v5 的改善写成所有 xD 计划都已改善；candidate 尚未进入公共 selector。
- 不能把 QuaRot candidate 的端到端改善写成默认安装结果；当前搜索路径仍有 static-model fallback。
- 不能把一次固定点 NCU 的 winner 当成整个设计空间的 winner；它只回答该 N、batch、线程数和 mapping 下的机制问题。
- 不能把历史 bit-reversed NTT 单点负例扩展成所有 NTT；2026-10-12 的 GPU-NTT Merge 对照 cohort 的 12 点全部领先，但它没有覆盖旧矩阵的所有输出顺序和应用组合。

## 下一步应按问题闭环，而不是继续盲目扩展矩阵

1. **先集成再测**：将 fused short FWHT 和 C2R copy-elision 加入公共 candidate registry，使用同一 A100 校准入口重新生成 registry，再复跑 QuaRot、xD real FFT 和综合矩阵。
2. **补完整 xD trace**：对 2048² B4 和 128³ B4 只采样已选 plan，分别记录 R2C/C2C/C2R、permutation、scale、D2D 和 API launch gap，不把注释事件或 interval sum 相加。
3. **补语义边界**：对 bit-reversed NTT、in-place FFT、小 N/B1 单独保留失败或落后行，确认是转换成本、别名限制还是 core 本身；不通过替换输出顺序来隐藏差距。
4. **补应用 breakdown**：FNO 记录每个 FFT 调用及其 host/device 区间，QuaRot 记录 cast、FWHT、scale、hadK 和 attention/linear 的占比，确保单算子收益能和端到端收益对应。
5. **FP64 单独处理**：FP64 仍需检查实际选择的是 register/local/axis 哪条 core，以及动态 twiddle 和寄存器资源；不能用 FP32 的结论代替 FP64。

这些步骤分别对应 lowering、kernel、selector 和应用边界，完成后才能判断剩余差距是可修的工程问题、硬件特定的 core 限制，还是理论范式本身的边界。
