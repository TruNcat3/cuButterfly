# FP64 factor-streamed FFT：CTA 内 tile 分组复测

本轮针对长一维 FFT 的供数和驻留问题，固定已有的 cuFFTDx block 计算核心，只改变
CTA 内连续处理的 tile 数、factor 形状、每线程元素数和预取深度。目标是回答一个
工程问题：在相同蝶形计算核心上，是否能通过公共映射参数找到更好的组织点。结果表明
可以找到稳定的改进点，但仍不能把这组有限显式枚举当作自动选择器的全空间最优结果。

## 实验口径

测试平台是 A100 80GB PCIe（SM80，GPU UUID
`GPU-2971e500-d3f6-d7e6-e5f2-cb67a06fff02`）。输入为确定性随机 FP64 复数，
`N=2^20`，主实验 `B=16`，正向和逆向均采用不归一化语义。每个 case 使用同一输入、
同一 CUDA stream、同一完整 CUDA Graph 和相同的 warmup=5、repeat=30、trial=7
协议。plan 创建、JIT 编译、第一次执行、workspace、Graph capture 和 CPU 参考均在
计时外；表中的时间只表示重复 Graph launch 的 CUDA event 中位数。cuFFT 也通过同一
公共 graph runner 建立 plan 和测量，因此不把 cuButterfly 的初始化或编译时间加入
任一方。

CPU 参考使用 FP64 逐 batch 计算。报告保留最大绝对误差、相对 L2、非有限值数量和
实际检查的 batch 数，不把“输出有限”写成通过了一个新设定的数值阈值。所有本轮输出
均有限，且误差数量级与同精度 FFT 参考一致。

实验分为两组。`factor_graph_20261025.json` 扫描原始候选，
`factor_graph_neighbors_20261025.json` 在最快点附近加入 `t=2`、`columns=8`、
`EPT=16` 以及 `9+11`、`11+9` 的不对称 factor。`cufft` case 明确关闭继承，
代表完整外部基线。所有 factor case 的逻辑宏阶段仍为 `[20]`，物理 factor 为
`10+10`，所以这里比较的是同一计算核心下的执行组织，而不是替换算法。

## 性能结果

单位为 ms，数字为各自 suite 的中位数。

| 方向、batch | `t=4,EPT=8,C=4,D=0` | `t=1,EPT=8,C=4,D=0` | cuFFT | 最佳点相对 cuFFT |
| --- | ---: | ---: | ---: | ---: |
| 正向，B=4 | 0.3563 | **0.2738** | 0.2689 | 0.982× |
| 正向，B=16（主点） | 1.1244 | **0.9715** | 0.8889 | 0.915× |
| 逆向，B=16 | 1.1283 | **0.9741** | 0.8944 | 0.918× |
| 正向，B=64 | 4.3148 | **3.7890** | 3.4434 | 0.909× |

相对于同 suite 的 `t=4` 点，`t=1` 在 B=4、16、64 上分别降低约 23.2%、13.6%
和 12.2% 的延迟。逆向方向的收益相近，说明这不是只对正向旋转因子有效的特例。
但长 batch 的绝对差距没有消失：B=64 时仍比 cuFFT 慢约 10%。因此本轮支持“核心
不变时组织参数有稳定收益”，不支持“已经全面超过 cuFFT”或“理论已达到全局最优”。

相邻点的 B=16 正向结果如下，完整负结果保留用于检查搜索是否偏好错误。

| 映射点 | 延迟 ms | 相对最快 `t=1` |
| --- | ---: | ---: |
| `t=1,EPT=8,C=4,D=0` | **0.9733** | 1.000× |
| `t=2,EPT=8,C=4,D=0` | 1.0644 | 1.094× |
| `t=1,EPT=16,C=4,D=0` | 1.1774 | 1.210× |
| `t=1,EPT=8,C=8,D=0` | 1.3035 | 1.339× |
| `t=1,EPT=8,C=4,D=0, 9+11` | 1.0088 | 1.036× |
| `t=1,EPT=8,C=4,D=0, 11+9` | 1.0445 | 1.073× |
| `t=4,EPT=8,C=4,D=0` | 1.1283 | 1.159× |

更大的 CTA tile 组数、EPT 或 columns 并没有带来单调收益。它们仍然是合法的设计
点，但资源占用和并发形状改变后，不能由“更大的 tile”这一名称推断更优。`t=1` 的
设计点概要是：**两阶段 10+10，tile 1024，CTA 内 1 个连续 tile 组，4 columns，
EPT=8，prefetch depth=0，512 threads，cuFFTDx block，动态 I/O，写回布局对齐**。
这里的 `t=1` 指每个 CTA 的连续组数，不表示整个 FFT 只执行一个 tile。

## 机制和 profile 证据

NCU 只用于解释差异，正式延迟来自未插桩的 runner。`t=4` 和 `t=1` 都是两个物理
kernel 节点，完整 Graph 节点数相同。对 B=16 的两个 stage 采样显示：

| 设计点 | registers/thread | 动态 shared/CTA | active warps | long-scoreboard stall |
| --- | ---: | ---: | ---: | ---: |
| `t=4` | 64 | 64 KiB | 47.9–49.1% | 41.4–49.1% |
| `t=1` | 64 | 64 KiB | 49.0–49.1% | 38.2–40.6% |

两点的 shared 和寄存器资源基本相同，`t=1` 的收益主要来自 CTA 组调度和供数等待的
变化，而非算术指令数减少。`t=1` 的 global load/store sector 利用率仍接近满载，
SM throughput 约为 29.1–39.4%，后半 stage 仍有约 40.6% 的 long-scoreboard stall。
因此剩余差距不能简单归因于“没有把 HBM 用满”，也不能仅凭该 profile 断定某一个
具体地址端口是唯一原因。下一步应在同一高性能核心上继续分离局部交换、阶段内供数
和批量调度的影响，再决定是否扩展模板；不应退回只扩大标量核心枚举。

## 正确性、生命周期和硬件身份

每个 case 都在 Graph capture 前完成 lazy provider 初始化，输入、输出、plan、graph
和 stream 在测量及验证完成后才析构。CPU reference 文件由独立的无 GPU prepare 阶段
生成，measure 阶段校验 config、binary、`libcuntt.so`、模块和 reference 的 SHA256。
运行时显式加载同一 A100 80GB hardware model，并由库再次核对可见设备的名称、SM
版本和显存容量。修改后的 runner 若没有确认该静态模型，会拒绝进入测量，避免把
V100、A100 40GB 或另一张 A100 的结果误写入同一组证据。

本轮所有结果的 `nonfinite=0`，B=4、16、64 均检查了完整 batch。FP64 主点相对 CPU
参考的最大绝对误差约为 `4.2–4.6e-12`，相对 L2 约为 `8.4–9.0e-16`；cuFFT 的
误差也单独报告。误差尚未被压成一个人为的“通过/失败”结论。

## 复现和边界

CPU preparation 和 GPU measurement 已拆开，可使用：

```sh
python scripts/run_factor_graph_experiment.py prepare \
  --config configs/experiments/factor_graph_neighbors_20261025.json \
  --prepared results/factor_graph_20261025/neighbors.json \
  --binary build-e13-cufftdx/cubutterfly_graph_bench \
  --hardware-model config/hardware/NVIDIA-A100-80GB-PCIe-sm80-84974239744B/hardware_model.json

CUBUTTERFLY_COMPILE_MODE=research python scripts/run_isolated_gpu.py \
  --gpu-uuid GPU-2971e500-d3f6-d7e6-e5f2-cb67a06fff02 \
  --report-dir results/factor_graph_20261025/neighbors_gpu1 \
  --until 2026-10-25T07:30:00+08:00 --idle-seconds 3 -- \
  python scripts/run_factor_graph_experiment.py measure \
  --prepared results/factor_graph_20261025/neighbors.json \
  --binary build-e13-cufftdx/cubutterfly_graph_bench
```

`results/factor_graph_20261025/*_gpu1/release/` 保存原始 JSON、stderr、GPU 身份、
准备 manifest 和 runner 日志。该实验是一个受控的显式 mapping 复测，覆盖了当前
发现的局部邻域和 B=4/16/64、正逆方向；它不是全量 N、batch、精度、算子矩阵，也不
证明公共 AUTO selector 已经把这些新点纳入排序。公共搜索和安装校准的自动接入仍应
作为下一阶段工作单独验证。

## 阶段内计算核心与供数结论

本轮的 FFT-H 因子路径没有显式选择一个可公开命名的“基 2”“基 4”或“step-4”
算法。当前最快的 `factor-streamed` 设计点由 `scripts/compile_module.py` 生成
`cufftdx::Block` 类型，组合了 FFT size、精度、`ElementsPerThread`、
`FFTsPerBlock` 和目标 SM，然后在 `src/fft_factor_streamed.cuh` 中调用
`FFT().execute(values, work)`。cuFFTDx 的内部 radix 和蝶形调度是私有实现，当前
MathDx 数据库只公开元素数、线程数、FFT 数、shared-memory 需求等约束。因此这里
只能准确地说“阶段内使用 cuFFTDx 的高性能局部 FFT 核心”，不能把它误标成基 2、
基 4 或 step-4。仓库中另有 `fft_factor_native.cuh`，它明确使用两个 radix-2
register codelet，但它不是本轮 A100 的最快路径；`fft_radix4_codelet.cuh` 也
不是当前因子点的实际核心。

阶段内的实际顺序是：global load 到 shared ring，必要时执行 `cp.async`，同步后把
tile 送入寄存器中的 cuFFTDx FFT，随后完成 shared 中的 twiddle、布局/写回，再同步
并 global store。这个路径已经把阶段内的局部算术交给高性能核心，但阶段边界仍有
shared/global 中间结果和同步。因而“阶段内供数”是可测的供数与调度等待问题，而不
是已经证明某一个 radix 公式错误。

## Prefetch 与核心形状复测

修复前的 `t=1, depth=1` 数据不能作为跨 tile 重叠的证据，因为 CTA 内没有后继
tile。旧版 `t>1,depth=1` 则已有 next-load/current-FFT 重叠：当前 tile 装入
寄存器后，输入槽即可承接下一 tile，FFT 使用独立工作区。因此不能把旧版一个输入
槽解释成“没有流水”。扩展预取距离后，
`prefetch_depth=d` 使用 `d+1` 个输入槽，并在槽位释放后回填后继 tile；`d=0`
保持单槽同步路径。对只有一个 tile 的 CTA，后继 tile 不存在，因此统一 lowering
将 `t=1,d>0` 退化为同步单槽，不为一个无效流水轴支付共享内存和同步成本。只有
`data_tiles_per_cta>1` 时才启用真正的 `cp.async` ring。

以下是扩展输入 ring 后、等待指令修正前的 A100 80GB、FP64、N=2^20、B=16、正向
FFT 独立 GPU1 运行，数字
是中位延迟（ms），越低越好。设计点均为两阶段 `10+10`、EPT=8、columns=2、
cuFFTDx block、动态 I/O；`t` 为每 CTA 连续 tile 数，`d` 为预取深度：

| 设计点 | 延迟 | 相对 cuFFT 延迟 |
| --- | ---: | ---: |
| `t=1,d=0` | 0.9139 | 1.019x |
| `t=1,d=1`（退化同步单槽） | 0.9269 | 1.033x |
| `t=1,d=2`（退化同步单槽） | 0.9330 | 1.040x |
| `t=2,d=1` | 1.2864 | 1.434x |
| `t=2,d=2` | 1.3527 | 1.508x |
| cuFFT | 0.8972 | 1.000x |

`t=1,d=1` 这一点在新语义下已经恢复到接近最佳 `t=1,d=0`，但它的优势来自退化
lowering 消除了无效的环形缓冲，而不是来自可观测的跨 tile 流水。若把它误写成“已
经完成双缓冲并获得流水收益”，会高估方法效果。`t=2,d=1` 在这个版本中较慢，
但尚不能归因于计算不足或资源开销：后续复核发现扩展为 `d+1` 输入槽后，仍沿用
`wait_group(d-1)`，会额外等待下一 tile。修正为 `wait_group(d)` 才与声明的预取
距离一致。旧测量保留为历史结果，不代表修正后异步供数的性能上限。

为排除 `[10,10]` 形状特例，重新扫描 t1d1 邻域。所有结果同样是 A100 80GB、FP64、
N=2^20、B=16、正向、Graph 中位延迟：

| factor partition / columns | `t=1,d=0` | `t=1,d=1` | cuFFT |
| --- | ---: | ---: | ---: |
| `[9,11]` / 1 | 1.6167 | 1.6139 | 0.8941 |
| `[10,10]` / 1 | 1.7183 | 1.7129 | 0.8941 |
| `[10,10]` / 2 | **0.9091** | 0.9284 | 0.8941 |
| `[11,9]` / 1 | 1.4881 | **1.4628** | 0.8941 |

因此 t1d1 的强点是一个合法且接近 cuFFT 的局部形状，但不能把它解释成流水收益，
也不能外推为所有 factor 形状都强。资源模型现在按同一语义计算：`d=0` 使用一个
tile/work 槽，`d>0,t>1` 使用 `d+1` 个输入槽加一个工作槽，即 `(d+2)×tile`
的必要下界；`t=1,d>0` 使用单槽下界。

主控组原始结果保存在
`results/factor_graph_20261025/core_prefetch_gpu1_true_double_buffer_v3/release/results.json`，
t1d1 形状扫描保存在
`results/factor_graph_20261025/t1d1_scan_gpu1_v2/release/results.json`。复现实验配置分别为
`configs/experiments/factor_graph_core_prefetch_20261025.json` 和
`configs/experiments/factor_graph_t1d1_scan_20261025.json`。

固定供数后继续扫描局部核心形状，仍是 `t=1` 更好。`ElementsPerThread=8,
FFTsPerBlock=2` 的点为 0.9101 ms，约为 cuFFT 延迟的 1.0156 倍，另一次独立邻域
运行是 0.9162 ms，约为 1.0078 倍。`FFTsPerBlock=1`、`ElementsPerThread=4`
或把 columns 调到 1、4、8 都更差；邻近的 factor partition 也没有稳定地超过
`factor-10+10`。这些完整实现的延迟接近 cuFFT，但不能由此隔离局部核心效率。
阶段边界、全局中间数据、供数等待和批量调度均是待区分的因素，当前数据不足以
定量分摊剩余差距。

## FFT-H 与 FFT-L 不能混为同一结论

FFT-H 指 FP32/FP64。当前 `factor-streamed` 和上述受控实验属于这一路径。FFT-L
指 FP16/BF16，长一维因子路径目前没有与 FFT-H 等价的 provider。低精度的通用 C2C
路径在 `LowFftOperator` 中使用低精度存储并可能提升累加，WMMA DFT8 只覆盖 FP16--
FP32、N=8 的局部例子；它不能代表长 N 的 FFT-L，也不能用 FFT-H 的 cuFFTDx 结果
替代。因此目前不能说“FFTL 和 FFTH 都同样惨”：FFT-H 已有接近 cuFFT 的局部点，
FFT-L 的主要问题首先是缺少可审计的长 N 局部 provider 和对应校准数据，性能结论
仍不完整。

## 当时列出的待办

1. **先修正统一 selector 的证据链。** `ept=8, columns=2, t=1, depth=0` 已经
   在公共设计空间内，但静态 cost model 目前对不同核心形状使用过于相同的硬件率
   先验。应把本轮按核心形状采集的 service samples 写入 A100 安装校准文件，重新
   排序并验证 AUTO 在 N=2^20、B=4/16/64 以及正逆方向都能选到该邻域。只有这一步
   通过，才能判断“统一搜索没有找到”是否是框架问题。
   本轮同时修复了旧校准产物中 canonical mapping 缺少或显式为空的
   `factor_ept/factor_columns/data_tiles_per_cta/prefetch_depth` 键合并问题，并以
   30 个 CPU 回归用例确认不同核心形状不会再共享错误的服务曲线。这个修复避免了
   错误合并，但不会凭空生成尚未采集的 A100 N=2^20 FP64 服务样本。
2. **再处理阶段物理边界。** 在不把整个 N 搬进片上的前提下，增加 provider-neutral
   的 resident exchange/fused factor lowering：只有当相邻 tile 的依赖闭包、shared
   容量、寄存器和并发约束同时满足时，才合并中间交换；否则保留合法的 HBM 换入换出
   路径。这样验证的是混合数据流的落地质量，而不是强行全驻留。
3. **为 FFT-L 增加独立 provider。** 评估可审计的 FP16/BF16 局部核心（包括 WMMA
   或其他公开实现），明确存储精度、乘法精度和累加精度，再把它接入同一设计空间、
   校准和搜索接口。没有这个 provider 前，不能用 FFT-H 的结果宣称低精度长 N 已被
   支持。

这些待办仍有工程价值，但当前数据并未证明它们就是全部性能差距的根因，也没有
证实理论最优性。后续优先级按更新后的研究原则调整为固定核心的组织收益与跨核心
复用，保留外部库竞争力对照。等待指令修正后的受控实验单独报告；不把 cuFFTDx
内部算法误写成已知 radix，也不把 FFT-H 结果外推到 FFT-L。
