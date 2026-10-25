# 长一维 FFT 的阶段内流水核查

2026-10-24。本轮属于 FFT-H（FP32/FP64）。问题是大阶段内部能否持续供给独立子图，并提高完整长一维
调用的效率。本页保留大阶段之间的合法依赖同步，不把跨大阶段提前执行
当作混合数据流的必要条件。当前状态为代码核查和两个 register 设计点的
NCU 完成，4 个请求的 CPU 参考与 20 个模块已准备。启动受控测量时目标
GPU 出现其他任务，独占检查在任何候选测量前拒绝启动；尚无本轮 F 组性能结论。

## 方法与实现的对应

APPT26 的 `Efficient Dataflow` 与 `Conflict-Free On-Chip Fragmentation
Algorithm` 将独立 tile 连续送入计算单元，并以写回映射适配后续读取。
阶段内流水、大阶段间同步、布局对齐分别负责持续供给、正确依赖和访存
效率。GPU 可以在一次 worker launch 内连续处理子图任务；每个 tile
不必成为一次 CUDA launch。任务或代码可复用与实际 CTA 驻留仍需区分。

此前讨论把下一大阶段的 fan-in 等待当作长一维主要问题，归因层次不准确。
这只能解释为何不能任意提前下一大阶段，不能否定当前大阶段内部的流水。
同理，`stage_overlap=false` 不能证明阶段内没有流水，多个 CUDA stream
的时间重叠也不能单独证明论文所指的阶段内组织已经实现。

| 层次 | 当前 register-prefix/grouped-suffix 路径 | 证据与边界 |
| --- | --- | --- |
| 子图内驻留 | 已有寄存器与 shared 状态复用 | 每个 CTA 处理固定的一组局部子变换，内部多步运算不逐步回 HBM |
| 连续处理 tile | 每 CTA 一组，完成后退出 | `fft_register_tile.cuh` 与 `fft_grouped_suffix.cuh` 以 `blockIdx.x` 固定变换编号，没有后续 tile 循环 |
| 阶段内预取 | wrapper 没有显式下一组预取 | 当前工作是 load、compute、store；硬件 warp/CTA 调度仍可能隐藏延迟 |
| 消费者布局 | 已直接写入 | prefix 写 `scratch[k1*Q+n2]`，suffix 按固定 `k1` 读取连续 `n2`，无独立中间 transpose |
| 最终布局 | shared ownership exchange 后直接写回 | suffix writer-aligned 交换恢复自然输出，无独立最终 transpose |
| 外部 batch 流水 | 可选，但本轮关闭 | 不等同于 N 内独立子图的阶段内流水 |

现有 `factor-streamed` 的 `data_tiles_per_cta` 和 `prefetch_depth` 已支持
同 CTA 连续消费 tile 及 `cp.async` 输入预取。它在读取当前输入到寄存器
后提交未来 tile 的输入拷贝，再执行局部 FFT。输出仍经 work shared 写回
并同步，不能称为已实现独立 store worker 或完整三段异步流水。该模板
与本轮 register-prefix 不同，比较二者时须保留核心/组织差异。

目前的静态核查不能证明缺少预取就是主要瓶颈。显式缓冲会增加 shared
需求；连续处理多个 tile 会减少 launch CTA 数。应同时检查供数等待、
发射能力、可驻留资源、实际访存事务和完整调用时间。

## 矩阵化核心的作用

多个独立局部变换可以作为矩阵的不同列，由小 DFT 矩阵一次作用于它们。
这些列可以来自单个 N 内部的子图，不要求只增加外部 batch。矩阵化因而
是合法的局部核心选择，可以改变计算吞吐和线程协作方式；它不会自动增加
独立 CTA 数，也不自动完成跨 tile 的 load/compute/store 调度。

[tcFFT](https://github.com/rox906/tcFFT) 提供 Tensor Core FFT 的 half 核心，
作者说明在 V100/A100 上测试过。
[FlashFFTConv](https://github.com/HazyResearch/flash-fft-conv) 提供面向长卷积的
FFT Tensor Core 实现，其公开接口支持 FP16/BF16，不能将这些性能直接用作
当前 FP32/FP64 C2C 的同精度证据。两者列入 FFT-L 的对应角色，适配状态
与可比较语义由 benchmark 对照目录单列。

仓库已有 `wmma-dft8`，但当前合同限定 `fp16-fp32` 与 8 点完整变换，见
`src/wmma_fft.cuh` 和 `src/butterfly.cu`。本轮长一维 register 路径没有
消费该核心。矩阵化小核心接入长一维、保持用户精度、验证有效算术/填充
开销，是独立的后续工作，不能通过偷偷改用低精度来替代。

A100 也提供原生 FP64 Tensor Core DMMA，因此矩阵化并非仅限 FFT-L。
FP64 局部矩阵核心可作为 FFT-H 候选，但当前没有该长一维实现和实测结果。
TF32 乘法不能标成完整 FP32 算术，见
[NVIDIA Ampere tuning guide](https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html)。

## 本轮受控问题与范围

硬件使用 A100 80GB PCIe，UUID
`GPU-0eae5f07-33de-321b-70a9-ffeae49309e6`。固定 forward C2C、自然顺序、
连续布局、out-of-place、B=16，取 N=2^16/2^20 与 FP32/FP64，共 4 请求。
CPU 同精度参考、输入和编译先完成，GPU 仅构造、验证和测量；CUDA Graph
计时排除编译及初始化。误差仍 report-only，不新设或放宽数值阈值。

| 组别 | 大阶段 / tile | 每 CTA 子图数 | 每 CTA 连续组 / 预取深度 | 允许的归因 |
| --- | --- | --- | --- | --- |
| R | `[6,10]` / `[64,1024]` 或 `[10,10]` / `[1024,1024]` | `[4,4]` | `1 / 无` | 当前 register-radix4 + cuFFTDx 完整路径对照 |
| F1 | 同上 | `[4,4]` | `1 / 0` | factor-streamed 的 cuFFTDx 基础组织 |
| F4 | 同上 | `[4,4]` | `4 / 0` | 相对 F1，连续消费与 CTA 分组变化 |
| F4P1 | 同上 | `[4,4]` | `4 / 1` | 相对 F4，输入预取及其缓冲成本 |
| F4P2 | 同上 | `[4,4]` | `4 / 2` | 相对 F4P1，预取深度与缓冲需求变化 |
| cuFFT | 外部内部结构未知 | 未知 | 未知 | 同语义完整变换对照 |

F 组在同一请求内固定 cuFFTDx 核心、EPT、列数=4、布局、twiddle 和分段。
N=2^16 使用 EPT=8，线程为 `[32,512]`；N=2^20 使用 EPT=16，线程为
`[256,256]`。前者若用 EPT=16，64 点阶段只有 16 线程，会被现有合法性
检查拒绝，因此显式选用 EPT=8，不改变请求尺寸或精度。保留大阶段同步，
关闭跨 batch/因子切片 overlap。R 与 F 的差异不能只归因为预取。
所有实际 mapping、线程配置、launch CTA 数、误差、每次 trial 和重放校验
保存在实验记录中；概要遵循 [报告合同](reporting_contract.md#设计点概要)。

NCU 关注 prefix/suffix 各自的 registers、shared、active/eligible warps、
long/short scoreboard、barrier 和 global/shared 事务。profile 与普通完整
调用时间分开。若寄存器、缓冲或总 CTA 数改变，连同设计点一起解释，不用
单一 occupancy 数字判断机制成败。若 F 组预取有效，再在 register 核心
保持不变的条件下接入相应组织；若无效，先定位受限资源，不直接追加大矩阵。

## 当前 register 设计点的 NCU

本节为本轮实际采集，不沿用旧版本计数。两点均为 forward C2C、B=16、
prefix radix4、suffix cuFFTDx、每 CTA 子图数 `[4,4]`、连续组 `[1,1]`，
prefix 内部布局 linear，suffix writer-aligned，阶段间 global-scratch，
无跨 batch overlap。完整参数见 [核心替换报告](radix4_completion_20261024.md)。

| 请求 | 大阶段 / tile | 线程 / EPT | launch CTA |
| --- | --- | --- | --- |
| FP32，N=2^16 | `[6,10]` / `[64,1024]` | `[32,256]` / `[8,16]` | `[4096,256]` |
| FP64，N=2^20 | `[10,10]` / `[1024,1024]` | `[128,256]` / `[32,16]` | `[4096,4096]` |

| 指标 | FP32 前段 | FP32 后段 | FP64 前段 | FP64 后段 |
| --- | ---: | ---: | ---: | ---: |
| registers / thread | 60 | 48 | 254 | 102 |
| NCU shared bytes / CTA | 3072 | 33792 | 66560 | 66560 |
| active warps / 峰值，% | 29.74 | 26.88 | 12.21 | 24.50 |
| eligible warps / scheduler cycle | 2.01 | 0.83 | 0.41 | 0.40 |
| long-scoreboard / active warp，% | 3.03 | 9.70 | 19.63 | 19.08 |
| short-scoreboard / active warp，% | 2.88 | 3.53 | 2.85 | 7.50 |
| barrier / active warp，% | 0.90 | 6.92 | 2.68 | 11.82 |
| SM throughput，% | 58.07 | 29.23 | 77.10 | 38.31 |
| DRAM throughput，% | 0.33 | 23.14 | 63.00 | 75.44 |

NCU shared 字段包含该次 launch 的完整报告值，不等同于用户动态 shared
参数。stall 百分比是每活跃 warp 的指标，不能直接解释为完整调用的时间
占比。SM throughput 是吞吐汇总指标，不能直接称为 FFT FLOP 峰值利用率。

FP64 前段的 12.21% 活跃度与约 255 寄存器/线程的静态驻留上限一致，但其
SM 与 DRAM 已分别达到 77.10% 和 63.00%；缺少活跃 warp 并不意味着算术
资源完全闲置。后段 DRAM 达到 75.44%，增加缓冲还需要考虑已有流量和
共享资源。FP32 小点前段的 long-scoreboard 较低，不能将它的差距全部
归为供数不足；后段 256 个 CTA 及较低 eligible warps 提示需要检查组织
粒度与核心服务效率。显式预取是否有效由后续固定核心对照决定。

原始文件为 [FP32 N16](../results/fft_intra_stage_20261024/profile/register_n16_fp32.csv)
和 [FP64 N20](../results/fft_intra_stage_20261024/profile/register_n20_fp64.csv)，
对应 `.ncu-rep` 同目录。使用 public plan bench，warmup=2、repeat=1，
过滤 register/grouped-suffix，跳过 4 次匹配 launch 后采集 2 次。
NCU kernel replay、cache-control=none、clock-control=none；未锁时钟的
profile 用于归因，不代替普通 Graph 稳态计时。共享 bank conflict 原始
计数也已保留，缺少事务归一化时不据绝对次数断言布局仍有严重冲突。

## 工件与恢复

新入口是 [run_fft_intra_stage_experiment.py](../scripts/run_fft_intra_stage_experiment.py)，
其准备与恢复测试共 8 项通过。它保存实际设计点、同精度误差、逐 provider
trial 和精确 JSON 重放结果。准备完成后只查找已经准备并校验 hash 的模块，
运行阶段遇到新编译请求会明确失败。每个 provider 单独保存一次完整误差
工件，避免反复压缩写入累计结果。

准备及启动记录见
[measurement/summary.json](../results/fft_intra_stage_20261024/measurement/summary.json)。
当前 `status=failed` 的原因是设备忙；`preparation_complete=true`，候选
测量记录为空。保留失败原始记录，使用下面的命令可继续已准备工件，不
重新生成参考或编译模块。GPU 必须独占，identity 不匹配时不会覆盖旧工件。

```bash
/home/wt/yes/envs/cubutterfly/bin/python scripts/run_fft_intra_stage_experiment.py \
  --library /home/wt/git/cuButterfly/build-e13-cufftdx/libcuntt.so \
  --gpu GPU-0eae5f07-33de-321b-70a9-ffeae49309e6 \
  --output results/fft_intra_stage_20261024/measurement \
  --nvcc /usr/local/cuda-12.4/bin/nvcc \
  --mathdx /home/wt/git/cuButterfly/external/mathdx/nvidia/mathdx --resume
```

NCU 2024.1.0 使用的库 SHA256 为
`887e296b52f48602b95205dea252a3271070f045206187805a9275d80176ce6e`，
plan bench SHA256 为
`909f010a5e2920cb15230b75668c1030bc3cc119a4b7ae8b1306226b14a84d69`。
本轮未修改 kernel 或重新构建库。
