# FFT suffix exchange audit

日期：2026-10-23。本文记录一个固定映射的局部实验，不代表自动搜索结果或
全量 FFT 横评。实验使用 A100 80GB PCIe，CUDA 12.4，FP32，`N=65536`、
`batch=16`、前缀/后缀分段 `6+10`。前缀均为 native register-tile，128
threads、EPT 8，后缀均为 cuFFTDx、512 threads、EPT 8，所有输入输出和
边界语义一致。只改变后缀的 `exchange_chunks[1]`，并保留 `0` 作为整块写回
控制。计划、库和验证输入的身份保存在
[`results/fft_suffix_exchange_20261023`](../results/fft_suffix_exchange_20261023)。

## 结果

CUDA Graph 捕获 20 次执行，预热 5 轮，每轮重放 200 次，报告五轮中位数。
计划创建、编译、分配和捕获不计入执行时间。每个结果都通过同精度 CPU
参考和 Graph 与 eager 输出一致性检查。

| 设计点 | 后缀交换块 | 中位执行时间 | 相对 cuFFT | 相对整块写回 |
| --- | ---: | ---: | ---: | ---: |
| cuFFT | — | 21.303 μs | 1.000x | — |
| cuButterfly | 0 | 27.048 μs | 0.788x | 1.000x |
| cuButterfly | 1 | 26.670 μs | 0.799x | 1.014x |
| cuButterfly | 2 | **26.637 μs** | 0.800x | **1.015x** |
| cuButterfly | 4 | 27.065 μs | 0.787x | 0.999x |
| cuButterfly | 8 | 27.387 μs | 0.778x | 0.988x |

这里的相对值是吞吐比例，即 `cuFFT latency / cuButterfly latency`。分块交换在
块大小 1 和 2 时减少约 1.4% 至 1.5% 的延迟；更大的块重新增加等待和共享
内存工作。这个结果支持把 `exchange_chunks=1/2` 留在候选空间，但不足以把
FFT 后缀差距归因于单一交换策略，也不足以宣称 FFT 已经追平 cuFFT。

## NCU 证据

NCU 只采集后缀 kernel，因此不能替代上面的端到端两组 kernel 时间。三组使用
相同的 256 CTAs、512 threads/CTA、32 registers/thread 和 0.59 waves/SM。

| 后缀 | Kernel 时间 | 动态共享内存 | 实际占用 | DRAM 吞吐 | L1/TEX 命中率 | SM 吞吐 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 19.87 μs | 32.77 KiB | 52.15% | 22.85% | 6.54% | 30.65% |
| 2 | 19.58 μs | 16.38 KiB | 52.29% | 23.12% | 6.55% | 25.84% |
| 8 | 20.03 μs | 32.77 KiB | 51.34% | 22.68% | 6.59% | 24.86% |

理论占用率均为 100%，实际活跃占用约 51% 至 52%，而 DRAM 和 SM 吞吐都远未
饱和。这排除了“只要继续增加 CTA 就能解决”的解释。块大小 2 的微小收益来自
写回路径与同步次数的平衡；块大小 8 增加了共享内存交换量，却没有提高活跃
并行度。后续应优先检查后缀的指令和数据依赖组织、前后缀之间的全局 scratch
边界，以及 cuFFTDx 的 block 形状，而不是继续盲目扩大交换块。

## 与 NTT 机制统一的边界

这个实验使用的是现有 `grouped_suffix` 的参数化交换路径，没有把 NTT 的
packet/ring 或模乘核心硬塞进 FFT。NTT 审查见
[`ntt_common_mechanisms_20261023.md`](ntt_common_mechanisms_20261023.md)：
公共 stage、驻留叶子和批次流水已经真正共享，内核内 role、publication 和
coefficient-supply 仍需先建立完整 descriptor，才能安全地成为跨算子候选。
因此本实验的负结果不说明混合数据流失效，而是说明当前 FFT 后缀 lowering
仍没有消费足够多的通用组织能力。
