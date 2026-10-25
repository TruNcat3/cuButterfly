# 历史实验观察

本文档整理可用于论文修订的历史实验观察。它们属于各自记录中注明的
V100/A100 cohort 和构建，不是当前版本的统一性能矩阵。这里的“收益”只表示
指定负载、映射和计时协议下的配对观测；不表示首创、全设计空间最优或未经核验
的普遍因果关系。

正文动机采用的记录冻结在
[historical_motivation.json](evidence/historical_motivation.json)，包含时间、控制条件和
七份源文件 SHA-256；独立于原有综合比较表的冻结清单。

## 1. A100：register-c16 替换为 dx-thread-c16

来源：[register-tile 报告](../results/fft_register_tile_20260912/report.md)，
[accuracy-corrected paired 摘要](../results/fft_register_tile_20260912/accurate_core_control/paired_summary.json)，
[试验元数据](../results/fft_register_tile_20260912/accurate_core_control/paired_metadata.json)。
报告的核心控制在 `report.md:51-66`，摘要的中位数在
`paired_summary.json:3-43`。

- **平台和负载。** A100 80GB PCIe；FP32 complex forward FFT，natural output、
  out-of-place，`batch=4`，`N=2^18` 和 `N=2^20`。同一进程使用相同输入，构造、
  分配、正确性检查和 warmup 不计入 CUDA-event 时间（报告 `:5-10`）。
- **替换和保持项。** `register-c16` 的 native prefix codelet（编译期 butterfly
  index/root）替换为 `dx-thread-c16` 的 cuFFTDx Thread codelet。报告明确 prefix
  columns=16，4096-point suffix 和 transpose path 相同（`:60-64`）；因此这是
  register-tile 路径中的算术核心替换观察，数据所有权和已声明 schedule 保持不变。
- **时间。** `N=2^18`：native `0.035031 ms` -> dx-thread `0.0336486 ms`，
  `1.041x`（约 3.95% faster）；`N=2^20`：`0.148951 ms` -> `0.145633 ms`，
  `1.023x`（约 2.23% faster）。每个 cell 为 5 个 paired rounds、每轮 100 次，
  相对误差和原 absolute-error 检查均通过（报告 `:60-65`，元数据 `:4-9`）。
- **证据边界。** 这是 accuracy-corrected paired core-control，不是新的完整
  factorial validation。端到端时间没有隔离 prefix 算术本身；不同 codelet 仍可能
  改变寄存器分配、occupancy 及与固定边界的交互。元数据标记 `layout_sweep=false`
  但没有枚举所有 mapping 字段，所以应称为“相同已声明 schedule/path 下的核心
  替换观察”。该历史记录也不替代当前版本的跨负载统一矩阵。

## 2. V100：固定局部核心下 single 组织与 dual 组织

来源：[dual warp-row-group 报告](../results/homogeneous_warp_dual_groups/analysis.md)，
其中控制条件和结果见 `analysis.md:3-20`，计数器诊断见 `:32-44`。

- **平台和负载。** Tesla V100；uint32 Shoup NTT，`logN=20`，`10+10` factorization，
  four-row static IO。
- **组织变化和保持项。** single 使用一个四-warp row group、128-thread CTA、
  4 CTAs/SM；dual 在 256-thread CTA 中复制两个独立同步的四-warp row subgraph、
  2 CTAs/SM。两者都暴露 16 个 resident warps，且使用相同 register-resident
  stage-7/8/9 continuation；两者均为 101 registers/thread、16-byte stack
  （报告 `:3-11`）。因此这里改变的是 subgraph replication/CTA 组织，不是局部
  butterfly codelet；但 CTA 形状、shared-memory 用量（约 20.6 -> 41.2 KiB）和
  物理驻留数随组织一起变化。
- **批量交叉。** 三次确认试验、5 warmups、30 repetitions，报告值为 trial
  medians（`:13-20`）：B=1 single/dual 均 `0.177 ms`；B=4 为
  `0.426 -> 0.470 ms`，dual/single `0.906x`，明确为负例；B=16 为
  `1.506 -> 1.305 ms`，`1.154x`，为正向观察。
- **证据边界。** 计数器中 dual/single 的 global-load sectors 和 warp instructions
  基本相同，而 long-scoreboard stall 为 `35.59% -> 28.58%`（`:32-38`），与
  独立 row subgraph 的 issue interleaving 解释相容，但仍是诊断支持而非形式因果
  证明。该记录是同一局部核心下的 paired organization screen，没有提供普通/优化
  核心与基础/混合组织的完整四格 factorial；B=4 的负结果必须保留。

## 3. V100：barrier 到 resident 10+10 的 batch 交叉

来源：[综合分析](../results/hierarchical_streaming/comprehensive/analysis.md)，
[原始摘要](../results/hierarchical_streaming/comprehensive/summary.csv)，以及
[wave-core checkpoint](../results/hierarchical_streaming/v07_wave_core_checkpoint.md)。
控制和交叉表见 `analysis.md:3-39`、`summary.csv:1-40`；同一成熟局部核心的说明见
`v07_wave_core_checkpoint.md:20-23`。

- **平台和负载。** Tesla V100-SXM2-16GB，`logN=20`，uint32/uint64 NTT，
  natural output；综合记录为 3 次 trial 的独立进程中位数，CUDA-event 使用相同
  V100 binaries/settings（`analysis.md:3`；原始记录为 10 warmups、50 repetitions）。
- **组织变化和保持项。** v0.6 barrier 与 v0.7 readiness-driven resident `10+10`
  比较；`v07_wave_core_checkpoint.md:20-23` 明确 resident `10+10` 使用与 v0.6
  相同的成熟 local radix-4 implementation，变化在 fixed-role partitioning、
  readiness publication 和 pipeline fill。因此这是数据流/执行组织变化的观察，
  不是算术核心替换；CTA/驻留和同步路径仍随组织变化，不能视为完全相同的物理
  mapping。
- **批量交叉。** uint32 的 barrier -> resident `10+10` 为：B=1
  `0.080179 -> 0.129372 ms`（`0.620x`），B=4 `0.271708 -> 0.347505 ms`
  （`0.782x`），B=8 `0.501760 -> 0.546652 ms`（`0.918x`），B=16
  `1.058591 -> 1.020949 ms`（`1.037x`），B=32 `2.852577 -> 2.398044 ms`
  （`1.190x`）。uint64 也在较大 batch 侧转正：B=4 `0.503357 -> 0.555151 ms`
  （`0.907x`），B=16 `1.895014 -> 1.840476 ms`（`1.030x`），B=32
  `3.712512 -> 3.623834 ms`（`1.024x`）。
- **证据边界。** 这是跨 batch 的 paired dataflow observation：小 batch 的负结果、
  大 batch 的正结果共同构成交叉点；trial spread 最高为 uint32 `16.99%`、uint64
  `13.02%`（`analysis.md:21,39`）。它不能单独证明 readiness 或 pipeline fill
  的纯因果收益，也不替代同一 mapping 下的统一 factorial。

## 当前矩阵状态与历史证据的关系

当前 A100 80GB 综合运行的 journal 只有 `5/72` cells 完成，并因目标 GPU 被另一
进程占用而中断；该状态见 [framework refactor progress](../docs/framework_refactor_progress.md)
的 `:12-22`。因此完整矩阵、完整迁移或跨 cohort 的统一统计仍为 pending。GPU 忙
只说明当前统一验证尚未完成，不会把上述已完成、可追溯的历史 paired observations
变成“没有证据”；同时这些历史观察也不能填补当前矩阵缺失的 cells。

补充反例：V100 vector-radix4 的事件时间领先在固定时钟 NCU 中反转（`1.238446 ms`
事件值；`1448.032 us` 对 reuse d4 的 `1409.856 us`），所以该记录只能作为
“固定时钟下未确认的核心替换正向结果”，不作为主要正例。来源为
[vector-radix4 analysis](../results/homogeneous_vector_radix4/analysis.md)，其中的
事件和固定时钟结论见 `analysis.md:19-44`。
