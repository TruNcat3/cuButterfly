# 公共约束与静态排序回归

日期：2026-10-24。范围是 rank-one 公共约束、编译资源反馈和 registry
未命中时的静态排序。这里不声明完成全理论设计空间、全算子数值域或 xD 的
统一成本建模。设计合同见 [静态资料决策](adr/0001-static-profile-driven-planning.md)。

## 实现与验证口径

- Butterfly/NTT 生产构造调用共同 legality gate，逐物理执行组消费编译资源。
  默认 shared 分段按 local/flow stage 展开，不把整个 N 当成一个片上 tile。
- registry 未命中时先投影并排名，再构造有限 finalists，按实际资源复核后
  重排。已知不可行的候选不参与回退；复核全部拒绝时从剩余候选补位。
- 拟合模型、硬件速率先验和无测量回退分别报告。本轮显式使用仓库 A100
  80GB 容量/历史速率种子，没有将旧 warning 模型改标为已校准。
- structured 矩阵语义沿用校准 CSV 的编码；benchmark 的选择说明改用公共
  CSV 转义函数，避免说明中的 JSON 引号破坏整行。原始解析失败记录保留。
- 排名助手只读复用已准备的 native CPU 调度库，缺失时使用同算法 Python。
  编译、候选构造、CPU 排名和参考计算均不计入 kernel 时间。

硬件为两张 NVIDIA A100 80GB PCIe，容量 84974239744 字节。
公共矩阵使用 `GPU-0eae5f07-33de-321b-70a9-ffeae49309e6`；缓冲实验和
C++ 公共 API/Graph 回归使用 `GPU-2971e500-d3f6-d7e6-e5f2-cb67a06fff02`。
仅显式允许纯 `G` 类型的 `/usr/lib/xorg/Xorg` 常驻进程；其他进程会阻止或
中断测试。计时是相同 bench 的 CUDA event direct-launch，warmup 5、repeat 20。
除单列的三轮预取对照外，性能表每点为一次进程运行内 20 次调用的平均值，
尚未构成多轮置信区间。正确性专用条目使用 warmup 1、repeat 1。

本轮核心库 runtime fingerprint 为
`4f09dacb5171817a5449ca9ef476f29b435728051ed386560bb9e07f2849a5e5:research`。
后续修复只改 benchmark 的 CSV 转义，核心库 SHA256 保持不变。复用的原始
记录保留各自 benchmark 哈希，不将不同二进制混写成一个版本。

## 缓冲粒度与约束

以下均为 FP64、B=4、两物理阶段、cuFFTDx-block 核心、EPT=8，CTA 内顺序
处理四组任务。这里的 `columns` 是同时处理的局部 FFT 数；`prefetch_depth=2`
表示两个输入槽，另外还需要工作区，不能只计算两个槽的内存。

| N | 每阶段局部 FFT / columns / depth | 每阶段 CTA 数 / threads | 每组实际 shared | kernel 时间 | 验证 |
| --- | --- | --- | --- | ---: | --- |
| 2²⁰ | 1024 / 2 / 2 | 512 / 256 | 96 KiB | 0.333210 ms | 四个 batch 全部通过 |
| 2²⁰ | 1024 / 4 / 0 | 256 / 512 | 64 KiB | 0.357734 ms | 四个 batch 全部通过 |
| 2¹⁸ | 512 / 4 / 2 | 128 / 256 | 96 KiB | 0.101888 ms | 四个 batch 全部通过 |
| 2²⁰ | 1024 / 4 / 2 | CPU 容量检查 | 至少 192 KiB | 不启动 | 超过每 CTA 上限，正确拒绝 |

前两行同时改变了 columns 和线程数，不能据此单独归因于预取。另做固定
N=2²⁰、B=4、columns=2、EPT=8、threads=256 的三轮交替对照，仅改变 depth。

| depth | 三轮 kernel 时间（ms） | 中位数（ms） |
| ---: | --- | ---: |
| 0 | 0.482099、0.482099、0.482406 | 0.482099 |
| 2 | 0.333722、0.333056、0.333107 | 0.333107 |

此设计点的预取提升为 1.4473×。这支持 tile 级预取的实际收益，不能推广为
所有负载均有相同比例收益，也不等于相对 cuFFT 的加速比。浮点 bench 的
`max_error` 仅保留六位小数，显示 `0.000000` 不代表逐位精确；未放宽原验收。

同一张卡、相同 FP64/B=4/输入与 direct-launch 口径的 cuFFT 对照如下。
加速比定义为 `cuFFT 时间 / 当前设计点时间`，小于一表示仍落后。

| N | 当前设计点 | 当前时间（ms） | cuFFT（ms） | 相对 cuFFT |
| --- | --- | ---: | ---: | ---: |
| 2²⁰ | 2×1024、depth=0（三轮中位数） | 0.482099 | 0.270950 | 0.562× |
| 2²⁰ | 2×1024、depth=2（三轮中位数） | 0.333107 | 0.270950 | 0.813× |
| 2²⁰ | 4×1024、depth=0 | 0.357734 | 0.270950 | 0.757× |
| 2¹⁸ | 4×512、depth=2 | 0.101888 | 0.062566 | 0.614× |

所以缓冲粒度问题得到实测验证，但这些指定设计点尚未超过 cuFFT，不能
把局部预取收益写成整体 FFT 已领先。该表不代表安装后测量搜索得到的最优点。

## 公共回归

C++ CPU 集成的 mixed_dataflow、hardware_model 和 static_ranking 三组通过。
GPU 普通 shared 机制、shared CUDA Graph 与 public planner/replay 三组通过。
Graph 测试包含正逆方向、多个 batch tile、FP32/FP64 和整数路径，共 88 个
合同检查；不把这些小型合同检查当成中大型性能结论。

算子矩阵声明 27 个正确性组合和 7 个性能点，覆盖 FFT FP32/FP64、FWHT、
subset/superset/xor、structured-2x2、NTT word32/word64。其中 5 个正确性组合
与性能点的命令仅 warmup/repeat 不同，由后者的全部 batch 验证覆盖，实际
无需重复执行 34 次搜索。命令等价检查、复用记录与别名保存在
`execution_manifest.json`。34 个条目全部得到覆盖，共 29 次独立执行；
5 个别名条目不重复计数。逐条来源、命令等价检查和哈希核对保存在
`validated_matrix.json`。早期两条 CSV 解析失败和一条手动停止记录仍保留，
对应条目由后来完成的同形状验证覆盖，没有删除失败历史。等待器完成任务后正常退出。

下表均为 B=16，展示空 registry 下的公共 AUTO 路径。这里有意使用硬件
速率先验检查新排序链路，没有加载既有调优赢家，不能用它替换论文或版本
综测结果。FFT 对照使用同一张 GPU、同精度、输入与计时协议，全部 16 个
batch 验证通过；此处 FP32/FP64 均属于 `fft-h`，未验证 `fft-l`。其余三点
本轮没有重跑外部对照。

| 算子 / 精度 | N | 本次 AUTO（ms） | cuFFT（ms） | cuFFT / AUTO |
| --- | --- | ---: | ---: | ---: |
| FFT FP32 | 2¹⁶ | 0.045312 | 0.023592 | 0.521× |
| FFT FP32 | 2²⁰ | 0.581274 | 0.383214 | 0.659× |
| FFT FP64 | 2¹⁶ | 0.073370 | 0.048384 | 0.659× |
| FFT FP64 | 2²⁰ | 2.581094 | 0.895488 | 0.347× |
| FWHT FP32 | 2¹⁶ | 0.061594 | — | — |
| NTT word32 | 2¹⁶ | 0.098611 | — | — |
| NTT word64 | 2¹⁶ | 0.117658 | — | — |

以下概要来自实际 execution groups。`a+b` 是逻辑阶段的分段；tile 是该
局部变换的点数，不等同于整个 CTA 处理的数据量。CTA 数表示 launch grid，
不表示同时驻留的 CTA 数。所有点均无显式预取和跨阶段 overlap。

| 点 | 分段 / tile | 每阶段 CTA 数 | 每阶段 threads | 核心 |
| --- | --- | --- | --- | --- |
| FFT FP32 / 2¹⁶ | 8+8 / 256、256 | 4096、4096 | 128、128 | cuFFTDx-block、cuFFTDx-block |
| FFT FP32 / 2²⁰ | 8+12 / 256、4096 | 16384、2048 | 64、512 | register-tile、cuFFTDx-block |
| FFT FP64 / 2¹⁶ | 7+9 / 128、512 | 8192、2048 | 128、128 | cuFFTDx-block、cuFFTDx-block |
| FFT FP64 / 2²⁰ | 10+10 / 1024、1024 | 16384、16384 | 128、128 | scalar、scalar |
| FWHT FP32 / 2¹⁶ | 9+7 / 512、128 | 2048、8192 | 128、128 | scalar、scalar |
| NTT word32 / 2¹⁶ | 8+8 / 256、256 | 4096、4096 | 128、128 | radix2-shoup、radix2-shoup |
| NTT word64 / 2¹⁶ | 8+8 / 256、256 | 4096、4096 | 128、128 | radix2-shoup、radix2-shoup |

## 排序分辨率的实测缺口

对最弱的 FFT FP64、N=2²⁰、B=16，追加三个已实现的合法 factor-streamed
设计点，全部通过完整验证。保持两阶段 10+10、cuFFTDx-block、EPT=8、每 CTA
顺序处理四组任务、recurrence twiddle 和 writer-aligned 布局。数据直接来自
公共 benchmark，编译与构造不计时。这里没有扩大搜索矩阵或发布新赢家。

| 实际设计点 | 每阶段 CTA / threads | 每组 shared | 静态先验（ms） | 实测（ms） | cuFFT / 当前点 |
| --- | --- | --- | ---: | ---: | ---: |
| AUTO：scalar、无预取 | 16384 / 128 | 16 KiB，编译资源不完整 | 0.727715 | 2.581094 | 0.347× |
| cuFFTDx：2×1024、depth=0 | 2048 / 256 | 32 KiB | 0.727715 | 1.818675 | 0.492× |
| cuFFTDx：2×1024、depth=2 | 2048 / 256 | 96 KiB | 0.727715 | 1.153792 | 0.776× |
| cuFFTDx：4×1024、depth=0 | 1024 / 512 | 64 KiB | 0.727715 | 1.124045 | 0.797× |

对同一批实际 execution groups 重新调用公共静态排名，四点的分数完全相同。
现有硬件速率种子只给出聚合吞吐，无法分辨这些核心、布局与缓冲组合；
相同分数按输入顺序保留。有限四个 finalists 的后排名也使用同一先验，
不会自动变成一次测量搜索。

具体代码是 `stage_cost_model.stage_terms`：未有核心服务样本时，compute
使用统一的 `equivalent_butterflies_per_second`，memory 使用统一带宽，
再组合 shared、barrier 和调度项。核心与预取身份虽然保存在 primitive key，
但空样本先验没有从这些身份得到不同实测服务时间。这解释了为什么当前
四点产生同分，不能靠扩大 finalist 数量保证消除该歧义。

因此，这里的选择损失有可复现实证：已有合法设计比 AUTO 快 2.2962×，
但先验未将它区分出来。不能把该结果归因为方法无法流水，也不能把它仅
描述为“重新校准即可”。需要补的是与真实 launch/core 对应的静态服务数据
及覆盖验证，再由安装后 tuning 确认、缓存赢家。当前还没有证明这三个候选
中的最好点是全空间最优，也没有超过同负载 cuFFT。

B=16 时四列无预取略快于两列预取，B=4 的六次受控测量则证明固定两列时
预取有效。这符合机制按负载和参数选择的原则；不同 columns 的结果不能
用于单独归因预取收益。

## 证据与边界

本地原始记录（local artifacts）：

- `results/common_planner_20261024/`：逐 case stdout/stderr、JSON、设计点、
  静态资料身份、选择说明、完整排名 trace 和空闲等待状态。
- `results/common_planner_20261024_buffering/`：三个合法缓冲组合。
- `results/common_planner_20261024_buffering_control/`：固定形状的六次交替测量。
- `results/common_planner_20261024_buffering_baseline/`：同卡、同负载的 cuFFT 对照。
- `results/common_planner_20261024_baseline/`：公共矩阵四个 FFT 点的同卡 cuFFT 对照。
- `results/common_planner_20261024_selection_control/`：大尺寸 FP64 的三个合法
  对照，以及相同物理描述送入公共静态排名的请求、分数与原始记录。
- `results/common_planner_20261024_gpu_tests/`：C++ 公共机制、Graph 与 API 回归。
- `results/recalibration_20261024/core_stage_calibration_descriptor_fixed_gpu1_retry.json`
  与对应 validation JSON：描述符修复后的三点 FP64 stage-service 校准及独立
  stage holdout；完整计划与跨候选选择仍明确标为未验证。

[物理启动审查](physical_launch_descriptor_audit_20261024.md) 中部分旧 NTT
后端仍缺少完整实际 launch 描述。它们保留执行和测量入口，但静态成本排名
明确标为 unranked。CompactStage 构造后可以复用已有 StageProbe 的真实三组
及编译资源。未知编译属性没有被填成实测值。

部分预编译 FFT FP32 和 scalar 路径仍没有输出完整编译资源；公共 estimator
能消费逐组反馈，不代表每个 provider 都已经提供反馈。xD 的独立选择器也
尚未接入本轮完整排名路径。

首次无 registry 命中的 FFT 排名会遍历数十万个投影候选，CPU 准备仍较重。
本轮只消除了大 JSON 的额外复制，并启用了预编译 CPU 调度库的只读复用；
没有用近似剪枝或缩小理论空间来掩盖准备成本。后续应继续补齐物理 launch
合同，并把相容请求的准备结果固化，分别验证初始化开销和运行性能。
