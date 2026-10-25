# FFT 边界布局与独立 tile 分组实验

本轮复用 shared-iterative 的标量 radix-2 核心，检查边界布局和 CTA 内独立 tile
分组的收益。实验入口为 `cubutterfly_boundary_bench`，配置为
`configs/experiments/shared_boundary_20261024.json`。这是显式核心组织实验，尚未接入
公共 AUTO 枚举、JIT 参数身份或安装校准；结果不能代表公共库自动选择性能。

## 控制变量和设计点

主性能点固定 `N=2^20, B=16, 10+10`，两个大阶段各计算 1024 点子图。
两个阶段使用同样的每 CTA tile 数与线程数，初始扫描 tile 数为 1、2、4，线程数
为 256。计算公式、输入、分段、twiddle 表、输出顺序保持不变。

| 组织 | stage 0 写回 | 边界节点 | stage 1 读取 | Graph 节点 |
| --- | --- | --- | --- | ---: |
| strided | 规范布局 | 无 | 跨步 | 2 |
| materialized | 规范布局 | 独立 tile 转置 | 对齐布局 | 3 |
| producer | 直接写消费者布局 | 无 | 对齐布局 | 2 |

每 CTA 同时分组独立 tile，属于数据维度展开。它不同于已有
`batch_tile_count` 对外部 batch 的顺序循环。每个大阶段仍等待前一大阶段完成，
本实验没有引入跨大阶段 overlap，也没有新增异步预取。

shared 核心的两个缓冲用于蝶形层间的读写分离，不是下一 tile 的异步预取缓冲。
其动态 shared 用量为 `2 × tile点数 × 分组数 × 元素字节数`。FFT FP64 的
1、2、4 分组分别使用 32、64、128 KiB；8 分组需要 256 KiB，在本机 A100 的
静态容量检查中应被拒绝。容量合法不等于吞吐最优，线程数、寄存器和驻留度仍需检查。

## 测量和数值口径

FFT/FWHT 均覆盖 FP32、FP64。小规模检查使用不等长分段 `5+3`、B=3、正逆方向、
1/2/4/8 分组，用于检查地址映射。所有方向在本入口采用不归一化语义；这不是完整
归一化选项矩阵。主性能点采用正向，Graph 回放 warmup=5、repeat=20、trials=5，
各 trial 轮换三种组织和 cuFFT 的执行次序。编译、初始化、capture、CPU 参考及拷贝
均不计入 CUDA-event 运行时间。保存每次 trial、最小值、中位数和最大值。

CPU 参考的蝶形运算采用对应 FP32/FP64，系数先生成再转换到该精度。分别保存最大
绝对误差、相对 L2、非有限值数量以及相对原单 tile 核心的输出差异。组织改动不改变
运算顺序，因此要求相对原核心差异为零；相对 CPU 和 cuFFT 的浮点误差保持报告模式，
不临时设定或放宽阈值。所有 batch 均检查。cuFFT 是完整外部基线，其内部映射未知；
CSV 中该行沿用的实验 tile/线程上下文不能解释为 cuFFT 的内部设计点，shared=0 表示
该字段未测量。FWHT 本轮只做固定核心机制比较，没有外部库性能结论。

## 集成审计

已有 factor-streamed 路径本身就融合了写回布局，不能把本次 shared 实验说成首次
实现了整个框架的融合写回。shared 的独立转置路径目前仅能显式指定；阶段探针曾
错误地用融合写回替代独立转置。本轮把该探针组合明确标为 unavailable，保留完整
plan 的执行能力，防止生成不对应实际执行的校准样本。后续要添加真实 boundary
probe，再验证完整组合服务代价；现有 strided 阶段采样不受影响。

资源投影还补齐了 `direct_boundary` 简写的读取，显式 boundaries 列表优先，融合
边界不计全局物化流量。CPU 回归覆盖简写、显式覆盖和融合边界索引。

## 复现

先构建 `cubutterfly_boundary_bench`。使用 `scripts/run_isolated_gpu.py` 指定 GPU
UUID、结果目录和明确截止时间，再执行以下命令。守护程序提供 `XD_REPORT_DIR`，
只将无外部进程干扰且成功完成的一次尝试标记为 release。

```sh
CUBUTTERFLY_COMPILE_MODE=research /home/wt/yes/envs/cubutterfly/bin/python \
  scripts/run_shared_boundary_experiment.py \
  --binary build-e13-cufftdx/cubutterfly_boundary_bench \
  --config configs/experiments/shared_boundary_20261024.json
```

## 实测结果

2026-10-24 在 A100 80GB PCIe、UUID
`GPU-2971e500-d3f6-d7e6-e5f2-cb67a06fff02` 完成测量。以下单位均为 ms，数字是五次
trial 的中位数。各行保持同一算术核心、分段、输入和输出语义。

| 算子/精度 | tile/CTA | threads | strided | 独立转置 | 写回对齐 | 同测 cuFFT |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FFT FP32 | 1 | 256 | 1.88785 | 1.84699 | 1.87141 | 0.38298 |
| FFT FP32 | 2 | 256 | 1.39520 | 1.53651 | 1.42853 | 0.38344 |
| FFT FP32 | 4 | 256 | 1.66876 | 1.80357 | 1.62355 | 0.38277 |
| FFT FP64 | 1 | 256 | 2.26478 | 2.45530 | 2.55785 | 0.89108 |
| FFT FP64 | 2 | 256 | 2.27620 | 2.55370 | 2.25853 | 0.88980 |
| FFT FP64 | 4 | 256 | 3.65932 | 3.96508 | 3.64549 | 0.89068 |
| FFT FP64 | 2 | 512 | 2.01999 | 2.30958 | 2.02578 | 0.89610 |
| FFT FP64 | 2 | 1024 | 2.07959 | 2.38172 | 2.09644 | 0.89590 |
| FFT FP64 | 4 | 512 | 2.66307 | 2.94871 | 2.64422 | 0.89088 |
| FFT FP64 | 4 | 1024 | 2.39299 | 2.70106 | 2.38418 | 0.89042 |
| FWHT FP32 | 1 | 256 | 1.04612 | 1.07976 | 1.07044 | — |
| FWHT FP32 | 2 | 256 | 0.98627 | 1.06834 | 0.97106 | — |
| FWHT FP32 | 4 | 256 | 0.93901 | 1.05160 | 0.95196 | — |
| FWHT FP64 | 1 | 256 | 1.24877 | 1.19071 | 1.14463 | — |
| FWHT FP64 | 2 | 256 | 0.99046 | 1.15779 | 1.00424 | — |
| FWHT FP64 | 4 | 256 | 1.17243 | 1.33755 | 1.17663 | — |

相对同核心原 1-tile strided 点，本轮已测最佳点的加速比分别为 FFT FP32
**1.353×**、FFT FP64 **1.121×**、FWHT FP32 **1.114×**、FWHT FP64 **1.261×**。
这些收益证明有限范围内的组织参数有价值，不等于相对各算子原有最快实现的收益。
例如这组 FFT 标量核心最佳点仍分别只有同测 cuFFT 的 0.275×、0.444×。
已有 factor-streamed/cuFFTDx FP64 路径在本次构建复测为 **1.12517 ms**，Graph 为
两个 kernel 节点，和此前约 1.124 ms 接近，仍明显快于本轮标量候选。该复测采用
原 Graph benchmark 的输入和 CPU 参考，属于原路径回归确认，不混入上表的五次
trial 统计。本次不能宣称提升了 cuButterfly 已有最佳 FFT 性能。

所有已测“最佳”均来自上表的有限枚举，不是模型自动选点，也不是全设计空间最优。
最优组织并不统一要求写回对齐：本轮四类的最小值均来自 grouped strided 点，
而写回对齐只在部分固定分组中有收益。完整负结果保留在表中。

## NCU 定位与受控验证

针对 FP64 分组变慢的问题，先固定 256 线程比较 1-tile strided 与 4-tile 写回对齐，
再固定 4-tile 写回对齐，只将线程数提高到 1024。表中区间是两个 stage 节点的指标
范围，不是统计置信区间；NCU 节点输出次序不作为 stage 编号。

| 设计点 | 正式 Graph ms | 动态 shared/CTA | registers/thread | active warps | SM throughput | 全局 store sector 利用率 | long scoreboard stall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 tile、256 threads、strided | 2.26478 | 32 KiB | 30 | 49.06–49.28% | 32.75–60.38% | 50–100% | 30.74–55.33% |
| 4 tiles、256 threads、写回对齐 | 3.64549 | 128 KiB | 30 | 12.49–12.50% | 25.74–27.00% | 100% | 41.21–44.11% |
| 4 tiles、1024 threads、写回对齐 | 2.38418 | 128 KiB | 30 | 49.86–49.91% | 40.80–48.14% | 100% | 34.58–39.07% |

增加分组确实改善了写事务，但 shared 容量使 256-thread 版本只能保持很少的活跃
warp。改变线程数后，shared、算术和写回布局不变，active warps 恢复到约 50%，
正式延迟降低 **34.6%**。这项受控结果把一部分损失定位到资源与线程映射，而不是
“大阶段必须同步”或混合数据流的依赖范式。它也表明容量约束只判断能否执行，
还需要模型评估驻留和服务吞吐。

恢复驻留后仍有约 35–39% long-scoreboard stall，global load sector 利用率为
60.48–71.49%，且该标量路径没有达到已有 factor 核心的速度。剩余差距涉及局部核心
及其供数/交换组织，本轮没有把这些因素完全分离；不能据此宣称理论最优，或反过来
宣称理论无效。下一步优先在现有高性能局部核心上验证相同的资源—线程—布局组合，
避免继续只扩大标量候选。

## 验证与证据位置

32 个小规模配置和 16 个大规模配置中，三种组织与原单 tile 核心的输出差异均为
零，输出全有限。FFT 大规模 CPU 最大绝对误差为 FP32 `7.55e-4`、FP64 `1.41e-12`，
相对 L2 分别为 `1.78e-7`、`3.55e-16`。主扫描 cuFFT 对相同 CPU 参考的对应值为
`1.34e-3`/`4.31e-12` 和 `3.33e-7`/`8.43e-16`。这些是误差报告，未新增浮点
通过阈值；FWHT 与 CPU 参考差异为零。

256 KiB 的 FP64 8-tile 组合在分配与 launch 前按静态硬件文件被拒绝。另在空闲
A100 40GB 上完成 `mixed_dataflow`、`native_stage_probe`、
`shared_graph_mechanism_correctness`、`fft_shared_iterative_correctness` 四项相关回归，
均通过。这里只报告受影响的检查，不代表全仓库测试矩阵已经重跑。

原始证据均在 `results/recalibration_20261024/`，新旧构建记录分开保存：

- `shared_boundary_groups_gpu1/release/`：32 个小配置、12 个性能配置、1 个资源拒绝，
  每个配置的 CSV/stderr 和完整 `manifest.json`。
- `shared_boundary_threads_gpu1/release/`：根据驻留诊断追加的 4 个 FP64 线程配置。
- `shared_boundary_profile_gpu1/release/`：两个原始 NCU profile。
- `shared_boundary_threads1024_profile_gpu1/release/`：1024 线程的确认 profile。
- `shared_boundary_regressions_gpu0/release/runner.log`：四项相关回归日志。
- `factor_confirm_boundary_build_gpu1/release/runner.log`：当前构建原 factor 路径确认。

上述正式性能扫描和 profile 均由独占进程守护完成，未发生混跑隔离。NCU 时间只用于
诊断，表中性能均来自未插桩的 CUDA-event 结果。构建、源码校验和及硬件身份保存在
manifest 中。针对性实验已完成，公共候选贯通、完整 plan 成本校准与 FFT 性能目标
仍是后续工作，不能以本报告替代。

2026-10-25 在同一 A100 80GB 上继续复用已有 cuFFTDx factor 核心，验证了 CTA 内
连续 tile 组数和邻域 factor 参数。该复测、逆向方向、B=4/16/64 结果及 NCU 证据见
[`factor_graph_exploration_20261025.md`](factor_graph_exploration_20261025.md)。
