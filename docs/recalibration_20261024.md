# A100 80GB 分层服务校准（2026-10-24）

本轮校准的目的，是把公共规划器使用的静态硬件资料与真实物理执行组连接起来，
而不是把某个测量结果直接写成全局赢家。校准对象、批量范围和可复现身份均写入
`results/recalibration_20261024/`；所有记录都使用当前 A100 80GB 的 GPU UUID、
硬件模型和构建指纹。

## 已完成的有限范围校准

公共入口由 `scripts/prepare_stage_calibration_points.py` 负责规范化显式点文件。
它只转换已有的 `mapping` 字段，不枚举候选、不推断缺失字段，也不将不支持的点
替换成相近点。本轮显式点为 FFT FP64、N=2²⁰、两阶段 10+10、cuFFTDx-block，
分别使用 factor columns=2/depth=0、2/depth=2 和 4/depth=0。校准范围声明为
B=12–16。

在 GPU 空闲窗口内完成的窄范围服务校准结果如下：

| 项目 | 结果 |
| --- | --- |
| 显式设计点 | 3 |
| 实际测量记录 | 14 |
| 每点离散负载 | B=12,13,14,15,16 |
| 精确负载覆盖 | 是 |
| 独立验证批量 | B=14 |
| 校准状态 | `calibrated-stage-service` / `complete` |
| 硬件 | A100 80GB PCIe，UUID 记录于 checkpoint |

完整结果见
`results/recalibration_20261024/core_stage_calibration_b12_16.json`。这是一组有
明确边界的服务曲线，可用于该三点及其 B=12–16 负载的验证；它不代表所有 FFT、
NTT、FWHT、精度、尺寸或 xD 请求都已完成校准。

描述符修复后的同范围重跑使用同型号的空闲 A100 80GB（GPU1），结果保存在
`core_stage_calibration_descriptor_fixed_gpu1_retry.json`。该 checkpoint 状态为
`complete`，仍为 3 条曲线、14 条记录、B=12–16 精确覆盖。公共计划验证记录在
`core_plan_validation_descriptor_fixed_gpu1_retry.json`：6 个独立 stage holdout
全部覆盖，中位相对误差 0.005665，P90 相对误差 0.022793。验证同时明确报告完整
多阶段计划、重叠组合和跨候选 top-1 选择仍是 `unvalidated`，因此该结果只证明
物理描述和独立 stage 服务模型已经闭环。

## 探索性宽范围结果

仓库原有的 28 个显式点也已经过公共入口规范化。受限预算运行得到 72 条记录和
24 条曲线，但每条曲线只覆盖三个负载，缺少服务模型要求的完整中间离散负载，
因此保留为诊断记录，不标记为可发布的全局校准。对应文件是
`results/recalibration_20261024/core_stage_calibration.json`。

## 描述符修复与验证边界

窄范围校准的 StageProbe 会从实际 kernel launch 捕获 `exchange`、
`dynamic_shared_bytes`、`work_blocks` 和编译资源。公共 benchmark 的
`execution_groups_json` 原先只序列化了逻辑组字段、`live_shared_bytes` 和
`grid_ctas`，没有完整携带这些实际 launch 字段。当前版本已经在公共描述中加入
`exchange`、`work_blocks`，并在资源查询成功时加入
`static_shared_bytes`、`dynamic_shared_bytes` 和 known 标志；纯逻辑估算点仍保持
unknown。该修复只改变物理描述的传递，不填默认值或近似值。描述符修复后的二进制
已经在同一 GPU1 上完成窄范围公共 bench。协议固定为 `N=2^20, B=16, FP64,
out-of-place, warmup=5, repeat=20`，并固定 `CUBUTTERFLY_COMPILE_MODE=research`
以匹配校准身份。每个点运行前后都检查 GPU 计算进程，所有正式记录均通过完整
batch 校验：

| resolved mapping | kernel ms | 相对 cuFFT |
| --- | ---: | ---: |
| factor columns=2, prefetch depth=0 | 1.818829 | 0.4912x |
| factor columns=2, prefetch depth=2 | 1.151078 | 0.7761x |
| factor columns=4, prefetch depth=0 | 1.123277 | 0.7953x |
| cuFFT | 0.893389 | 1.0000x |

该结果是同一硬件、同一输入语义下的窄范围实测，不外推到其他尺寸或负载。原始
CSV、stderr 和完整描述符快照见
`results/recalibration_20261024/bench_descriptor_fixed_gpu1/`，汇总见
`descriptor_fixed_clean_summary.json`。此前混入短任务的旧尝试保留在目录中，但未
计入汇总。

同一协议的 `--auto-select` 运行通过校验，但当前公共选择器报告
`no compatible validated cost model`，按硬件速率先验选择了 `online-reorder`
（2.578125 ms，0.3465x cuFFT），没有自动复用上述三点的窄范围 stage-service
曲线。这说明描述符修复已闭环，而完整计划模型和跨候选 top-1 集成仍待验证；本轮
不把显式最佳点冒充为全局自动选择结果。安装流程仍可复用已有硬件资料，用户显式
刷新时再按相同入口扩展点集和负载范围。

## 可执行 CUDA Graph 验证

此前的 factor-streamed 记录只证明了 tile/stage lowering 的普通 kernel launch，不能
单独称为已编译的 tile 子图。本轮新增 `ButterflyPlan::capture_graph` 和 RAII
`ButterflyGraph`，在显式非阻塞 CUDA stream 上捕获实际 lowering，然后实例化并重复
回放。捕获、实例化、计划构建和输入拷贝均在计时之外；计时只覆盖
`cudaGraphLaunch`。图绑定捕获时的输入、输出和 workspace 地址，调用者必须保持这些
资源有效并串行化同一计划的回放。

GPU1 上的固定协议为 FP64 FFT、N=2²⁰、B=16、两因子 10+10、factor columns=4、
warmup=5、repeat=30。实际图包含 2 个 kernel 节点，没有额外的非 kernel 节点，回放
平均 1.12442 ms。同一 plan 的普通异步提交为 1.12790 ms，graph 只带来 1.00310×
的提交路径收益。最大绝对误差 4.47044×10⁻¹²，逐 batch 校验通过。完整机器身份、
构建指纹、映射和原始 stdout
见 `results/recalibration_20261024/bench_descriptor_fixed_gpu1/graph-c4-d0.research.json`。

对同一图做的轻量 NCU 采样显示，两阶段的 global load/store sector efficiency
分别为 99.40%/100%，所以当前主要缺口不是简单的未合并访问。阶段 0/10 的 SM
throughput 为 39.93%/26.78%，active warps 为 47.70%/48.18%，long-scoreboard
stall 为 33.42%/42.65%，并且每个 kernel 使用 64 registers/thread 和 64 KiB
动态 shared memory。也就是说，事务已经对齐，但共享内存和依赖等待限制了可维持的
活跃 warp 与 SM 利用率，后半段尤其明显。NCU 运行会改变时序，因此这些数字只作
机制诊断，不替代正式 CUDA-event 结果。

这里的“子图”是可回放的 CUDA Graph launch DAG。它固化了 tile/stage kernel 及其
依赖关系，但不是把所有 tile 数据永久驻留在片上，也不把 graph capture 的一次性
成本计入算子运行时间。本轮已经在 factor-streamed 和 shared-iterative 的 strided
边界点上完成了 2-node Graph 实测，并对 tiled-transpose 边界完成了 3-node Graph
实测；shared resident 专用模块、重叠 pipeline、xD 以及 C API/Python 入口仍未完成
同等验证，因此不能把这些结果外推为所有算子都已有 resident 子图实现。

### shared-iterative 的边界布局对照

随后在同一 GPU1、同一显式非阻塞 stream 和同一 `N=2^20, B=16, FP64` 协议下，
对 shared-iterative 的两阶段 `10+10` 设计点做了严格串行的 Graph 对照。strided
候选包含 2 个 kernel 节点，显式 tiled-transpose 候选包含两个 stage kernel 加一个
边界物化 kernel，且两者都逐 batch 通过 CPU FP64 参考校验：

| shared 设计点 | 边界布局 | Graph 节点 | Graph ms | 普通 plan ms | Graph/plan | 最大绝对误差 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| writer-aligned | strided | 2 | 2.22481 | 2.25184 | 1.01215x | 1.38802e-12 |
| writer-aligned | tiled-transpose | 3 | 2.41773 | 2.43825 | 1.00848x | 1.38802e-12 |

这项对照同时验证了真正的 shared 子图路径和边界 layout 参数，而不是只在 planner
中记录一个候选名。tiled-transpose 由独立的 32×8 CTA 使用 32×33 shared-memory tile
物化边界，随后把物化结果交给下一 stage，因此图中确实出现第三个 kernel 节点。
完整协议和 NCU 原始记录分别在
`shared-c10-c10.research.json`、`shared-tx-c10-c10.research.json` 和对应的
`*.ncu.csv` 中。

NCU 机制采样解释了为什么显式物化仍没有直接变快。转置 kernel 本身耗时
317.760 us，global load/store sector efficiency 均为 100%，但 DRAM throughput 为
84.70%，long-scoreboard stall 为 72.78%。两个 stage kernel 的诊断时间为
1625.920/966.528 us，第一阶段的 load/store efficiency 为 57.17%/50.00%，第二阶段
为 75.44%/100%。因此它确实修复了边界转换的访问表达，但新增了一次完整的全局读写
和一个 graph 节点，正式 event 计时仍比 strided 增加约 8.3%。这不是正确性问题，也
不是 Graph capture 问题，而是当前 `10+10` 单 CTA tile 组织下，边界物化的流量尚未
被后续阶段的访存收益抵消。

后续审计修正了这里的集成状态：shared `tiled-transpose` 当前支持显式 mapping，
**尚未进入 shared 自动候选枚举**。资源投影会计入额外边界读写，但完整时间模型
尚未单独表示物化 kernel 的启动与服务时间，因此不能称为已经完成成本校准与
自动选择。原阶段探针还曾用融合写回替代该独立边界操作，与完整 plan 不一致；
当前已禁止对该组合输出可独立重组的阶段校准记录，并明确报告缺少 boundary probe。
完整 plan 的上述 Graph 测量不受该探针问题影响。下一轮先用固定核心对照验证
producer 写回与独立 tile-group 的效果，再决定是否补入公共枚举和服务模型。

#### batch tile 扫描

为检查“批量 tile 能否摊薄边界转换”的假设，又固定同一 `N=2^20, B=16, FP64,
10+10` 设计点做了普通 plan 的 event 扫描。这里的 `boundary-layout` 明确写入
边界列表，避免把旧的 `direct_boundary` 简写字段误当成有效布局：

| 边界布局 | batch tile | kernel ms | 最大绝对误差 | 正确 |
| --- | ---: | ---: | ---: | --- |
| strided | 1 | 2.201327 | 原输出舍入为 0 | yes |
| tiled-transpose | 1 | 2.395989 | 原输出舍入为 0 | yes |
| tiled-transpose | 2 | 2.689399 | 原输出舍入为 0 | yes |
| tiled-transpose | 4 | 2.800435 | 原输出舍入为 0 | yes |

在当前实现中，显式转置按整个 batch 物化边界，`batch_tile_count` 只改变 stage CTA
一次处理的 batch 数，不会减少转置读写。因此 tile 从 1 增到 2/4 不能摊薄这项固定
流量，反而因并行 CTA 数减少而变慢。该结果支持将批量分组保留在搜索空间，但不应
把“增大 batch tile”当成转置收益的充分条件；真正有机会获益的候选需要让 producer
写回、边界转置和 consumer 读取共享同一 tile-group 生命周期。

该表沿用当时 benchmark 的正确性判定；其低精度格式化的误差输出不能解释为逐位
一致。新的组织对照分别保存足够有效位的 CPU 误差和同核心输出差异，不沿用这个
被舍入的零值。

后续的同核心 tile-group 扫描、写回对齐对照和 NCU 驻留验证见
[FFT 边界布局与独立 tile 分组实验](fft_boundary_exploration_20261024.md)。该报告保留
正负结果，并区分受控标量实验与原 factor-streamed 高性能路径。
