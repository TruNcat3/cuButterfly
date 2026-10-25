# FFT 固定核心的组织对照

本轮将研究重点收敛到组织收益的因果证据。cuFFT 继续作为完整库竞争力对照，
不把逐点超过它当作方法成立的必要条件。固定核心后比较运行组织，再跨核心复验，
才能区分局部算术效率和分层混合数据流的贡献。

## 有限实验及归因边界

配置见 [factor_graph_core_organization_20261025.json](../configs/experiments/factor_graph_core_organization_20261025.json)。
所有点固定 FP64、正向、自然序、不归一化、N=65536、B=16、相同随机输入，
使用完整 CUDA Graph 的 event 中位延迟。预编译、plan 初始化、capture、CPU 验证均
在计时外。warmup=5，repeat=30，trials=7，trial 内轮换各 case 的执行顺序。

| 固定项 | 设置 |
| --- | --- |
| 逻辑宏阶段 / 物理 factor | `[16]` / `[8,8]`，2 个物理 kernel |
| 每个局部 FFT | 256 点，EPT=16，4 个并行子变换，64 threads/CTA |
| 核心 A | 仓库内 `NativeFactor`，两个 radix-2 寄存器 codelet 和中间交换 |
| 核心 B | cuFFTDx block，保持相同输入形状和 EPT |
| 布局 / 系数 / I/O | writer-aligned / recurrence / dynamic |
| 跨 factor slices / overlap | 1 / 关闭 |
| 每核心的组织选项 | `(连续 tile 数,预取深度)` = `(1,0),(4,0),(4,1),(4,2)` |
| 完整库对照 | cuFFT，同语义和计时边界，内部映射未知 |

每个核心内部只允许 `data_tiles_per_cta` 和 `prefetch_depth` 变化。脚本在准备和
实测后核对请求与实际 mapping，保存完整差分；核心身份、factor、EPT、columns、
布局、I/O 或 slices 不一致时拒绝单变量归因。`t4d0→t4d1/2` 隔离预取变化，
`t1d0→t4d0` 隔离 CTA 连续消费数量变化。跨核心比较单独解释。

这是**匹配组织对照**，不代表外部库的原生组织，也不是完整的“常规组织 vs.
混合数据流”消融。两列均使用本仓库的 factor 模板。NativeFactor 是仓库内可审计
实现，借用了 cuFFTDx 的复数类型但未调用其 FFT 算术；它不等于移植了 VkFFT 或
rocFFT 的局部核心。成熟开源核心上的完整组织迁移、跨算子复验、AUTO 选择与
独立复测，仍需要各自的证据。不得把本轮九点写成已完成这些研究目标。

## 等待协议修正

扩展输入 ring 后，预取深度 D 对应 `current + D` 个槽。迭代 i 等待前最近提交
的是组 `i+D`；保留最近 D 组待完成，才恰好等待 current i。旧指令
`wait_group(D-1)` 多等一个未来 tile，修正为 `wait_group(D)`。尾部空提交维持
相同序号关系。这与 [NVIDIA PTX 等待指令语义](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-wait-group-cp-async-wait-all)
一致；编译产物的 PTX 已分别核对 D=1/2 对应的等待立即数。

旧版只有一个输入槽的 D=1 路径也能重叠：current 装入寄存器后释放输入槽，
下一 tile 的 copy 可与 current FFT 并行。增加输入槽改变的是预取距离与资源
占用，不能称为首次实现重叠。历史报告已更正这一解释，原始数据保持原样。

`t=1` 没有后继 tile，仍执行同步单槽。请求 mapping 保留用户指定的深度，
执行组现在报告实际深度 0，与 Python 物理投影一致。不能从请求中的 d1 推断
实测执行了异步流水，也不能从等待协议正确推断一定获得性能收益。

## 验证与测量状态

- 相关 Python 回归 202 项及 7 个 subtests 通过；C++ 混合数据流资源与投影测试通过。
- A100 40GB（UUID `GPU-2b759faa-3e58-7106-10aa-ea6e387c3f8e`）上四组现有 pipeline
  正确性回归通过，涵盖 FP32/FP64、两个核心、D=1/2、2/3/4 factor、partial slices、
  CTA 尾部、串行/重叠调度，以及逆向、stride 和 in-place 组合。采用既有回归规则，
  未新增或放宽数值阈值。日志见
  [正确性记录](../results/factor_graph_20261025/waitfix_correctness_gpu0/release/runner.log)。
- 同卡 FP32 NativeFactor 四 factor、D=2、含尾部与 partial slices 的
  [racecheck](../results/factor_graph_20261025/waitfix_racecheck_gpu0/release/runner.log)
  完成，0 errors、0 warnings。九点 Graph 的集成检查也通过，已验证仅加载预编译
  模块、输出有限及实际 mapping 差分；该 [smoke 记录](../results/factor_graph_20261025/core_organization_smoke_gpu0/release/runner.log)
  不作为性能报告，也不据此宣称满足新的浮点误差阈值。
- A100 80GB 的九点性能实验于 2026-10-25 13:39（Asia/Shanghai）在独占窗口完成。
  测量只加载 manifest 中的匹配模块，未在 GPU 测量阶段补编译。
  [完成状态](../results/factor_graph_20261025/core_organization_waitfix_gpu1/status.json)
  与 [原始结果](../results/factor_graph_20261025/core_organization_waitfix_gpu1/release/results.json)
  分别保存，40GB 正确性结果不混作 80GB 性能证据。

浮点实验继续单独报告输出非有限值、最大绝对误差、相对 L2 与 CPU FP64 参考，
同时保留 cuFFT 的误差。收益或负结果均保留；本轮不据结果自动追加无界扫描。

## 实测结果与探索收尾

下表为单一进程内七轮轮换测量的中位数，单位 µs；不是七个独立进程的统计估计。
设计点均为两物理阶段、256 点 tile、每 CTA 四个并行子变换、64 threads、EPT=16，
区别只有核心及表列 `(连续 tile 数, 预取深度)`。cuFFT 为完整库对照，内部设计未知。

| 核心 | t1d0 | t4d0 | t4d1 | t4d2 |
| --- | ---: | ---: | ---: | ---: |
| NativeFactor | 82.671 | 138.377 | 79.974 | 98.884 |
| cuFFTDx block | 80.657 | 138.684 | 73.762 | 93.457 |

cuFFT 为 **48.503 µs**。固定 t4 后，d0→d1 分别有 **1.730×、1.880×** 的组织收益。
但 t4d0 本身不是最快的无预取方案；与 t1d0 比较，t4d1 的收益是
**1.034×、1.093×**。必须同时保留这两个参照，不能把较弱控制上的 1.88× 当作
相对最佳基线或完整库的加速。d2 对两个核心均退化，说明更深预取不单调获益。
最快两个自选点相对 cuFFT 分别为 0.607×、0.658×，这个固定局部形状仍有明显差距。

全部九点检查完整 B=16，非有限值为零。NativeFactor 的最大绝对误差为
8.55e-13、相对 L2 为 7.38e-16；cuFFTDx 为 1.01e-12、7.17e-16；cuFFT 为
8.20e-13、6.79e-16。未新增浮点通过阈值。输入/输出与局部算法未变，编译器的
寄存器分配可随组织变化，不能将固定核心解释成整个 kernel 指令和资源相同。

本轮有限探索到此收尾，支持“同一组织机制可在两个核心上产生收益，收益有条件”。
它不证明全空间最优，也不完成全库性能验收。下一阶段按既有语义范围重测算子、
机制与应用，当前和历史 cohort 分开；完整数据检查后再更新论文总体图和 GitHub。

## 复现

使用项目 conda 环境，使运行时模块查询器也使用一致的 Python。先完成已有 build，
再运行准备；硬件文件对应 A100 80GB，不能用于冒充 40GB 测量。

```sh
export PATH=/home/wt/yes/envs/cubutterfly/bin:$PATH
CUDA_VISIBLE_DEVICES='' python scripts/run_factor_graph_experiment.py prepare \
  --config configs/experiments/factor_graph_core_organization_20261025.json \
  --prepared results/factor_graph_20261025/core_organization_waitfix.json \
  --binary build-e13-cufftdx/cubutterfly_graph_bench \
  --hardware-model config/hardware/NVIDIA-A100-80GB-PCIe-sm80-84974239744B/hardware_model.json

python scripts/run_isolated_gpu.py \
  --gpu-uuid GPU-2971e500-d3f6-d7e6-e5f2-cb67a06fff02 \
  --report-dir results/factor_graph_20261025/core_organization_waitfix_gpu1 \
  --until 2026-10-25T15:00:00+08:00 --idle-seconds 3 -- \
  python scripts/run_factor_graph_experiment.py measure \
  --prepared results/factor_graph_20261025/core_organization_waitfix.json \
  --binary build-e13-cufftdx/cubutterfly_graph_bench
```

已有监控器时不重复启动；复测使用新的 report 目录并设置实际需要的有界截止时间。
