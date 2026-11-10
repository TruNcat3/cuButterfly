# xD FFT 端到端优化复测

本轮针对完整应用中已经定位的两个开销做了有限修复和复测：长轴
2048/4096 的轴核心可以按 CTA 处理多条 FFT，逆实数 FFT 的归一化或输出
缩放可以并入最后复数轴的写回。两者都作为公共 xD 搜索候选，旧候选保留
作对照。编译、搜索和 Graph 捕获不计入稳态调用时间；每个应用包含设备输入
刷新、完整 Graph 和独立输出拷贝。

## 当前构建和验收

最终隔离构建为
`build-e2e-scale-20261013/libcuntt.so`，SHA-256 为
`89a038905a45e2d2ab072c142f9bde17ed97b42ed86106bdac656e5ec0f0f33b`。
构建复用旧版 32 个未修改对象，只重新编译 `c_api.cpp`、`xd_real_fft.cu`
和 `xd_axis_fft.cu`，输入、编译命令和对象哈希记录在该目录的
`incremental_build_receipt.json`。旧基线库保持为
`build-e2e-20261013/libcuntt.so`，SHA-256 为
`ddb7d70cc8e0fac900c0b912f03d7aae18855a807d06f0dff4a0ebad1bcb665b`。

逆 C2R 验证覆盖 FP32/FP64、二维和三维、单位与非单位尺度。8 个请求的
旧路径和 prefix-scale 路径均输出有限，输入保持，Graph 与同后端 eager
逐位一致，非法的 forward/C2C/unfused 请求均被拒绝。相对同精度 CPU 的
FP32 相对 L2 误差为 `1.459e-7–1.676e-7`，FP64 为
`2.991e-16–3.900e-16`，这里只报告分布，没有新增浮点通过阈值。

公共 cache 回放也已验证。两个新建的 `RealFFT` adapter 都命中
`selection_source=user_cache`，实际 forward/inverse mapping 与应用生成的
mapping 完全一致，并保留 `c2r_prefix_scale=true`。2048²、B=16 的 FP32
回放相对 CPU FP32 的相对 L2 分别为 `3.424e-7` 和 `5.810e-7`，输入保持
检查通过。

最终使用 Driver API 的资源探针复核了全部 201 个配置，其中 165 个可执行、
36 个因设备资源限制被明确拒绝、0 个执行失败。覆盖二维的 FP32/FP64、
R2C/C2R/C2C 正逆向、EPT8/16、每 CTA 2/4 条轴线、exchange 0/2/3，
另包含一个三维 C2R 请求和八个大网格请求。可执行请求的 373 个 kernel 节点
均能查询资源且具有非零 occupancy。相对同精度 CPU 的最大相对 L2 误差为
FP32 `4.194e-7`、FP64 `4.40e-16`。这里的“可执行”只表示执行、有限值和
结构性检查完成，不代表已经设定并通过了浮点误差阈值。

资源探针曾产生误诊：通过 CUDA runtime 查询 Graph 中由 driver 加载的函数，
会出现 `invalid device function`，残留的 CUDA 错误又被后续合法 launch 读到。
关闭探针后，同一个原始库的 201 个配置中，165 个正常执行，36 个因真实线程或
shared-memory 限制被拒绝，零启动失败。曾因此试加的 group/方向/至少 1024
轴线限制已全部撤回，`build-e2e-scale2/3/4-20261013` 仅作为废弃诊断构建保留，
不代表最终实现。之前标为成功但包含失败案例的验证报告也不作为验收证据。
修正后的逐例探针检查与同精度误差检查另存为新报告，没有覆盖失败记录。

本机 CPU Torch 的大尺寸 FP32 复数 `ifftn(norm="backward")` 还出现了额外
一次 `1/N` 归一化。独立 CPU SciPy 对照确认，使用 CPU Torch 未归一化逆变换
后显式按 FP32 除以 N 可得到匹配的参考。验证器记录两套 CPU 参考的误差，
没有将此前约 N 倍的相对误差解释为“接近零”或接受为正确。此调整只涉及
C2C 验证参考，Poisson 的 CPU 实数变换参考和输入不变。

## 完整 Graph 应用结果

下表中的加速比为对照完整调用延迟除以 cuButterfly 延迟，大于 1 表示
cuButterfly 更快。主矩阵使用 A100 80GB PCIe，GPU UUID
`GPU-2971e500-d3f6-d7e6-e5f2-cb67a06fff02`，每个请求两个独立进程。
Poisson 的主矩阵采用同一输入、同一归一化和同一 Graph 协议；旧库控制组
使用冻结的旧二进制。下表保留 wall 中位数和 CUDA event 中位数，不混合两种
口径。

| 场景 | 类型 | 对照 | 加速比 |
| --- | --- | --- | --- |
| 周期 Poisson，2048²、B=16，Graph | FFT FP32，2D R2C/C2R；grouped axes + C2R prefix scale | Torch/cuFFT | wall `1.110–1.126×`；event `1.111–1.126×` |
| 周期 Poisson，2048²、B=16，Graph | 同上 | CuPy/cuFFT | wall `1.142–1.146×`；event `1.144–1.148×` |
| 周期 Poisson，2048²、B=16，Graph | 同上 | 旧 cuButterfly | old/new `1.171–1.177×` |
| 周期 Poisson，2048²、B=4，Graph | FFT FP32，2D R2C/C2R | Torch/cuFFT | wall `1.109–1.114×` |
| 周期 Poisson，2048²、B=4，Graph | 同上 | CuPy/cuFFT | wall `1.106–1.109×` |
| 周期 Poisson，1024²、B=4/16，Graph | FFT FP32，2D R2C/C2R | Torch/cuFFT | wall `1.005–1.145×` |
| 周期 Poisson，1024²、B=4/16，Graph | 同上 | CuPy/cuFFT | wall `1.034–1.150×` |
| 图像滤波，1024²、31² 滤波核、B=16，Graph | FFT FP32，2D R2C/C2R，缓存固定滤波核频谱 | Torch/cuFFT，同 padding 控制组 | event `1.090–1.093×` |
| 图像滤波，1024²、31² 滤波核、B=16，Graph | 同上 | CuPy FFT convolution | event `1.069–1.070×` |
| 相位相关，2048²、B=16，Graph | FFT FP32，2D R2C/C2R + 峰值归约 | Torch/cuFFT | wall `0.992–0.993×` |
| 相位相关，2048²、B=16，Graph | 同上 | CuPy/cuFFT | wall `2.306–2.336×` |

最终库对 2048²、B=16 Poisson 的两个进程测量为：cuButterfly
`3.237/3.225 ms`，Torch `3.593/3.632 ms`，CuPy `3.698/3.698 ms`。
同轮重新运行冻结旧库得到 `3.841 ms`，对照 Torch 为 `3.626 ms`，再次
复现原先落后约 6% 的情况。新请求自动选中 forward 的
`ffts_per_block=2, EPT=16, exchange=2`，以及
inverse 的 `ffts_per_block=4, EPT=8, exchange=0` 和
`c2r_prefix_scale=true`。这说明端到端收益来自公共选择路径中的实际设计点，
不是手工替换应用流程。

Poisson trace 中，原先按 batch 分别启动的 32 次前缀轴执行减少为 2 次，
独立的实数输出归一化和 C2R 输入复制消失，整个应用 kernel 数由 38 降至 7。
相同局部 cuFFTDx 计算核心之上的执行组织和边界处理因此能带来端到端收益。
但不能说我们的计算核心已经全面快于 cuFFT：新 trace 的两个前缀轴累计
约 `1.420 ms`，Torch 为 `1.038 ms`，收益来自完整变换的组合成本。

相位相关仍略低于 Torch。它的 trace 中，cuButterfly 前缀轴累计约
`1.734 ms`，Torch 约 `1.047 ms`，额外核心时间抵消了减少边界操作的收益。
相关运算和归约进一步稀释收益。CuPy 的峰值归约约 `8.08 ms`，因此相对
CuPy 的 2.3× 不能归因于 FFT 组织。这个负结果保留；下一步要检查应用中的
核心访存效率及选择稳定性，不能只凭单算子结果推算端到端加速。
图像滤波的实际 FFT 为非二次幂的 1080²，本轮分组候选未介入。新旧库延迟
比为 `0.9999–1.0001×`，说明其原有优势保持，不能将其计为本轮新增收益。
该控制组只采集了 CUDA event 时间，不能与主矩阵的 wall 延迟混合统计。

本轮最终汇总包括 13 个独立进程报告：8 个 Poisson、2 个相位相关、2 个
图像滤波和 1 个冻结旧库控制组。所有预声明请求均保留，缺失报告和身份或
协议不匹配报告均为零。范围是这三个 FFT 应用，不代表 NTT/FWHT 应用、
全部精度或完整硬件迁移已重新验收。
所有报告、profile、缺失项和
失败 attempt 均保留在 `results/e2e_fft_opt_20261013/`，没有用缺失结果替代
负结果，也没有把 setup 时间混入加速比。

## 复现与原始证据

- [逐进程完整延迟、误差与选择记录](../results/e2e_fft_opt_20261013/summary.json)
  及[可读汇总表](../results/e2e_fft_opt_20261013/summary.md)。
- [分组配置与 Driver 资源探针](../results/e2e_fft_opt_20261013/grouped_verified_driver/release/grouped_verified_driver.json)。
- [逆变换尺度放置验证](../results/e2e_fft_opt_20261013/prefix_validation/release/prefix_validation.json)。
- [公共 API 缓存回放](../results/e2e_fft_opt_20261013/public_replay_verified/release/public_replay_validation.json)。
- [增量构建脚本](../experiments/tools/build_e2e_incremental_20261013.py)、
  [Poisson/配准与旧库控制矩阵](../experiments/tools/run_e2e_fft_opt_matrix_20261013.py)、
  [结果汇总器](../experiments/tools/summarize_e2e_fft_opt_20261013.py)。滤波复测调用
  `experiments/workloads/e13/run_fft_filter_end_to_end.py`，两个隔离任务的
  `status.json` 均保留完整命令及参数。

三个生产源文件与相关头文件的 24 项输入哈希已和最终构建记录逐项核对。
相关 Python 热路径和归一化回归测试 25 项通过。本轮使用隔离研究构建，
未将旧实验或正在使用其他二进制的任务重标为此版本。
