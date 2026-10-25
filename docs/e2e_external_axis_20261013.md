# 外部计算核心进入公共 xD 选择

本轮将 VkFFT 从外部横评对象扩展为可组合的单轴计算核心。cuButterfly 仍
负责轴顺序、布局、实数边界、尺度放置及完整候选的测量选择，没有将整个
外部多维变换重命名为自己的实现。核心来自谁不作为选择条件，完整成本和
语义能力才是条件。

## 实现范围

`axis_resources[].core="vkfft"` 可与既有 cuFFTDx/register/local 核心混合。
统一族 `xd:axis-vkfft` 用于显式验证；应用的主结果使用公共自动选择。
新核心进入 C2C、R2C/C2R 复杂轴的统一候选及逐轴细化，未添加应用名称、
GPU 型号或某个性能请求的强制选择规则。缓存更新为 v20，并包含 provider
是否启用。旧映射的实际布局和数值语义保持。

自然布局中的一轴写作 `[prefix, length, suffix]`。适配器让 VkFFT 仅计算
length 轴、跳过 suffix 轴，并将所有 prefix 纳入 batch，从而避免旧
strided cuFFT 逐 prefix 的外层调用。已验证的适配域为 FP32/FP64、正逆向、
至少 2 点的二次幂轴长，suffix 可以不是二次幂。这不是 VkFFT 本身的完整
支持范围；适配器未开放的情况保留原有候选并记录缺口。

此适配器不暴露任意输出缩放。最后一轴若使用它，lowering 仍执行并计时
必要的 scale pass；不能仅因核心被导入，就省略归一化。临时缓冲区和查表
由 provider 持有，单独报告字节数。计划生成、NVRTC 编译与分配在计时外，
执行只更新地址和 stream 并提交已有核心。

构建开关为 `-DCUBUTTERFLY_ENABLE_XD_VKFFT=ON`，默认跟随既有
`CUBUTTERFLY_ENABLE_VKFFT`。禁用时链接 stub，报告该候选未编译。

## 构建和验证证据

研究修订号为 `0.9.3.8-e2e-external-axis-20261013`。最终库
`build-e2e-vkfft2-20261013/libcuntt.so` 的 SHA-256 是
`4f25ddd1fca9f5cfeb34db589980d7ffae8935d36be0e0b3fc39c4ff9d6eebbd`。
基线是上一轮 `89a0389…` 分组轴优化库，保持不变。增量构建记录了全部
链接对象、源和头文件、外部 VkFFT 头文件及库的哈希；继承旧嵌入指纹这一
限制在收据中明确记录，以最终库 SHA 和 v20 缓存隔离研究结果。

验证采用同精度 CPU SciPy 参考，浮点误差仅报告，没有新增通过阈值。

| 层次 | 覆盖与结果 | 最大相对 L2，FP32 / FP64 |
| --- | --- | --- |
| 单轴 provider | 48 组，2048/4096、suffix 1/17/1025、prefix 1/2、正逆；原地/异地、地址变化、Graph 均完成 | `4.43e-7 / 3.73e-16` |
| 公共 API | 38 个二维/三维请求，C2C/R2C/C2R、归一化及额外缩放；exact JSON 重放一致；另 3 个超出适配域请求明确拒绝 | `3.88e-7 / 3.79e-16` |
| 混合核心 | 22 个请求执行完成，2 个非法融合缩放请求正确拒绝；两种核心顺序、保留转置布局、batch tile 和 Graph 回放 | `2.59e-7 / 3.32e-16` |

所有可执行验证均检查输入保持、有限输出及 Graph 与同后端 eager 逐位一致。
公共 API 首次验证因测试脚本漏写 `xd:` 显式族前缀而失败，原 attempt 保留；
修正后的最终证据位于 `public_validation_v2`。provider 的初始化后异常资源
清理也已补齐，最终库包含该修复。25 项现有适配热路径/归一化 CPU 回归通过。

## 比较口径与解释

复用上一轮固定的四个 Poisson 请求（1024²/2048² × B=4/16）和
2048²、B=16 的相位相关请求，每个请求两个独立进程。A100 80GB PCIe
`GPU-2971e500-d3f6-d7e6-e5f2-cb67a06fff02` 独占测量；所有后端完整
CUDA Graph，计入设备输入刷新、计算和独立输出拷贝，不计编译、搜索或捕获。
结果保留 wall 与 CUDA event 两套口径。新旧报告的输入、runner、归一化、
协议及库/GPU 身份逐项匹配，未完成项不替代为预期值。

主矩阵的 10 个进程均已完成且协议匹配，零缺失、零身份冲突。下表为两个
进程的完整调用 wall 加速比范围，大于 1 表示 cuButterfly 更快。

| 场景 | 类型 | 对照 | 加速比 |
| --- | --- | --- | --- |
| 周期 Poisson，1024²、B=4 | FFT FP32，2D R2C/C2R，Graph | Torch/cuFFT | `1.045–1.075×` |
| 周期 Poisson，1024²、B=16 | 同上 | Torch/cuFFT | `1.184–1.191×` |
| 周期 Poisson，2048²、B=4 | 同上 | Torch/cuFFT | `1.152–1.163×` |
| 周期 Poisson，2048²、B=16 | 同上 | Torch/cuFFT | `1.171–1.190×` |
| 周期 Poisson，2048²、B=16 | 同上 | CuPy/cuFFT | `1.203–1.207×` |
| 相位相关，2048²、B=16 | FFT FP32，2D R2C/C2R + 峰值归约，Graph | Torch/cuFFT | `1.047–1.081×` |
| 相位相关，2048²、B=16 | 同上 | CuPy/cuFFT | `2.351–2.513×` |

另补两次非二次幂滤波回归（图像 1024²、滤波核 31²、B=16，实际 FFT
1080²），汇总共 12 个进程。此组仅采集 CUDA event，保持单独口径：

| 场景 | 类型 | 对照 | 加速比 |
| --- | --- | --- | --- |
| 固定核图像滤波，Graph，缓存滤波核频谱 | FFT FP32，2D R2C/C2R | Torch/cuFFT，同 padding 控制组 | event `1.095–1.096×` |
| 同上 | 同上 | CuPy FFT convolution | event `1.075–1.079×` |

1080 轴超出本轮 VkFFT 适配域，自动选择保留已有 packed cuFFT 核心。
其 old/new 为 `0.9961–0.9972×`，即新测量略慢约 0.3–0.4%；同轮外部
对照时间也略长。保留这些原始差异，不将此组表述为新增加速或归因于核心替换。

与上一轮同请求相比，Poisson 的 old/new 为 `1.035–1.053×`，相位相关为
`1.037–1.055×`。相位相关从原 `0.992–0.993×` 转为快于 Torch；CuPy
归约本身较慢，因此对 CuPy 的较大加速比仍不能归因于 FFT 核心。
两次相位相关的 cuButterfly 延迟为 `5.141/5.298 ms`，Torch 为
`5.382/5.727 ms`，存在运行间波动，不能将两点范围当作统计置信区间。
其返回平移量全部精确匹配已知输入位移；同精度 CPU 相关面诊断最大相对 L2
为 `1.407e-6`，继续仅报告误差。

首个大 Poisson 请求的完整调用为 `3.073 ms`，对照 Torch `3.597 ms`。
自动选择正向 VkFFT 与逆向 cuFFTDx：正向 VkFFT 完整变换测得 `0.885 ms`；
逆向 VkFFT 加独立缩放为 `1.301 ms`，慢于含融合缩放的 `0.999 ms` 候选。
这说明核心可独立替换，但最终选择必须包括其余边界成本。

同一次普通 trace 中，正向 VkFFT 核心约 `526.6 us`，逆向 cuFFTDx 核心约
`705.1 us`；Torch 两个复杂轴共 `1042.2 us`。我们的轴核心累计仍较高，
完整应用通过减少额外归一化与输入复制取得收益。此次新增收益同时包含
核心更换，不能将其全部解释为“固定核心的数据流收益”。

独立核心诊断固定 N=2048、suffix=1025、prefix=16、FP32 和单位输出尺度，
每项 7 个样本、每样本 20 次 Graph 回放。cuFFTDx 参数固定为上轮所选点，
并非强制其重新搜索后的全空间最优值。

| 方向与核心 | 核心 Graph 中位数 | threads / regs per thread | shared / active blocks per SM |
| --- | ---: | --- | --- |
| 正向 cuFFTDx，EPT16、group2、exchange2 | `1013.6 us` | `256 / 80` | `32784 B / 3` |
| 正向 VkFFT 单轴 | `617.8 us` | `512 / 54` | `65536 B / 2` |
| 逆向 cuFFTDx，EPT8、group4、exchange0 | `684.0 us` | `1024 / 32` | `65568 B / 2` |
| 逆向 VkFFT 单轴 | `618.6 us` | `512 / 54` | `65536 B / 2` |

这是不同核心实现的完整轴对比，包含其内部访存组织。VkFFT 的线程块采用
`[4,128,1]`，同时处理相邻 suffix 线，寄存器和线程组织均不同；正向 active
block 数并未增加，因此不能概括成 occupancy 提升。尚无本轮带宽或等待
计数器证据来唯一归因，独立核心时间也不能替代应用内 trace。逆向核心本身
虽更快，但完整变换的额外缩放成本仍让已有融合路径占优。

逐请求最终结果以[自动汇总表](../results/e2e_vkfft_axis_20261013/summary.md)
和[包含原始误差、候选时间及 mapping 的 JSON](../results/e2e_vkfft_axis_20261013/summary.json)
为准。

## 复现入口

- [隔离增量构建](../paper/tools/build_e2e_vkfft_axis_20261013.py)，以
  `--destination build-e2e-vkfft2-20261013 --execute` 构建；已有库不可覆盖。
- [公共应用矩阵](../paper/tools/run_e2e_fft_opt_matrix_20261013.py)，指定
  `--library`、`--library-sha256`、新 `--output`、明确 `--until`，使用
  `--skip-old-control` 复用不变旧库证据。
- [公共验证](../results/e2e_vkfft_axis_20261013/public_validation_v2/release/public_validation.json)、
  [混合验证](../results/e2e_vkfft_axis_20261013/mixed_validation/release/mixed_validation.json)、
  [独立核心验证](../results/vkfft_axis_provider_validation_20261013/release/summary.json)。
- [真实应用缓存回放](../results/e2e_vkfft_axis_20261013/public_cache_replay/release/replay.json)：
  两个新 adapter 的正逆共 4 个 plan 均命中 user cache，mapping 与源应用一致。
- [固定请求的核心计时与 Driver 资源探针](../results/e2e_vkfft_axis_20261013/core_profile/release/core_profile.json)。
- [身份与协议检查后的汇总器](../paper/tools/summarize_e2e_vkfft_axis_20261013.py)。
  滤波回归复用既有 `run_fft_filter_end_to_end.py`，完整参数保留在两个
  `filter_b16_t*/status.json` 中。

本轮没有修改其他蝶形的核心，也没有据此宣称所有算子、精度和平台都已重新
验收。该接口和验证说明外部核心可参与我们的组织与选择，整体研究的全面
性能结论仍按各应用的实际矩阵分别报告。
