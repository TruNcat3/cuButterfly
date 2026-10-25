# 公共 xD radix-4 pair 核心接入

这是 FP32 首次接入阶段的历史记录。后续 FP64、实数 packing 和长一维 JIT 的实现及测试见 [补齐报告](radix4_completion_20261024.md)，下文保留原构建的支持范围与数据。

本报告记录 2026-10-24 将经过独立微基准验证的 `radix4_register` 局部核心接入公共 xD lowering、候选搜索、缓存重放和 Python API 的结果。它补齐了公共调用闭环，但只宣称已验证的物理范围：连续 complex64、二维 `32×32` pair 以及包含该 pair 的三维 `[8,32,32]` 样例。长一维 FFT、FP64 和 compact-real 的 pair packing 仍是单独的支持项。

## 公共路径

公共候选名为 `xd:pair-tiled-radix4`，映射身份中同时记录 `pair_core=radix4-register`、`pair_axis_radices=[4,4,2]`、`io_exchange=padded-shared-complex64` 和 `transform_semantics=separable-axes`。`xd_pair_core` 统一封装可用性查询、准备、发射、核心名称和 twiddle policy，避免候选只在某个 benchmark 中可见。严格的名称解析用于显式 mapping 重放，未知核心不会静默回退到默认实现。xD 缓存版本升至 `v25`，因此旧缓存不会跳过新候选。

公共 runner 为每个请求分别构建 `auto`、强制 radix-4、已有 cuFFTDx pair，以及二维请求的完整 cuFFT provider。计划初始化、编译、分配和 CPU 参考不计入 kernel 时间。所有 provider 都放入预分配缓冲区和 CUDA Graph 串行链中，逆变换的归一化 scale kernel 计入其完整调用时间。PyTorch `torch.fft` 单独保留为框架路径基线，不能把它与完整 cuFFT provider 混为一谈。

另外用 `CUBUTTERFLY_ENABLE_CUFFTDX=OFF`、`CUBUTTERFLY_ENABLE_XD_CUFFTDX=OFF` 的现有静态构建重新编译了 `cuntt`，新增 radix-4 pair provider 通过；这只验证可选 MathDx 头文件不是该 provider 的编译依赖，不等同于在无 cuFFTDx 构建上重复本轮 GPU 数字。

## A100 80GB 公共调用结果

测量 GPU 为 A100 80GB PCIe，UUID `GPU-0eae5f07-33de-321b-70a9-ffeae49309e6`。每个点使用 5 个 trial、每个 trial 5 次预热和 20 次 Graph 链重复，取中位数，单位为 μs。`cuFFT whole` 是同一公共库中显式委托完整变换的 provider，逆向包含独立输出 scale kernel。

| 形状 / B | 方向 | auto 选择 | cuButterfly radix-4 | cuFFTDx pair | cuFFT whole | cuFFT whole / auto（加速比） |
|---|---|---:|---:|---:|---:|---:|
| `32×32 / 1` | 正向 | 4.181 | 4.174 | 4.531 | 5.235 | 1.252× |
| `32×32 / 1` | 逆向 | 4.218 | 4.222 | 4.480 | 6.669 | 1.581× |
| `32×32 / 64` | 正向 | 4.333 | 4.334 | 4.798 | 5.742 | 1.325× |
| `32×32 / 64` | 逆向 | 4.366 | 4.365 | 4.803 | 7.445 | 1.705× |
| `32×32 / 256` | 正向 | 5.099 | 5.086 | 5.266 | 8.518 | 1.671× |
| `32×32 / 256` | 逆向 | 5.141 | 5.152 | 5.269 | 11.264 | 2.191× |

二维六个点均自动选择公共 radix-4 pair。相对完整 cuFFT，auto 为 `1.252–2.191×`，强制 radix-4 与 auto 的差异在测量噪声范围内。三维 `[8,32,32]`、B=2 没有强制完整 cuFFT provider，保留 PyTorch cuFFT 作为框架路径基线。自动搜索在正向/逆向分别选择 `xd:axis-tiled-mixed`，为 6.821/6.938 μs，即相对 PyTorch 路径 `1.432/1.408×`。强制 pair-radix4 为 8.760/8.742 μs，说明公共搜索确实会在包含其他轴时选择完整调用更优的组织，而不是强制使用局部候选。

## 正确性、缓存和边界

每个 provider 都对照 SciPy complex64 CPU FFT，保留完整误差分布而不新增阈值。二维 radix-4 的最大绝对误差为 `1.57×10⁻⁵`，相对 L2 约 `1.40×10⁻⁷`；三维 auto 的最大绝对误差为 `6.82×10⁻⁵`（正向）和 `8.79×10⁻⁹`（逆向），相对 L2 为 `1.98×10⁻⁷` 和 `2.33×10⁻⁷`。所有输出有限，Graph 与 eager 输出逐元素一致，输入保持不变。

每个二维点的 default cache 重放都命中 `selection_source=user_cache`，mapping、输出和输入检查全部通过。强制 radix-4 和 cuFFTDx 的 JSON mapping replay 也逐元素一致。额外的位反转边界请求完成了 forced mapping 与 exact JSON replay；FP64、未编译尺寸和未知 pair core 均被拒绝并保留错误记录，不通过升精度、缩小 batch 或替换算子代跑。

## 证据边界

这次结果证明局部 radix-4 核心已成为公共 xD 候选，并能由同一选择器在二维和三维样例中参与比较。它不证明所有 FFT 尺寸都应使用 radix-4，也不证明长一维 FFT、compact-real 或 FP64 已经获得同样收益。旧的局部驻留报告仍保留微基准与 NCU 证据；本报告的数字来自公共 API 的完整调用路径。

复现脚本为 [`run_public_radix4_pair_experiment.py`](../scripts/run_public_radix4_pair_experiment.py)，结果和逐元素误差分布位于 [`results/public_radix4_pair_20261024/external_baseline_rerun/`](../results/public_radix4_pair_20261024/external_baseline_rerun/)。
