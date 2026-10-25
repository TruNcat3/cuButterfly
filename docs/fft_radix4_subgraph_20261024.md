# 可替换 FFT 核心与片上驻留的受控实验

本轮完成了 radix-4 正确性修复、寄存器局部 FFT、同核心驻留/拆分对照，以及共享内存向量对齐修复。
在 A100 80GB 的 **32×32、FP32 C2C 二维 FFT** 上，新驻留候选在六个正逆测试点达到
cuFFT 的 **1.227–1.680×**。这是一组局部子图诊断数据，尚未成为公共选择器的自动性能结果，
也不代表长 1D FFT 或全部 FFT 的提升。

## 范围与来源

- GPU 为 NVIDIA A100 80GB PCIe，UUID `GPU-0eae5f07-33de-321b-70a9-ffeae49309e6`，108 SM。
- CUDA 12.4、驱动 550.127.05，复用 `build-e13-cufftdx` 的 SM80 构建及 cuFFTDx pair provider。
- 输入/输出均为连续自然顺序 complex64，out-of-place，B=1/64/256。正逆变换均不归一化。
- `[32,32]` 是真实二维语义，不等同于 `[1024]` 一维 FFT。变成后者还需跨因子 twiddle 和相应布局。
- 实验状态为 `diagnostic`。候选显式指定，没有用此次微基准冒充安装校准或自动搜索。

四点 DFT 参考了本地 VkFFT commit `066a17c17068c0f11c9298d848c2976c71fad1c1`
的 `inlineRadixKernelVkFFT` 中 `case 4 / stageSize == 1`。我们独立实现其和差、±i 旋转和自然顺序输出公式，
没有导入 VkFFT 完整生成器或完整 kernel。来源和 MIT 署名保留在
[`fft_radix4_codelet.cuh`](../src/fft_radix4_codelet.cuh)。因此本报告的 radix-4 候选不能称为“原生 VkFFT”。
完整 VkFFT 的旧对照见[前一轮报告](open_source_template_exploration_20261023.md)。

## 实现与控制变量

| 候选 | 局部计算 | 轴内边界 | 轴间交换 | CTA 线程数 |
|---|---|---|---|---:|
| radix2_stage | 五级 binary DIT | 共享内存 ping-pong | 片上 | 1024 |
| radix4_stage | 两级 radix-4 加一级 radix-2 | 共享内存 ping-pong | 片上 | 1024 |
| radix4_register | 同一 mixed-radix 32 点 FFT | 寄存器 | 补齐的共享 tile | 128 |
| radix4_register_split | 同上 | 寄存器 | 全局缓冲区，两次 kernel | 128 |
| cuFFTDx pair | 已有 Thread FFT 核心 | 寄存器 | 补齐的共享 tile | 128 |

前两项维持原行列 ownership，比较合并计算阶段的效果。后两项自研候选使用完全相同的
`register_radix4_32`，只改变轴间边界、相应加载/写回和 kernel 数量，作为固定核心的组织对照。
从 stage 直接跳到 register 同时改变核心执行和 ownership，不能将其全部收益归给驻留。

上一轮 radix-4 错误有三个具体来源。binary DIT 的位反序中间状态必须将物理槽位 1/2 对调后，
再按逻辑槽位施加旋转因子。列入口只应执行一次位反序。`3*j` 的旋转因子还可能超过半周期表的范围，
必须补足半周期符号映射。现在两个阶段模板都使用这些一致的映射。
保留的 `stockham` 文件名是历史名称，当前算法是显式 bit-reversal 的 DIT，不是 Stockham autosort。

## 计时与完整结果

每个 CUDA Graph 串行包含 32 次完整变换，读取同一只读输入并覆盖输出。
每个 trial 预热 5 次 Graph、计时 20 次 replay，除以 640 得到每次变换时间。
取 5 个 trial 中位数，轮换 candidate/cuFFTDx/cuFFT 的测量顺序。
这表示固定工作集、缓存预热后的 Graph 稳态时间，不是一次独立请求的端到端延迟，
32 次 replay 内的变换也不计作额外 batch。

编译、plan/Graph 构建、分配、CPU 参考和文件 I/O 均不计时。split 的两次 kernel 及中间全局读写都计时。
cuFFT 在这些点使用两个 kernel，驻留候选使用一个。NCU 与 sanitizer 的计时另存，完全不进入此表。

下表取最终 8 字节对齐版本，单位 μs。加速比为同次进程中的 cuFFT 时间除以候选时间。

| B | 方向 | 新 radix-4 驻留 | 已有 cuFFTDx 驻留 | cuFFT | 对 cuFFT 加速比 | 同核心 split | split/驻留 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | 正向 | 4.0224 | 4.3824 | 4.9904 | 1.241× | 6.1280 | 1.523× |
| 1 | 逆向 | 4.0688 | 4.3712 | 4.9904 | 1.227× | 6.1744 | 1.517× |
| 64 | 正向 | 4.1472 | 4.6528 | 5.5456 | 1.337× | 6.3808 | 1.539× |
| 64 | 逆向 | 4.2080 | 4.6656 | 5.5584 | 1.321× | 6.4288 | 1.528× |
| 256 | 正向 | 4.9328 | 5.1168 | 8.2848 | 1.680× | 7.9296 | 1.607× |
| 256 | 逆向 | 4.9808 | 5.1120 | 8.2848 | 1.663× | 7.9760 | 1.601× |

在全部六点，新候选比已有 cuFFTDx 驻留模板快约 2.6%–12.2%。这是两个具体实现的比较，
不能推出所有自研核心都优于 cuFFTDx。仅替换 stage 算术仍落后，尤其 B=256 的
radix4_stage 比 radix2_stage 更慢；完整 24 点保留在原始 JSON，不从报告中删去负点。

## Profile 解释与对齐修复

第一次寄存器实现只为复数声明 4 字节对齐。补齐 `[32][33]` tile 后仍有 bank conflict。
只将复数类型改为 `alignas(8)`，共享内存大小和逻辑布局保持不变，重新构建并跑完整相同矩阵。
B=256 正向时间由 5.2576 降到 4.9328 μs，下降约 6.2%；逆向由 5.2400 降到 4.9808 μs。
这是声明元素访问粒度带来的工程修复，不能归因于选择了新的分段方法。

NCU 下列数值来自 B=256 正向单个 kernel 的独立采样。它们不与 Graph 时间相加。

| 指标 | stage radix-4，修复对齐前 | register radix-4，对齐前 | register radix-4，对齐后 | cuFFTDx pair，最终构建 |
|---|---:|---:|---:|---:|
| 每线程寄存器 | 55 | 77 | 75 | 80 |
| 共享内存 / CTA，用户部分 | 16 KiB | 8.25 KiB | 8.25 KiB | 8.25 KiB |
| 寄存器允许的 CTA / SM 上限 | 1 | 6 | 6 | 6 |
| 每条已执行指令的平均活跃线程 | 9.68 | 32 | 32 | 32 |
| shared load bank conflicts | 0 | 16384 | 0 | 0 |
| shared store bank conflicts | 16384 | 16384 | 0 | 0 |
| short-scoreboard stall | 9.49% | 2.73% | 2.55% | 2.31% |

stage 版本保留稀疏 lane ownership，大量指令只有少数 lane 工作；合并阶段还把寄存器由
radix-2 的 32 提到 55，使寄存器资源允许的 CTA/SM 从 2 降到 1。这解释了它在 B=256 的负结果，
与“多级数据流必然无效”不是同一结论。寄存器版本消除了级间 shared ping-pong，恢复完整 warp 的算术执行。
最终 register kernel 的实际 warp occupancy 仍约 14.45%，barrier stall 约 36%，
因此本结果也不表示资源已满载或不存在后续优化空间。

split 对照证明该局部工作集上保留中间结果有益。它没有单独分离 kernel 启动与全局地址空间往返的贡献，
且重复地址可命中 L2。NCU 中 DRAM write 为零也不能解释成没有全局写回；不能据此声称 HBM 带宽已饱和，
或者已经证明长链 FFT 的 HBM 换入换出被流水完全掩盖。

## 数值与检查

所有候选及 cuFFT、cuFFTDx 都对照 SciPy 1.15.3 CPU complex64 FFT。正向采用默认不归一化，
逆向使用 `norm="forward"` 保持不归一化。误差累加使用双精度，参考运算本身保持 FP32。
最终随机输入矩阵中没有非有限输出，误差上界如下；这里继续遵循用户指定的“报告误差，不新增验收阈值”。

| 实现 | 最大绝对误差 | 最大相对 L2 |
|---|---:|---:|
| 新寄存器驻留候选 | 1.630e-5 | 1.372e-7 |
| cuFFTDx pair | 1.573e-5 | 1.383e-7 |
| cuFFT | 3.371e-5 | 2.127e-7 |

每点保留输入、三家输出、全部逐元素绝对误差，以及 min/median/p95/p99/max。
另用非原点脉冲和二维复数 tone 检查顺序与方向。CPU 四点核心测试覆盖全部实/虚 basis、
显式顺序、4096 组随机输入和正逆 roundtrip，已注册为 CTest `fft_radix4_codelet_cpu`。
最终构建的四个候选均完成 racecheck、memcheck、synccheck，共 12 次检查，全部为零错误，
racecheck 也无警告。日志及命令见结果目录的 `final_validation.json`，诊断工具时间不作性能证据。

## 复现与剩余边界

```bash
cmake -S . -B build-e13-cufftdx
cmake --build build-e13-cufftdx --target stockham_subgraph_bench cubutterfly_fft_radix4_codelet_tests -j8
ctest --test-dir build-e13-cufftdx -R '^fft_radix4_codelet_cpu$' --output-on-failure
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/wt/yes/envs/cubutterfly/bin/python \
  scripts/run_stage_pair_experiment.py \
  --gpu GPU-0eae5f07-33de-321b-70a9-ffeae49309e6 \
  --output results/fft_radix4_reproduction --batches 1 64 256
```

原始结果在 [`results/fft_radix4_subgraph_20261024/`](../results/fft_radix4_subgraph_20261024/)。
`matrix/` 保留对齐前 24 点，`aligned_matrix/` 为最终 24 点；两者都有源码和可执行文件 hash。
`structured_inputs/` 与 `aligned_structured_inputs/` 为对应数值检查，chain=1 的时间不混入性能表。
根目录首次尝试因 GPU 利用率滚动采样残留而中断，保留但不纳入矩阵。
runner 现在在无进程时最多等三个一秒采样窗口，任何新 GPU 进程仍直接中止测量。

局部核心现已接入公共 xD core/organization 描述，并由 lowering 传递 twiddle、布局、方向和缩放。
公共 API 的完整调用、缓存重放和二维完整 cuFFT 对照见
[公共接入报告](public_radix4_integration_20261024.md)。长 1D、更多轴长、FP64、跨块依赖及
compact-real 的 pair packing 仍需分别验证，不能把这次 32×32 公共结果当作全部 FFT 的完成证明。
