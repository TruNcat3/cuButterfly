# A100 FFT 映射实例：`register-c16` 与 `dx-thread-c16`

本文档只冻结一个可追溯实例，供正文描述 GPU 映射时使用：A100 80GB PCIe
上的 FP32 complex forward FFT，`N=2^18`、`batch=4`，比较
`register-c16` 和 `dx-thread-c16`。两行使用同一个 prefix ownership、global
boundary、4096-point suffix 和 transpose path；变化项只有 prefix 的局部
arithmetic codelet。这里的 4096 是历史控制点的 suffix size，不是当前最终
API 的固定 suffix 规则。

## Evidence chain

| 事实 | 可追溯来源 |
|---|---|
| 负载、natural/out-of-place 语义、CUDA-event 计时边界、A100 设备身份 | [`report.md:5-10`](../results/fft_register_tile_20260912/report.md:5)、[`paired_metadata.json:4-9`](../results/fft_register_tile_20260912/accurate_core_control/paired_metadata.json:4) |
| 历史控制的 `columns=16`、4096 suffix 和 transpose path，以及两行时间 | [`report.md:60-66`](../results/fft_register_tile_20260912/report.md:60)、[`paired_summary.json:3-29`](../results/fft_register_tile_20260912/accurate_core_control/paired_summary.json:3) |
| 五轮、每轮 100 次、运行 binary 和 verification binary 的 SHA-256 | [`paired_metadata.json:2-24`](../results/fft_register_tile_20260912/accurate_core_control/paired_metadata.json:2) |
| `register-c16` 和 `dx-thread-c16` 的 N=2^18 正确性日志 | [`verify-log18-batch4.log:6`](../results/fft_register_tile_20260912/accurate_core_control/verify-log18-batch4.log:6)、[`verify-log18-batch4.log:13`](../results/fft_register_tile_20260912/accurate_core_control/verify-log18-batch4.log:13) |
| 历史 runner 如何先调用 register prefix，再以 `stage=2` 调用旧 suffix | [`fft_resident_experiment.cu:68-84`](../apps/fft_resident_experiment.cu:68)、[`fft_resident_experiment.cu:99-114`](../apps/fft_resident_experiment.cu:99) |
| 当前源码中 prefix 的 lane、register、shared 和 global index ownership | [`fft_register_tile.cuh:61-106`](../src/fft_register_tile.cuh:61)、[`fft_register_tile.cuh:108-119`](../src/fft_register_tile.cuh:108) |
| 当前源码中 4096 direct suffix 和 tiled transpose 的 dispatch | [`cufftdx_fft.cu:1183-1212`](../src/cufftdx_fft.cu:1183)、[`cufftdx_fft.cu:1337-1371`](../src/cufftdx_fft.cu:1337)、[`cufftdx_fft.cu:1529-1566`](../src/cufftdx_fft.cu:1529) |

`paired_metadata.json` 中的运行设备记录为 `NVIDIA A100 80GB PCIe`、compute
capability `8.0`、`84974239744` bytes global memory；该文件还记录了
`binary_sha256=b7fb0eaf...24119b2` 和
`verification_binary_sha256=5236a7be...1288b38`。元数据没有 source revision
或 compiler flags，因此下面把“源代码可推导的 launch 形状”和“历史 binary
实际资源”严格分开。

## Frozen mapping

### Workload and fixed/changed fields

| 项 | `register-c16` | `dx-thread-c16` | 证据和含义 |
|---|---|---|---|
| FFT | FP32 complex, forward, natural output, out-of-place | 相同 | [`report.md:5-10`](../results/fft_register_tile_20260912/report.md:5) |
| `N`, batch | `2^18=262144`, `4` | 相同 | [`paired_metadata.json:10-17`](../results/fft_register_tile_20260912/accurate_core_control/paired_metadata.json:10) |
| prefix logical size | `2^6=64` | `2^6=64` | runner 对 N=18 调用 `launch_prefix<6,C>`；[`fft_resident_experiment.cu:71-77`](../apps/fft_resident_experiment.cu:71) |
| prefix columns | `C=16` independent columns/CTA | `C=16` | runner 的 `case 16`；[`fft_resident_experiment.cu:73-77`](../apps/fft_resident_experiment.cu:73)、报告 [`:62-64`](../results/fft_register_tile_20260912/report.md:62) |
| prefix register values | `R=2^(6/2)=8` complex values/thread | 相同 | `Complex v[R]`；[`fft_register_tile.cuh:67-81`](../src/fft_register_tile.cuh:67) |
| prefix CTA | 128 threads = `R*C`, 4 warps | 相同 | [`fft_register_tile.cuh:108-119`](../src/fft_register_tile.cuh:108) |
| prefix grid | 1024 CTAs for `B=4`: `(4*2^12)/16` | 相同 | `blocks=(batch << (log_n-LocalLog))/Columns`；[`fft_register_tile.cuh:113-119`](../src/fft_register_tile.cuh:113) |
| prefix local arithmetic | native compile-time FFT-8 codelet, called twice | cuFFTDx Thread Size-8 codelet, called twice | [`fft_register_tile.cuh:82-94`](../src/fft_register_tile.cuh:82)、[`fft_thread_codelet.cuh:9-22`](../src/fft_thread_codelet.cuh:9) |
| suffix | old cuFFTDx direct 4096, 256 threads, EPT=16, 1 FFT/CTA | 相同 | runner passes `256,16`；[`fft_resident_experiment.cu:80-83`](../apps/fft_resident_experiment.cu:80)、dispatch [`cufftdx_fft.cu:1183-1211`](../src/cufftdx_fft.cu:1183) |
| suffix grid | 256 CTAs = `batch << local_log_n = 4*64` | 相同 | [`cufftdx_fft.cu:1130-1144`](../src/cufftdx_fft.cu:1130) |
| final writeback | suffix writes contiguous intermediate to `scratch`; separate 64x4096 transpose writes natural `output` | 相同 | `ContiguousOutput` branch [`cufftdx_fft.cu:676-716`](../src/cufftdx_fft.cu:676)、transpose dispatch [`cufftdx_fft.cu:1360-1367`](../src/cufftdx_fft.cu:1360) |
| changed field | native codelet only | cuFFTDx Thread codelet only | 报告称为 same schedule/path 下的 arithmetic substitution；[`report.md:51-66`](../results/fft_register_tile_20260912/report.md:51) |

这里的 prefix `EPT=8` 是 register tile 每线程持有的 `v[8]`，不是 runner
给旧 cuFFTDx suffix 传入的 `prefix_ept=16`。后者只在
`launch_cufftdx_online_reorder` 的旧 control 参数中出现；本例的手工 prefix
调用使用 `launch_prefix<6,16>`，因此真实 prefix 参数是 64-point、8 values
per lane、16 columns、128 threads。

### Prefix lane and shared ownership

对 prefix CTA，`row=threadIdx.x/16`，取值 0..7；`col=threadIdx.x%16`，取值
0..15。`transform=blockIdx.x*16+col`，低 12 bits 是 suffix-column index
`n2`，高位选择 batch；`base=(transform>>12)*batch_distance`，而本例
`batch_distance=N`、`element_stride=1`。每个 lane 先读取 8 个 global values：

```text
input[base + (j*8 + row)*4096 + n2],  j=0..7
```

第一次 8-point codelet 后，线程把 8 个值写到动态 shared tile 的
`tile[(j*8+row)*16+col]`，执行一次 `__syncthreads()`，再以
`tile[(row*8+j)*16+col]` 读回。这是同一 column 内的 8x8 shared exchange，
随后第二次 8-point codelet 使用同一组 registers。最后的 cross twiddle 由
recurrence 生成，并写入 global scratch：

```text
scratch[base + (row + 8*j)*4096 + n2],  j=0..7
```

因此 prefix 的 source-level shared launch argument 是
`64*16*sizeof(Complex32)=8192` bytes/CTA，shared payload 是 1024 values/CTA。
8192 bytes 是当前 launch 表达式，不是历史 binary 的 ptxas resource report。

### Suffix, boundary, and transpose

runner 给旧 online-reorder 函数传入 `local_log_n=6`、`suffix_threads=256`、
`suffix_ept=16`、`DirectBoundary::TiledTranspose`、`stage=2`。`stage=2` 会跳过
first pass，只执行 second pass；因此本例不是重复执行 prefix，而是：

```text
register prefix (64 = 2^6) -> global scratch
    -> cuFFTDx direct suffix (4096 = 2^12, one FFT/CTA)
    -> contiguous scratch -> 32x8-thread tiled transpose -> natural output
```

对于 N=2^18、B=4，suffix 每个 CTA 固定一个 `k1` 和一个 batch transform，
256 threads x 16 values 覆盖整个 4096-point FFT。direct suffix 的
`ContiguousOutput=true` 使其把结果写回 scratch 的连续 `(k1*4096+k2)` 布局，
而不是直接写 strided natural output。之后 transpose 的逻辑尺寸是
`rows=64, columns=4096`；每个 transform 有
`ceil(4096/32)*ceil(64/32)=128*2=256` tiles，共 1024 transpose CTAs，
block shape 为 `(32,8)`。当前源码的 static tile shape 是 `32x33`，但这只是
源码布局事实；历史编译器报告的 static shared allocation 仍为 unknown。

suffix 读路径在当前源码中还经过 `aptt_physical_index`；在默认的
`CUBUTTERFLY_APPT_LAYOUT=0` 或 `CUBUTTERFLY_APPT_STAGE_CONTRACT=0` 下它返回
logical index，但历史 metadata 没有保存这些 preprocessor flags。因此上面的
`k1*4096+k2` 是 suffix 的 logical contiguous write contract；若要声称历史
binary 的 physical scratch permutation，证据不足，应写 `unknown`。

## `U_s/T_s/U_d/T_d` 的解释边界

本文沿用正文定义：`U_s` 是物理上同时驻留的 stage positions，`T_s` 是复用
这些 stage service 的 stage-time folds；`U_d` 是每个 stage 同时展开的
butterfly/data groups，`T_d` 是复用同一 worker 的 data-time folds。对于完整
N=2^18 radix-2 图，数学域为 `L=18` stages、`D=N/2=131072` pair positions，
所以任何无越界映射都只需满足：

```text
1 <= U_s <= 18,       T_s >= ceil(18/U_s)
1 <= U_d <= 131072,   T_d >= ceil(131072/U_d)
```

等号表示没有 tail 的理想覆盖，不表示 GPU 一定以该数量的物理执行单元
并行运行。

对本例更小的 prefix hierarchy，可以安全报告以下“可解释范围”，而不要把
source loop 数量误写成物理流水：

* prefix 的逻辑 stage domain 是 6；每个 register codelet 是 3 个 radix-2
  stages，两个调用形成 `3+3`。这只确定了逻辑分组和 lane/register service
  的复用；不能直接写成 `U_s=3,T_s=2`，因为没有证明三个 stage positions
  在相应硬件层上空间展开。正文使用 `3+3` 描述分组，另行列出实际 ownership。
* 严格按“物理 stage service”定义时，当前证据只给出 `1 <= U_s <= 6`、
  `T_s >= ceil(6/U_s)` 的覆盖范围；没有 binary resource/role trace 能把
  `U_s` 选成唯一数值。特别是 `#pragma unroll`（[`fft_register_tile.cuh:30-47`](../src/fft_register_tile.cuh:30)）只要求编译器展开索引和 stage loop，不能证明 `U_s=6` 的物理流水。
* `Columns=16` 是 `pi_h` 的 lane/CTA ownership 参数，不应直接等同于
  `U_d`。每个 prefix CTA 暴露 16 个独立 64-point columns，即每 stage
  `16*(64/2)=512` 个 logical butterfly pair positions；按该 prefix packet
  观察，`1 <= U_d <= 512`、`T_d >= ceil(512/U_d)` 是可解释范围。若正文
  只按 independent columns 计数，则应写 `C=16`，而不是声称物理
  `U_d=16`。
* suffix 的 source-level facts 是每 CTA 一个 4096-point FFT、每 stage
  2048 logical pairs、无 CTA 内 serial FFT loop；cuFFTDx 内部 stage roles、
  actual `U_s/T_s/U_d/T_d` 和 scheduler waves 没有被该历史 metadata 冻结。
  `EPT=16` 是每线程 values，不是 `T_d=16`。

因此本实例可以支撑“3+3 register codelets、16 columns、一个 shared handoff、
4096 suffix、global transpose boundary”的映射陈述，但不能支撑“6-stage
physical pipeline”或一个未经校准的 hardware `U_d` 数值。

## 解析流量，与实测值分开

### 解析 logical traffic

`B*N=4*262144=1048576` complex32 points，每个 point 为 8 bytes。历史
transpose path 的全局 value movement 是：

| 路径 | full-array logical transfers |
|---|---:|
| prefix global input read + scratch write | 2 |
| suffix scratch read + contiguous scratch write | 2 |
| transpose scratch read + natural output write | 2 |
| 合计 | 6 |

所以解析全局值流量为 `6*B*N=6291456` complex values，即
`50331648` bytes = 48 MiB = 48 bytes/complex32 point。它是 logical
load/store accounting，不是实测 HBM traffic；L1/L2 hit、sector 合并、写分配、
twiddle loads 和编译器 spills 均未由该数值确定。报告也明确将旧 transpose
path 与新 grouped path 对比为 48 versus 32 bytes/point；[`report.md:68-73`](../results/fft_register_tile_20260912/report.md:68)。

可由当前 source 计算但不应冒充历史 profiler 计数的 shared payload 还有：
prefix tile 的一次 write+read 为 `2*1024` values/CTA，1024 CTAs 合计 16 MiB
logical shared value traffic；transpose tile payload 同样为 16 MiB。4096
cuFFTDx suffix 的动态 shared 使用量是 `FFT::shared_memory_size`，历史数值
unknown。

### 实测 timing and correctness

| mapping | N=2^18, B=4 median kernel ms/repeat | 相对 `register-c16` |
|---|---:|---:|
| `register-c16` native prefix + old suffix/transpose | 0.035031 | 1.000x |
| `dx-thread-c16` cuFFTDx Thread prefix + same suffix/transpose | 0.0336486 | 1.041x, 3.946% lower time |
| cuFFT reference | 0.030761 | reference only |

数值来自 [`paired_summary.json:3-29`](../results/fft_register_tile_20260912/accurate_core_control/paired_summary.json:3)，并与报告的控制段 [`report.md:60-65`](../results/fft_register_tile_20260912/report.md:60) 一致。每个 cell 是 5 个 paired rounds、每轮 100 次；runner 在 event 前执行
verification 和 warmup，event 包围 100 次 kernel execution，再除以 repeat，
见 [`fft_resident_experiment.cu:167-179`](../apps/fft_resident_experiment.cu:167)。
N=2^18 的两个目标行都记录 `relative_max_error=4.04965e-07`，见
[`verify-log18-batch4.log:6`](../results/fft_register_tile_20260912/accurate_core_control/verify-log18-batch4.log:6) 和
[`verify-log18-batch4.log:13`](../results/fft_register_tile_20260912/accurate_core_control/verify-log18-batch4.log:13)；报告说明原 absolute-error check 也同时执行。这里的差异是完整 prefix+suffix+transpose 的 end-to-end event，不是 prefix arithmetic 单独计时。

## 编译资源与历史/当前源码边界

`accurate_core_control/paired_metadata.json` 只记录 binary hash、设备、试验
次数和命令，没有 `registers/thread`、register stack/local bytes、实际
static/dynamic shared bytes、occupancy 或 spills。因此对历史 target binary，
以下字段均应写 `unknown`：

```text
registers/thread       unknown
register stack/local   unknown
prefix compiled shared unknown (source launch argument is 8192 B)
suffix compiled shared unknown (source symbol is FFT::shared_memory_size)
transpose compiled resource unknown (source tile shape is 32x33)
occupancy / resident CTAs / spills  unknown
```

现有的 `ncu_register*` 记录 profile 的是 `<10,8>` register prefix 和 grouped
suffix；`ncu_old_xor*` 记录的是另一条 XOR prefix + 4096 suffix；它们不是本例
的 N=2^18、`<6,16>` native/Thread pair，不能移植其 REG 或 shared 数值。当前
本地 executable 的 SHA 也不等于 metadata 中的 target hash，因此不使用当前
`cuobjdump` 输出替代历史资源记录。

源码也存在需要显式标注的时间边界：报告记载早期 prototype 使用
`__sincosf(2*pi*x)`，之后才改为 pi-scaled `sincospif` 并重新锚定 recurrence；
[`report.md:95-101`](../results/fft_register_tile_20260912/report.md:95)。当前
`fft_register_tile.cuh` 已是 `sincospif` 版本；[`fft_register_tile.cuh:13-21`](../src/fft_register_tile.cuh:13)。所以当前源码足以解释上述
schedule、index 和 ownership，但 metadata 没有 historical source snapshot，
不能用当前 source 推定 target binary 的具体 instruction、register allocation
或 resource 数值。

另一个不能混淆的变化是当前公共 register dispatch：它要求
`DirectBoundary::Strided` 并绑定 grouped suffix，见
[`fft_register_dispatch.cu:71-80`](../src/fft_register_dispatch.cu:71) 和
[`fft_register_dispatch.cu:125-143`](../src/fft_register_dispatch.cu:125)；本例
则是 runner 手工调用 `DirectBoundary::TiledTranspose`、`stage=2` 的旧 control
路径。最终 API 的 1024 suffix 结果不应回填到本例。

## 补充：四因子的可直接追溯实例

正文另用独立的 FWHT composition cohort 展示四因子的实际描述，记录位于
[`public_completed_services.json`](../results/staged_migration_20260916/a100-80gb-composition/public_completed_services.json)
第 26 个 row 的 `physical_groups[0]`。来源哈希及该候选的服务/时间已冻结在
[`method_cases.json`](evidence/method_cases.json)。N=4096、B=64、tile batch=8，
分段为 8+4；第一段描述器为：

```
stage_space=1, stage_time=8, data_space=128, data_time=1
threads=128, stage_count=8, live_shared_bytes=2048
grid_ctas=128, work_blocks=128
```

第一段组件有 2^8=256 个值，每 stage 有 128 对。在 CTA assignment 层，
四因子为 (1,8,128,1)：128 个 pair positions 分给线程，复用同一 stage position
处理八个阶段，shared 保留中间量。每个 transform 有 4096/256=16 个组件，
tile batch=8 得到 128 个 CTAs。warp 发射可以跨多个周期，因而不能把该 assignment
读成每周期完成 128 次硬件 pair 运算。这一描述不依赖前面 FFT 历史 binary
缺失的资源元数据，也不把两个 cohort 的速度混合。

## 最小 LaTeX 表格建议

下表只放主例 N=2^18、B=4；正文脚注应注明两行共用 64-point prefix ownership、
4096 suffix/transpose，时间为 5x100 CUDA-event median，并将编译资源写成
``unknown``。

```latex
\begin{table}[t]
  \centering
  \caption{A100 80GB, FP32 complex FFT, $N=2^{18}$, batch 4.}
  \label{tab:a100-register-thread-case}
  \begin{tabular}{lccccc}
    \hline
    mapping & prefix & columns & prefix CTA & suffix & median (ms) \\
    \hline
    native register & 64 ($3+3$) & 16 & 128 threads, 8/lane & 4096, 1 FFT/CTA & 0.035031 \\
    cuFFTDx Thread & 64 ($3+3$) & 16 & 128 threads, 8/lane & 4096, 1 FFT/CTA & 0.0336486 \\
    \hline
  \end{tabular}
\end{table}
```

表格中的 `$3+3$` 是两个 8-point codelet invocation 的逻辑 stage grouping，
不是六级物理流水；`columns=16` 也只是 CTA ownership 参数。全局 48 bytes/
complex32 point 的解析流量和历史编译资源 unknown 应留在正文说明或脚注，
不要把它们写成 NCU/HBM 或 ptxas 实测值。
