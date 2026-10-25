# 端到端性能路径诊断（2026-10-21）

## 结论

单算子加速比不能直接等同于应用加速比。完整调用的时间可以写成

\[
T_{app}=T_{butterfly}+T_{other}+T_{boundary}+T_{submit}.
\]

cuButterfly 主要改变 `T_butterfly`。当应用中蝶形计算占比不高，或者请求很短时，固定的布局转换、Python/FFI 边界和 kernel 提交时间会限制端到端收益。这是 Amdahl 效应与执行路径问题，不是混合数据流本身失效。

当前数据支持这一判断：

| 场景 | 模式 | cuButterfly / 对照 | 主要解释 |
| --- | --- | ---: | --- |
| FNO Darcy 64²/128²，B=4 | eager | 0.9554x / 0.9574x | 4 次 rFFT 与 4 次 irFFT 的 Python/FFI 和提交开销占比高 |
| FNO Darcy 64²/128²，B=4 | CUDA Graph | 1.0299x / 1.0564x | Graph 消除了重复提交边界，算子收益被传递到应用 |
| 2D Poisson 2048²，B=4 | CUDA Graph | 1.159x | 自适应 xD 计划减少阶段和边界 kernel |
| QuaRot FWHT N=32/64 | eager | 1.008x–1.016x | 短请求中 decode 和非蝶形工作占主导 |
| Image filter，B=16 | eager | 约 1.067x（CuPy），约 1.090x（Torch） | 批量较大，蝶形部分占比更高 |
| CKKS Graph，N=16384/32768/65536，B=1 | CUDA Graph | 1.065x / 1.067x / 1.104x | 大 N 下 NTT 占比提高，执行边界被 Graph 隐藏 |
| 2D FFT 1024²，B=4 | xD Graph steady state | 约 1.00x vs cuFFT | 完整 C2C cuFFT 计划已进入同一搜索空间；该点不再因缺少完整核心而落后，轴级核心仍由现场测量决定 |

因此，FNO eager 和短 FWHT 是固定开销受限的场景；Poisson、Image filter 和 CKKS 说明统一框架在蝶形占比较高或批量较大的负载上可以传递算子收益；1024² FFT 则说明仍有少数尺寸需要提升计算核心本身。

## 已完成的公共路径修复

计划创建时已经将轴计划绑定到句柄 stream。此前每次 `cubutterflyExecute` 都会对所有轴计划重复调用 `set_stream`，这会在 rank-2 和通用 xD 路径上增加主机侧工作。现在 `cubutterflyPlan` 保存最近一次传播的 stream，只有句柄 stream 变化时才递归更新 `xd_selected`、数值适配子计划、边界子计划和任意轴子计划。

这项修改只改变 host-side 绑定缓存，不改变数据布局、kernel 顺序、数值语义或搜索空间。首次执行、切换 stream 和复合子计划仍会正确传播 stream。`cuntt` 目标和 xD 布局测试已重新构建，两个 xD 正确性测试通过；前端回归也通过。性能仍需在同一端到端基准上重新测量，不能用正确性测试替代性能结论。

一次 A100 80GB 的 1024²、B=4 smoke 测试出现了 `workload-profile-miss` 日志。进一步读取 selection reason 后确认，该日志表示预编译 profile 未覆盖该精确键，并不表示跳过了运行时搜索：默认路径实际测量了 55 个完整候选。最优 cuButterfly 候选约为 0.1186 ms，cuFFT 约为 0.099--0.107 ms，因此该点应归入计算核心效率短板，不能通过修改 profile 标签或重复校准掩盖。

随后将已有的 `DirectFftPlan` 作为 `cufft-whole` 加入相同的 xD 候选集合。它只覆盖 packed、自然布局、标准 rank-2 C2C complex FFT，并把完整 cuFFT 计划的时间纳入同一个 selector。相同 smoke 负载现在选择 `xd[cufft-whole]`，实测 `1.002x` cuFFT，输出误差为零。

本轮又补齐了 packed、自然布局的 rank-2/3 R2C/C2R 完整 cuFFT 候选。它与轴级实数边界候选使用同一套完整计划搜索和缓存键，C2R 额外使用计划级 scratch 保持用户频谱输入不被 cuFFT 覆写，并把逆变换归一化融合进 scratch copy，避免再遍历一次实数输出。A100 FP32 的 16²、64²、128²、B=2 正确性验证通过，rank-3 `4×8×16` 也通过；在固定输入输出缓冲、相同 `norm='backward'` 的 64²、B=4 smoke 中，自动路径的 R2C 约为 PyTorch 的 0.99--1.01x、C2R 约为 1.5x，完整 cuFFT 候选的 R2C 约为 1.1--1.2x、C2R 约为 1.3--1.4x。旧的 1.52x/1.44x 数字来自不同归一化和缓冲协议，已不再作为结论。这项候选只在语义契约允许时出现，非连续布局、顺序转换和自定义边界仍走原有适配路径。

本轮还修正了 compact-real 调优的计时协议。cuFFT 的实数边界可能提交辅助工作，单独记录调优 stream 的 event 会低估完整 R2C/C2R 计划。现在候选排序使用设备同步包围的完整调用，并在自动选择完成后用原始调用 stream 对暂定赢家和完整 cuFFT 候选做一次复核。复核结果写入 `selection_reason`，只影响安装期的选择，不把同步带入运行时。这样轴级核心仍可在确实更快的尺寸上胜出，完整 cuFFT 也能在适合的尺寸上被统一搜索路径选中。

通用 rank-N composite 路径的轴顺序、置换向量和布局现在在计划创建期缓存，执行期不再反复分配 `vector` 或查找轴位置。该改动只作用于确定性的 host-side 调度资料，保留原有输入输出、stride、workspace 和 stream 语义。

Python 前端和 E13 adapter 还缓存了 ctypes 的 ABI 函数指针，重复执行不再为每次 FFT 调用动态解析 `cubutterflyExecute`。这只削减 host-side FFI 开销，不改变 kernel、数据布局或异步语义；公共前端回归为 46 个测试通过，adapter 回归保持通过。

针对固定计划的重复调用，C ABI 还提供了 `cubutterflyExecuteSteady`。它在首次提交或 stream 变化时自动回到完整 dispatcher，稳定 stream 上直接提交已经绑定的 xD/real 计划。16²--32² 的小批量实测端到端提交时间下降约 2--3%，而 64²、B=4 的结果保持在同一测量噪声范围内。该接口不删除必要的 pack、permute、scale 或 real-boundary kernel，因此对照仍包含完整算子语义和必需边界成本。

## 按风险排序的后续工程优化

1. **消除可证明的恒等置换。** 只有输入轴连续、目标轴顺序相同且公共布局契约允许时，才跳过 `launch_permute_nd` 或恢复置换。任意布局、跨 batch stride 和实数边界仍必须保留转换，并计入 selector 的 cost。
2. **提供长期存在的框架入口。** PyTorch 侧使用长期存在的 custom op 或批量 adapter，减少 Python monkeypatch 与 ctypes 往返。CUDA Graph 是当前可用的低风险路径，已经在 FNO 上显示出从略慢到超过对照的变化。
3. **把边界成本纳入搜索目标。** selector 不能只比较蝶形 kernel 时间，还要使用 `mapping_cost_json()` 中的 permutation、restore、real-boundary 和 scale launch 计数，将完整执行计划的边界成本加入总 cost。
4. **按负载占比组织报告。** 单算子、机制消融和端到端分别报告，不用端到端结果反推理论优劣；短 N、低 batch 只作为固定开销敏感区，主结论使用中大型 N/B 和 Graph/eager 对照。

## 不应采用的修复

- 不通过缩小 batch、升高精度或替换算子来掩盖缺失支持。
- 不删除所有 pack/permute；非连续布局和真实边界转换是语义的一部分。
- 不把一次特定 GPU、特定应用的最优参数写死到统一框架。硬件特调应通过平台 profile 和搜索接口进入 cost model。
