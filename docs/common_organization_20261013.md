# 非核心机制的跨算子推广

版本 `0.9.3.11-common-organization-20261013`。本轮先完成实现，再集中验证，
不在各补丁之间启动性能调优或扩展横评矩阵。

## 共用实现

FFT、NTT、FWHT、subset zeta、superset zeta 和 structured 2x2 共用组织
候选工厂。算子保留自己的算术 trait、精度、系数及核心支持域，组织由实际
lowering 的能力决定。公共映射能显式指定合法组合，自动搜索使用有限种子，
不会将有限种子清单解释为完整理论空间。

| 组织选择 | 六类算子的共用路径 | 参数及边界 |
| --- | --- | --- |
| 逻辑分段与物理融合 | SharedIterative | 任意正阶段划分；相邻 resident-fused 边界合并为物理组，单组受设备容量约束 |
| 组内子图驻留 | SharedIterative JIT | 每个物理组独立的 local_stage_partitions；现有 leaf 支持 1–5 阶段 |
| 共享布局 | SharedIterative | linear、writer-aligned；由同一索引模板消费 |
| CTA 线程宽度 | SharedIterative、xD resident | 设备允许的正 warp 整数倍；自动种子 32/64/128/256/512/1024 |
| bulk batch 时间展开 | SharedIterative | stage_overlap=false 时 batch_tile_count=1..4；一个 CTA 顺序处理独立 batch，复用共享存储 |
| 跨组流水 | 共用 BatchPipeline | stage_overlap=true 时 batch_tile_count 为流水块大小；每个物理边界双槽，全融合单组无跨组流水 |
| 多维驻留与轴顺序 | xD resident | rank 2..8、独立轴任意排列；系数及逆运算仍绑定原轴，整个状态须装入 CTA |
| 正逆、输入输出顺序与布局 | 既有语义适配 + 共用组织 | 不修改请求语义；必需的边界转换完整计入调用 |
| 离线编译和校准 | 共用 mapping 投影 | 编译、IR、服务曲线按融合后的物理组和实际 batch_space 一致建模 |

`batch_tile_count` 的两种组织语义明确区分。bulk 模式下，一个物理组的 CTA
数为 `ceil(B / batch_tile_count) * 2^(logN - group_stages)`，共享内存不随该值
增长。overlap 模式下，它表示每个流水块的 batch 数，块内 CTA 按 batch 展开，
边界双槽存储随该值增长。这两种实现不能使用同一个服务曲线身份。

NTT 的逻辑分段与实际 launch 分段此前只有部分后端解耦。现在 shared 路径
保留公共逻辑 mapping，而执行、workspace、JIT 和离线预编译统一使用融合后的
物理分段。overlap 只把物理组间边界改为 ring，不覆盖驻留边界。

xD resident 不再按 rank≤3 或两个线程常数限制公共选择。自动搜索保留旧轴序
作为种子，并使用有界排列/线程组合；显式 mapping 可指定任意合法排列和设备
允许的线程宽度。xD 缓存 namespace 更新为 v22，实际库源码指纹另外区分版本。

## 生成、选择和校准一致性

- `shared_organization_candidates.hpp` 是共用组织库存，两个配置类型仅负责字段适配。
- `research_compile_requests.py` 与运行时一样合并 NTT 和浮点蝶形的逻辑边界，
  保留 bulk batch width，overlap 的独立组编译使用 batch width=1。
- `calibration_space.py` 以 serialized mapping 为权威，旧顶层别名不能覆盖真实
  线程、布局、分段或流水参数；预测时不修改输入 mapping。
- 独立服务校准保留 bulk batch_space，去除 overlap 时仅去除流水块宽度。
  batch_space 是实现形状，batch_time 是工作量，不混为同一个校准维度。
- 抽象空间默认列出全部声明的合法核心，状态仍是 abstract；不会因为枚举出了
  参数就标记为可执行或已编译。

## 保留的真实边界

本轮没有增加新算术核心。SharedIterative 仍使用现有 radix2 trait；NTT 使用
Shoup；低精度舍入规则、结构化矩阵和模数限制保持不变。其他局部核心的
radix、寄存器 items、coefficient reuse、warp-role、APPT 专用 packet/ring
调度仍属于其内部契约，不能只复制字段就声称已移植。

共用 shared 输出已经归属最终写线程，不需要 FFT grouped suffix 的
`exchange_chunks` 所描述的寄存器所有权交换。因此非零该字段仍明确拒绝，
而不是提供一个无效开关。FFT 的 cross-twiddle、实数 Hermitian 边界，以及
NTT 的模乘参数属于数值语义/核心特征，不强行添加到无对应运算的算子。

当前共用路径仍有资源与实现边界，不能据此宣称所有 kernel 与所有组织的
笛卡尔积可执行，也不能宣称性能已提高。新的组织选择需要当前构建下重新
测量与校准；旧版性能数据不改标签。

## 集中验证

构建与结果保存于 `results/common_organization_20261013/`；修改前共享库另存
`previous/libcuntt.so`。验证入口：

- `shared_organization_contract`：有限库存、能力过滤和旧候选保留。
- `shared_mechanism_correctness`：六算子 bulk/fusion/overlap、mapping 与实际 IR。
- `shared_resident_mechanism_correctness`：同六算子的 JIT local leaves 与融合/流水组合。
- `shared_wide_mechanism_correctness`：FP64 与 uint64 的 bulk/fusion/overlap。
- `xd_resident_rejection_contract`：非法线程、轴序和 rank 的诊断 mapping 可安全读取。
- `tests/check_xd_common_organization.py`：六算子 rank 2/4/8、正逆、32/96/512
  线程、非默认轴序、精确 mapping 回放、变输入 CUDA Graph。
- Python 回归：抽象空间、离线编译、校准投影与服务模型身份。

整数/NTT 精确比较，浮点使用同精度 CPU 参考并报告误差分布，暂不增加
用户未批准的容差。完成情况以本轮结果文件为准，不以测试代码存在作为通过证据。

## 本轮结果

最终库 SHA256 为 `d5fa2a1eb82721a2ad3dbfa65888419e4487765de31b7e124fff6928bf5d4742`。
构建目录为 `build-contract-20261013`，相同哈希的归档在
`results/common_organization_20261013/final_library/libcuntt.so`。两张卡均为
SM80 的 A100 80GB PCIe，运行时分别持有独占进程检查。结果归档：

| 检查 | 结果 | 证据，相对 results/common_organization_20261013 |
| --- | --- | --- |
| Python 组织、编译和模型回归 | 137 passed，75 subtests passed | cpu_tests.log |
| C++ 映射、候选、IR、公共选择及执行 | 8/8 个 CTest 通过 | shared_final_guard/release/runner.log |
| shared 实际执行 | 36 常规 + 12 JIT 驻留 + 10 宽精度，共 58 组合；8 个非法组合正确拒绝 | 同上 |
| xD 公共执行 | 108 组合；36 个重复轴请求正确拒绝 | xd_final_guard/release/xd.json、runner.log |
| xD 无效计划诊断 | 4 个非法配置可安全读取诊断 | shared_final_guard/release/runner.log |
| JIT 准备 | 12 个精确 SM80 请求预编译成功 | precompile_results.json |

Python CPU 回归中的另外 30 项是未设置 GPU 二进制入口而跳过的既有检查，
不计入通过数。本轮 GPU 结果来自表中显式执行的两组测试，不借用旧构建数据。
xD 使用 `GPU-2971e500-d3f6-d7e6-e5f2-cb67a06fff02`，shared 和小型计时使用
`GPU-0eae5f07-33de-321b-70a9-ffeae49309e6`。

所有整数/NTT 对照精确一致。浮点输出均有限，shared FP32 最大相对 L2 为
`1.30e-7`，FP64 为 `2.18e-16`。xD 中 FFT、FWHT、structured 的最大相对 L2
分别约 `1.10e-7`、`1.11e-7`、`1.36e-7`。这些是误差报告，未新设数值验收阈值。
xD 还验证了完整 mapping 精确重放、输入不被修改、Graph 对变输入及恢复输入
的输出与 eager 逐位一致。

集中验证曾在 `mixed_dataflow` 抓到流水 `batch_time` 错误计为 B 的回归。
现已修复为 `ceil(B / batch_tile_count)`，与真实流水提交次数一致，并加入实际
执行组断言。`cpu_contracts.log` 保留这次失败；上述 final 结果均在修复后重跑。
早期 `xd_guard`、`shared_guard`、`library` 也仅是中间构建记录，不能混入最终报告。

## 固定核心小型计时

固定同一算术核心及组织条件，仅比较 shared bulk width=1 与 4：N=65536、B=32、
分段 `[8,8]`、128 线程、writer-aligned，预热 10 次、平均 100 次，使用 CUDA
事件记录执行时间，编译/初始化/CPU 检查不计入。下面是一次固定样例计时，
不是调优后的库横评，也不是跨负载统计结论。

| 算子 | 精度 | bulk=1，μs | bulk=4，μs |
| --- | --- | ---: | ---: |
| FFT | complex FP32 | 180.767 | 193.300 |
| NTT | uint32，p=998244353 | 196.413 | 208.097 |
| FWHT | FP32 | 129.475 | 133.386 |
| subset zeta | uint32 | 131.297 | 135.660 |
| superset zeta | uint32 | 130.253 | 134.758 |
| structured 2x2 | FP32 | 137.554 | 145.121 |

这一组 bulk=4 慢约 3%–7%，不能默认启用；公共库存保留 bulk=1。
原始命令、完整 mapping、构建指纹及计时位于
`bulk_smoke_guard/release/measurements.json`，复现脚本为
`results/common_organization_20261013/measure_bulk_smoke.py`。本轮扩展证明新增选择可以被公共框架消费，尚未完成
新组织轴的服务曲线重标定和综合基线横评，不宣称新版本整体更快。

## 后续代码审查

[2026-10-23 NTT 专用机制与公共组织模板审查](ntt_common_mechanisms_20261023.md)
区分公共表征、候选覆盖与真实共享执行，并记录 APPT 变体的迁移缺口。
该审查未新增 GPU 测量，不改变本文原构建的实验数据与结论。
