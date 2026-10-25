# 核心模板参数贯通与回归记录

版本：`0.9.3.10-unified-template-parameter-20261013`。

本轮将核心策略吸收、参数贯通和数据流收益的判断标准写入
[研究原则](research_principles.md)，并完成一轮共用链路审查和有边界的修复。
这不是全算子全面性能测试完成的声明。跨算子的已有实现、公共可达性和剩余
缺口见 [模板审查](template_parameter_audit_20261013.md)。

## 原则如何落到工程

局部核心弱时，应将可复用的算术、线程协作、系数供给或交换策略纳入模板及
参数选择。调用外部 provider 可以先提供有效核心，但不能等同于已经完成其
内部策略移植。新增模板沿以下链路接入，不为应用名称固化赢家：

`支持域/参数 → 合法性 → 生成或分派 → 公共候选 → 实际 mapping → 缓存/校准身份 → 实测`

复制、重排、写回和独立缩放的减少，若由我们的组织方式产生且完整语义与
输入输出责任一致，就是方法收益。固定核心的组织消融可用这些收益说明贡献，
不要求每次局部算术指令本身也更快。换核心和换组织的收益仍分别记录。

## 已修改的共用路径

| 路径 | 原问题 | 本轮修复 |
| --- | --- | --- |
| xD FFT 核心参数 | 公共 JSON、lowering 和搜索分别维护参数范围 | 新增 [xd_axis_resources.hpp](../src/xd_axis_resources.hpp)，共享资源结构、校验和候选工厂 |
| 已有模板的可达性 | packed exchange=4、N=64 的 local 32-line 配置未进入按轴搜索 | 补入有限编译候选，原候选及顺序保留；构造时继续核查实际尺寸和设备支持 |
| 参数解析 | 无符号窄化可能把超范围输入变成另一个合法值；provider 不消费的参数可能被忽略 | 窄化前检查类型与范围；VkFFT 的非默认资源参数明确拒绝 |
| local 支持域 | N=128/cap=32 可能到执行期才发现不可用 | 在计划构造时拒绝，搜索记录不可用原因 |
| 各浮点蝶形的平台规则 | 缺少 output_scale/output_rounding，规则与 registry 语义字段不一致 | 补齐实际缩放和最终舍入字段，验证精确命中及两种不匹配 |
| 缓存 | 旧资源契约的映射可能复用 | 使用 v21 xD namespace，标准构建重新生成源码指纹 |
| FWHT/structured 核心形状 | `tile_threads` 可能只是候选记录，模板仍固定按 N 派生线程 | 新增 `register_threads=0` legacy/非零显式语义；生成器按 shape 过滤并实例化真实模板，launcher、mapping、校准身份和搜索候选统一消费 |
| NTT 专用核心 | 既有 homogeneous/static-IO/dataflow 核心只有零散 legacy prior | 通过有限工厂加入 word64 的 16 个、word32 的最多 26 个明确支持域候选到公共 NTT inventory；保留 word/modulus/方向/order/batch 语义，并接入带 ring/role descriptor 的基础 APPT register-tail |

 xD 候选数为 70，启用 VkFFT 时为 71；FWHT/structured 的线程候选由每个 logN
和真实生成清单决定。这些都是有限库存，既不是完整理论空间，也不是
每个轴都能执行的配置数。各配置必须经过 shape/device 检查。新增候选不保证被
选中，完整调用测量仍负责比较。

 NTT 专用模板工厂在 word64 的支持域生成 16 个候选，在 word32 的支持域最多生成
 26 个候选（其中包含额外的 warp128/static-IO 变体和基础 APPT register-tail）；这些候选只在 logN=20、Shoup
 乘法和自然输入/输出顺序下生成，其他语义组合返回明确的不支持原因。

## 构建与功能验证

标准 CMake 构建目录为 `build-contract-20261013`，SM80，shared library，开启
cuFFTDx 和 VkFFT。没有覆盖此前的测量二进制。

库 SHA256：`2e9a873582de9c198a19901f727891ba99f19097929df34bf0dacd9987b2b077`。

| 检查 | 结果与证据 |
| --- | --- |
| CPU 参数、平台 Python、应用缓存回归 | 8 项通过；`test_xd_axis_resource_contract.py`、`test_platform_tuning_python.py`、`test_plan_cache_adapters.py` |
| 平台规则 C++ 回归 | 新标准构建的 `platform_tuning_contract` 通过；验证 output_scale、最终舍入的精确匹配与不匹配；不构造 GPU plan |
| 原公共 VkFFT 路径 | 38 个配置通过结构性检查，3 个非法形状拒绝；[记录](../results/core_contract_validation_20261013/public.json) |
| 混合轴资源路径 | 22 个配置通过，2 个不支持组合按预期拒绝；[记录](../results/core_contract_validation_20261013/mixed_guard.json) |
| 新增资源公共回放 | 18 个 FP32/FP64、C2C/R2C/C2R、正逆及 exchange=4 配置通过，6 个非法配置正确拒绝；[记录](../results/core_contract_validation_20261013/resources_guard/release/resources.json) |
| FWHT/structured 线程模板回归 | 20 个真实 generated 线程变体在 A100 GPU2 上逐项通过算术结果校验；[运行记录](../results/core_contract_validation_20261013/register_threads_guard/release/runner.log) |
| 六类公共映射回放 | FFT、NTT、FWHT、subset、superset、structured 的 C/C++/公开 API 映射和 replay 通过；[运行记录](../results/core_contract_validation_20261013/public_planner_guard_v2/release/runner.log) |
| NTT 专用模板工厂 | word64 支持域生成 16 个带完整 stage/boundary/mapping 描述的候选；不支持的 logN、模数、乘法和顺序组合不生成候选；CPU contract 通过 |
| NTT 专用模板公共回放 | word64 16、word32 26 个模板，正向/逆向、逐批次精确校验共 84/84 通过；除平台规则补全的 cta weight 外，实际 mapping 与请求一致；[运行记录](../results/core_contract_validation_20261013/ntt_templates_guard_v3/release/runner.log) |

验证 GPU 为 A100 80GB，UUID `GPU-0eae5f07-33de-321b-70a9-ffeae49309e6`。
新增资源检查使用两组确定性随机输入，Graph 按输入 0→1→0 回放；核对输入不被
修改、Graph 与 eager 一致、公开映射精确回放。CPU 参考使用同精度 SciPy FFT。

继续遵守浮点误差只报告、暂不设通过阈值的约定。上述“通过”表示执行、有限性、
映射及 Graph 契约通过，不表示已经确认数值容限。新增 18 配置的最大相对 L2
误差为 FP32 `1.93e-7`、FP64 `3.50e-16`；各输入的绝对误差分位数保存在记录中。

可重复的新增资源验证命令：

```bash
/home/wt/yes/envs/cubutterfly/bin/python scripts/run_isolated_gpu.py \
  --gpu-uuid GPU-0eae5f07-33de-321b-70a9-ffeae49309e6 \
  --report-dir results/core_contract_validation_20261013/resources_guard \
  --until 2026-10-13T17:00:00+08:00 --idle-seconds 2 -- \
  /home/wt/yes/envs/cubutterfly/bin/python tests/check_xd_core_contract.py \
  --library build-contract-20261013/libcuntt.so --output resources.json
```

复测时使用新的报告目录和明确的等待截止时间，不覆盖已发布数据。

## 完整应用回归

本轮是 5 个语义请求、每个 2 次独立进程，共 10 次测量：Poisson 的
1024²/2048² × B=4/16，以及相位相关 2048²、B=16。未重跑图像滤波，也未扩展
为全算子矩阵。早期进度信息中的“12 次”包含了未在本轮执行的 2 次滤波；以
本记录的 10 次和原始测量文件为准。

硬件为另一张 A100 80GB，UUID `GPU-2971e500-d3f6-d7e6-e5f2-cb67a06fff02`。
完整 CUDA Graph 调用计入输入更新和输出责任，编译、搜索、初始化和 capture
在稳态计时之外。对照为同进程 PyTorch/cuFFT 和 CuPy/cuFFT；结果范围是两次
试验的加速比范围，不是置信区间。

| 场景 | 类型 | 对照 | 加速比 |
| --- | --- | --- | --- |
| Poisson 1024²，B=4 | 2D 实数 FFT，正/逆 | PyTorch / CuPy | 1.049–1.072× / 1.083–1.092× |
| Poisson 1024²，B=16 | 2D 实数 FFT，正/逆 | PyTorch / CuPy | 1.178–1.187× / 1.176–1.182× |
| Poisson 2048²，B=4 | 2D 实数 FFT，正/逆 | PyTorch / CuPy | 1.153–1.160× / 1.148–1.153× |
| Poisson 2048²，B=16 | 2D 实数 FFT，正/逆 | PyTorch / CuPy | 1.172–1.188× / 1.207–1.208× |
| 相位相关 2048²，B=16 | 2D 实数 FFT，正/逆 | PyTorch / CuPy | 1.021–1.031× / 2.336–2.509× |

这些自动选中方案仍组合 VkFFT 正向轴与 cuFFTDx 逆向轴/缩放。它们证明本次
参数修复后已有应用收益仍可获得，不单独证明已完成外部核心策略移植或固定核心
归因。相对上一版，Poisson 延迟变化约在 0.4% 内；相位相关延迟约增加 1–2%，
未带来新的普遍加速。仅两次试验不将小幅差异断言为稳定的性能回退或提升。

完整测量及匹配旧版对比见
[本轮汇总](../results/e2e_core_contract_20261013/summary.md)。

## 保留的工程缺口

基础 NTT APPT register-tail 已进入公共自动候选，并由 ring、CTA 和 producer/tail/writer
role mapping 契约约束；APPT warp、radix8、grouped 和 resident 变体仍需要独立的
liveness 证据。FWHT/structured 的 `register_threads` 已贯通，
但 `items=4` 和短 FP16 provider 支持域仍是明确边界；subset/superset 尚缺第二种核心。共用 resident
模板也不是所有核心都能任意嵌入。后续应逐个补齐真实模板、支持域、公共可达性
和匹配验证，不以文档抽象、外部 provider 或重新校准代替这些实现。
