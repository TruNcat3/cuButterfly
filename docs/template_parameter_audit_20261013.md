# 跨算子模板与参数贯通审查

审查日期：2026-10-13。范围：rank-one FFT/NTT/FWHT/subset/superset/structured 的共用链路及非 FFT 算子核心。本文记录静态审查和共用平台字段修复；同期 xD 参数修复、GPU 验证及应用测量见 [本轮实施记录](core_template_contract_20261013.md)。依据 [研究原则](research_principles.md) 和 [设计空间](butterfly_design_space.md)，分别记录表达、生成和公共选择状态。

后续更新：shared 组织库存已合并为六算子共用工厂，抽象枚举不再默认只取第一个
合法核心；bulk batch、NTT 逻辑融合、校准物理投影和 xD resident 的公共选择也已
扩展。下面保留审查时的发现；这些问题的当前处理范围和验证证据见
[非核心机制推广记录](common_organization_20261013.md)。专用核心的支持域缺口仍保留。

## 可核查的共用链路

| 对象 | 已实现入口及消费路径 | 当前边界 |
| --- | --- | --- |
| G 语义 | [ButterflyConfig](../include/cuntt/butterfly.hpp) 的算子、precision/accumulation、方向、stride、stage_matrices、output_scale/final_storage_rounding；[PlanConfig](../include/cuntt/ntt.hpp) 的 modulus、word_bits、方向及顺序；[make_operator_graph](../src/mixed_dataflow.cpp) 有两类 config 重载 | 结构化矩阵是语义输入，不可作为寻优时改变的参数；NTT 与其他算子仍使用不同 config 类型 |
| A 组织 | 两类 config 都有 stage_partition、local_stage_partitions、exchange_chunks、stage_overlap/batch_tile_count；[project_execution_groups](../src/mixed_dataflow.cpp) 投影物理执行组；[shared_resident_kernel](../src/shared_resident.cuh) 执行 resident leaves | shared resident leaf 为 1–5 阶段；已实现空间并非抽象 Us/Ts/Ud/Td 全组合 |
| P 核心 | [operator_traits.cuh](../src/operator_traits.cuh) 的 apply/apply_lane/finalize；[NttStageOperatorT](../src/ntt_traits.cuh)；FWHT/structured 专用 [RegisterFwhtTraits](../src/register_fwht.cuh)；FFT 的 FftCore/segment_mappings | 共同 trait 是蝶形算术接口，尚不是任意外部 codelet 的统一 provider 接口 |
| L 布局 | [shared_iterative_kernel](../src/shared_iterative.cuh) 的 writer_aligned 与 iterative_layout_index；[BooleanZetaOperator](../src/operator_traits.cuh) 区分左右更新；NTT 的 terminal bit-reversed output | 不应把抽象空间列出的 Stockham/任意布局视为各算子已有 lowering |
| F 实现 | shared kernel 模板含 StaticLogN/First/Stages/Layout；[batch_pipeline.cuh](../src/batch_pipeline.cuh) 负责物理组间双槽、事件与 stream；threads 被实际 launcher 消费 | CUDA block、双缓冲、交接是可执行机制；各专用核心资源和支持域仍独立限制 |
| Q 生成选择 | [render_shared](../scripts/compile_module.py) 为六算子生成 V2 module；[prepare_shared_module](../src/jit_module.cpp) 两类重载；[portable_butterfly_candidates/portable_ntt_candidates](../src/plan_registry.cpp) → [select_butterfly_mapping/select_ntt_mapping](../src/mapping_selector.cpp) → registry 或 executable fallback；[tune_butterfly/tune_ntt](../src/c_api.cpp) 消费候选 | 默认 miss 是 first executable、unmeasured-feasible；公共 MEASURE 才测量。不能把回退当已调优结果 |
| H 硬件 | [capacity_partitions](../src/plan_registry.cpp) 查询 query_hardware_resource_model，按 shared 容量生成分组；registry_records 匹配型号、SM、容量；runtime_fingerprint 包含 compile policy；[platform_candidates](../src/platform_tuning.cpp) 消费平台规则与 legacy prior | 当前 C++ 共用分组首先使用容量限制；不能因存在完整 Python cost model 就声称每条默认路径消费服务代价 |

证据定位：`src/plan_registry.cpp:38` 的容量分组、`:474`/`:742` 的候选函数、`:805`/`:824` 的 fallback；`src/jit_module.cpp:176`/`:184` 的 shared 序列化；`scripts/compile_module.py:302` 的生成器；`src/mixed_dataflow.cpp:286`/`:301` 的语义投影。这里的行号是审查快照，符号名可用于后续定位。

## 算子实现和公共可达性

| 算子 | 真实已有模板与参数入口 | 生成与公共选择状态 | 未完成部分 |
| --- | --- | --- | --- |
| FFT rank-one | scalar trait、寄存器 codelet、cuFFTDx imported core、factor-streamed；FftCore、prefix_codelet/lanes、factor_partition/EPT/columns、exchange 与边界参数 | portable_butterfly_candidates 在 FFT 分支枚举核心/布局/分组；register/factor/shared JIT 均有真实生成路径 | 不在本审查扩展 FFT 实现；不同核心可表达的组织范围仍须逐一核验 |
| NTT | NttStageOperatorT word32/64 的 radix2 Shoup；ntt.cu 中 homogeneous/APPT/packet/static-IO/register-tail 核心；NttSubgraphMapping、boundary_mappings、role mapping、coefficient_reuse_stages | common shared JIT 已实现；portable_ntt_candidates 显式枚举 compact、Hybrid2D、baseline/tile256、shared partitions/resident seeds/overlap；平台规则及 legacy prior 可注入专用 mapping | generic inventory 没有枚举完整 NttSubgraphCore 家族；shared lowering 明确只支持 radix2 Shoup，不能称 Barrett/APPT 已在同一 shared 模板任选 |
| FWHT | FwhtOperator/LowFwhtOperator；RegisterFwhtTraits 与 register_fwht_kernel；短 FP16→FP32→FP16 kernel；precision、accumulation、local_exchange、final_storage_rounding/output_scale；显式 `register_threads` | common shared JIT；generated FP32 warp-register logN=3–15 的候选按真实线程库存接入；short_fp16_fwht_supported 将 forward、out-of-place、连续 N=32/64 final-only 核心放到公共候选前列；校准和 mapping identity 保留线程轴 | `items=4` 仍是固定模板约束，线程数必须满足真实实例化和 shape 资源；短 provider 支持域不能推广到 BF16/逆变换/原位 |
| subset zeta | BooleanZetaOperator<false,UInt>，inverse 控制加减；uint32/64；common generic/shared/resident 组织 | render_shared 支持 subset-zeta 与旧 xor-zeta；portable_butterfly_candidates 共用分组/布局/线程/overlap；checked_zeta_measure 做逐元素验证 | 没有专用寄存器或第二个不同算术核心/provider；warp-register 的构造检查仅允许 FWHT/structured |
| superset zeta | BooleanZetaOperator<true,UInt>；其余共用参数同 subset | 独立语义分支与 exact reference，共用生成和选择路径 | 同 subset；不能以与 subset 共享 trait 代替另一核心的证据 |
| structured 2x2 | Structured2x2Operator/LowStructured2x2Operator；register_structured_kernel 复用 FWHT 的 register hierarchy；stage_matrices 支持 broadcast/per-stage；显式 `register_threads` | generated FP32 warp-register logN=3–15；generic/shared/resident lowering；portable inventory 将真实线程实例与 FWHT 分开验证；系数策略由矩阵数量决定 | broadcast/per-stage 是语义与存储表示，不能自由更换权重；`items=4` 不是搜索轴，其他矩阵/精度支持域仍需真实模板 |

生成证据：`scripts/generate_design_points.py:13` 的 SUPPORTED 明确列出 FWHT/structured FP32 warp-register；`:60`/`:104` 导出真实 launcher。`src/butterfly.cu:776` 构造期检查 generated availability，`:1883` 执行 structured 核心。`src/ntt.cu:9535` 明确拒绝 shared 非 radix2/Shoup 请求。`scripts/compile_module.py:325`–`:343` 逐算子选择 trait，不能把 Python 抽象轴表代替这些实现。

## 外部 provider 与设计策略移植

| 项目 | 当前完成程度 | 可支持的结论 |
| --- | --- | --- |
| cuFFTDx rank-one codelet | [external_fft_units.cuh](../src/external_fft_units.cuh)、cufftdx_fft.cu 及 JIT 路径实际调用 imported 局部核心，并与本库分组/布局配合 | 是真实核心 provider；固定核心的组织比较仍需匹配算术/shape/支持域 |
| cuFFT 整体 rank-one | ButterflyBackend::Cufft 与 Bluestein external 路径 | 整体外部基线，不等于把 cuFFT 内部 codelet 移植成可搜索模板；xD 单轴 provider 在其他审查范围 |
| fast-hadamard-transform 策略 | `src/register_fwht.cuh:8` 明确署名 register hierarchy/XOR exchange 来源；RegisterFwhtTraits、warp_wht、register_structured_kernel 为本库 standalone 实现，生成与公共候选可达 | 已完成有限支持域的策略移植，且用于 structured；不是直接调用外部 FWHT provider，也不是所有 shape 参数可搜索 |
| GPU-NTT / HEonGPU | [heongpu_ntt_adapter_probe.cu](../paper/experiments/e13/heongpu_ntt_adapter_probe.cu) 是实验 adapter/probe，使用 gpuntt::GPU_NTT_Inplace/GPU_INTT 并显式顺序与 twist 转换；src/include 的公共 NTT 链路未见 gpuntt provider | 有实验语义适配证据；不能宣称外部局部核心已导入公共 NTT 模板 |

减少中间写回、布局转换、复制或独立缩放 pass 是 A/L/F 与边界组织收益。计入相同完整语义调用，并核对归一化及舍入位置后，可以归为方法收益；无需把局部乘加指令变快作为唯一收益定义。但切换 provider 的算术收益不能混作同核心组织收益。

## 最重要的缺口与最小后续动作

1. **抽象设计空间与可执行库存未逐点对应。** `scripts/butterfly_design_space.py:287` 的 expand_design_choices 默认选择 projection 的第一个 core，生成的是参数记录；C++ 实际库存由 plan_registry.cpp 两个函数独立构造。最小动作：给候选导出增加来源和明确 lowering 状态，针对一个具体 projection 建立“抽象记录→已有 serialized mapping”的转换及拒绝原因；不把所有空间做一次无边界重写。
2. **NTT 专用核心已经加入有限公共库存，APPT 其余变体仍保留缺口。** `src/ntt_template_candidates.hpp` 按 `ntt.cu` 的真实支持域生成 homogeneous/static-IO、两层/三层 dataflow，以及带完整 ring、CTA 和 producer/tail/writer role descriptor 的基础 `ApptOnlineRegisterTail` 候选，并由 `portable_ntt_candidates` 接入。word64 的 logN=20/Shoup/自然顺序域生成 16 个，word32 生成最多 26 个；实际公共映射正逆 84/84 通过。工厂同时拒绝非法 batch、模数、32-bit 范围和 N 根条件。APPT warp、radix8、grouped 和 resident 变体仍需要各自的 liveness 证据，不能借基础核心宣称所有 `NttSubgraphCore` 已可自动搜索。
3. **FWHT/structured 的形状轴已接入，但范围仍受真实实例限制。** `RegisterFwhtTraits<LogN, Threads>` 固定 `kItems=4`，`register_threads=0` 表示旧的按长度派生形状，非零值必须存在于 generated inventory；生成器、launcher、公共 mapping、校准身份和真实执行回归均消费该字段。后续只需在有独立模板和数值证据时扩展 items 或低精度支持，不能把固定 items 宣称成自由搜索轴。
4. **共用 resident 模板的 P 仍较窄，固定核心跨组织验证存在缺格。** shared-resident 使用 trait.apply 的蝶形实现及小 leaf；FWHT 独立全变换 register core、NTT 专用 APPT 核心并非通过同一局部 core ABI 嵌入 shared resident。subset/superset 没有第二个已导入核心。最小动作：选择一个可隔离局部 codelet，定义支持域、输入输出顺序和资源接口，然后接入两种已有合理组织；其余缺格如实记录，不能用另一精度/算子补格。
5. **平台规则遗漏输出 epilogue 语义：本轮已修复。** 修复前 `semantics(ButterflyConfig)` 不导出 output_scale/output_rounding，规则过滤这些字段触发 unknown platform semantic field；registry 在 `src/plan_registry.cpp:148`–`:155` 已支持。现新增 `output_scale=c.output_scale`、`output_rounding=c.final_storage_rounding ? 1 : 0`，保持 registry 同名同值；CPU 回归覆盖精确命中、缩放不匹配及舍入不匹配。

## 本轮修改与验证

本文记录的是静态审查和平台字段回归。后续统一模板参数修复实际修改了
FWHT/structured 的 `register_threads` 生成与公共候选、NTT 的有限专用候选工厂，
以及相应的 mapping、校准和 GPU 回放；完整修改与证据统一见
[本轮实施记录](core_template_contract_20261013.md)，不要将本节旧的审查范围误读为
“没有修改候选库存”。

CPU 验证使用当前源码的 `platform_tuning.cpp` 和 `test_platform_tuning.cpp`，链接已有 `build-a100/libcuntt.a`，输出到临时目录，避免重建主 Agent 正在使用的构建树；运行时设置 `CUDA_VISIBLE_DEVICES=`。结果及文档检查在交付报告记录。该测试不构造 GPU plan，不作性能证据。
