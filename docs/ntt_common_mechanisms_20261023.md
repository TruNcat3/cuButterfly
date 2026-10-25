# NTT 专用机制与公共组织模板审查

审查日期：2026-10-23。范围：当前源码中的 NTT 专用组织、公共表征、
执行模板与候选入口。本文是静态代码审查，没有运行 GPU、重新验证数值或
新增性能测量；不更新 [2026-10-13 组织实验](common_organization_20261013.md)
的原始数据和验收结论。代码行号用于定位本次审查时的实现，后续编辑可能移动。

项目及软件署名见 [项目与作者](about.md) 和 [CITATION.cff](../CITATION.cff)。
本次代码核对与文档整理由 Codex 执行；不据此填写人工实验负责人或验证者。
证据口径遵循 [研究原则](research_principles.md) 与
[报告署名约定](reporting_contract.md#署名与引用)。

## 当前结论

公共表征与调用入口已汇合，SharedIterative、局部寄存器驻留叶子和
BatchPipeline 已使用真正共享的执行代码。APPT 的 grouped、resident 2D、
角色遍历和细粒度 publication 等机制仍主要由 NTT 专用 kernel 消费。
基础 APPT register-tail 已接入公共候选；其余变体的公共描述和搜索覆盖仍有缺口。
因此，既不能说所有 APPT 都未接入，也不能说这些机制已经完全泛用。

需要分别记录四件事：公共 mapping 能否序列化重放、公共 IR 是否完整描述、
公共候选是否枚举、多个算子是否消费同一执行模板。一个状态不自动证明另一个。

## 九项机制证据

| 机制 | 公共表征与实际代码位置 | 共享执行状态及剩余边界 |
| --- | --- | --- |
| SharedIterative | [ExecutionGroup](../include/cuntt/mixed_dataflow.hpp#L92) 表达 stage/data/batch 和 layout；[NTT 调用](../src/ntt.cu#L9907) 与 [其他蝶形调用](../src/butterfly.cu#L2122) 使用同一 [shared_iterative_kernel](../src/shared_iterative.cuh#L4)。 | 已共享。算术由 `Operator::apply/finalize` 注入；[NTT trait](../src/ntt_traits.cuh#L25) 保留模域、Shoup 和归一化语义。无需重新实现这一组织机制。 |
| 局部寄存器驻留叶子 | 公共 `local_stage_partition/exchange_chunk` 进入 [ExecutionGroup](../include/cuntt/mixed_dataflow.hpp#L116)；[JIT 生成器](../scripts/compile_module.py#L389) 注入 NTT trait，并按 local partition 选择 [shared_resident_kernel](../scripts/compile_module.py#L400)。 | 已共享 FFT、NTT、FWHT、zeta、structured 的生成与模板；[run_leaf](../src/shared_resident.cuh#L25) 在寄存器中执行依赖闭合子图。当前每叶 1–5 stages 是模板实现域，不代表 APPT resident 2D 已迁移。 |
| BatchPipeline / 双槽 ring | `stage_overlap/batch_tile_count` 投影为 [BatchPipeline 与 RingBuffer](../src/mixed_dataflow.cpp#L550)；[NTT](../src/ntt.cu#L8108) 和 [其他蝶形](../src/butterfly.cu#L1279) 创建同一调度器。 | 已共享。[BatchPipeline](../src/batch_pipeline.cuh#L9) 用 stream/event 和每边两槽管理 batch tile；它与 APPT kernel 内 token/readiness ring 是不同执行协议。 |
| APPT warp-role / register-tail / token ring | [NttSubgraphMapping](../include/cuntt/ntt.hpp#L247) 有 `role_stages/token_interleave`；[ApptRoleMapping](../include/cuntt/ntt.hpp#L268) 独立表达 producer/tail/writer 分配；[launcher](../src/ntt.cu#L7433) 实际消费角色权重。 | mapping 可表达，执行仍是 NTT 专用 [cooperative kernel](../src/ntt.cu#L1776)；[APPT pipeline](../src/ntt.cu#L405) 另有 packet/token ring。当前 online 的 logN=20、7+7+6、256 threads [检查](../src/ntt.cu#L9473) 是工程实现域，不是理论限制。 |
| Grouped producer / writer-final | [分派](../src/ntt.cu#L7570) 以 `data_space` 选择 `ProducerAGroup=8/16/32`；[core 选择](../src/ntt.cu#L7615) 决定 writer-final 与 data-time 模板分支。 | NTT 专用实现已存在；公共 `ExecutionGroup` 尚无等价的 producer 分组和最终阶段职责描述。当前 writer-final 的自然序要求见 [验证](../src/ntt.cu#L9535)，属于此布局实现的契约。分组和职责转移本身可复用。 |
| APPT resident 2D / quarter | 专用 [half](../src/ntt.cu#L1703)、[quarter](../src/ntt.cu#L1739) helper 融合行列计算；[launcher](../src/ntt.cu#L7391) 选择 shared 容量及 occupancy 目标。 | 尚未迁移到公共 `shared_resident`。这里还包含固定二维 tile、row/column codelet 与角色 publication；容量、形状和 data-space=32 是当前工程/硬件域。模运算由 trait 承担。 |
| APPT data_time / 角色掩码 | 公共 `ExecutionGroup.data_time` 已存在；APPT 另用 `data_time_role_mask`，通过 [task 解码](../src/ntt.cu#L1693) 和 [角色循环](../src/ntt.cu#L1891) 按空间任务重排 batch transforms。 | 字段名称部分统一，执行含义仍需细分。SharedIterative 的 [data_time 投影](../src/mixed_dataflow.cpp#L579) 是 tile 内 butterfly/线程遍历次数，APPT 此处控制 transform 分组与角色队列顺序；不能仅凭同名认定已共享。 |
| Packet readiness / packet radix4 | NTT `PlanConfig` 的 packet readiness/layout/barrier 选项进入公共 mapping；实际 [packet core](../src/ntt.cu#L4476) 固定 word32、四行及 128-thread group。 | 专用实现可重放，尚未成为跨算子执行协议。公共 [partial_dependency_ready](../include/cuntt/mixed_dataflow.hpp#L109) 只是粗粒度字段，不能重建完整 publication 协议。分组、bitmap 和局部 barrier 可复用；当前类型/形状限制来自实现。 |
| Coefficient reuse | [coefficient_reuse_stages](../include/cuntt/ntt.hpp#L255) 经 mapping 进入模板；[执行分支](../src/ntt.cu#L3611) 实际选择 reused/cached register stages；[公共候选](../src/ntt_template_candidates.hpp#L164) 已包含部分 reuse cores。 | 已参数化并部分接入候选，尚无通用 coefficient-supply policy。公共 shared 模板仍调用 `op.apply`，而 [NTT trait](../src/ntt_traits.cuh#L35) 按 stage/offset 取系数。按 lane 缓存和供给是可复用策略，Shoup 辅助系数及模数范围是算术差异。 |

## 公共入口与真实执行

[serialize_mapping](../src/plan_registry.cpp#L399) 保存 subgraph、boundary、packet
及 APPT roles，说明专用机制已有公共重放身份。但公共 IR 的
`ExecutionGroup/ExecutionBoundary` 仍不能独立重建所有角色职责、任务顺序和
publication 协议。例如 [check_lowering](../src/mixed_dataflow.cpp#L705)
仍把非 BatchPipeline 的 `RingBuffer` 判为 `UnsupportedBoundary`；枚举存在
不等于 APPT ring 已获得通用 lowering。

[portable_ntt_candidates](../src/plan_registry.cpp#L930) 消费专用模板工厂，
[tune_ntt](../src/c_api.cpp#L1616) 消费这一公共 inventory，并按搜索预算裁剪。
基础 [ApptOnlineRegisterTail 候选](../src/ntt_template_candidates.hpp#L179)
已有完整 ring、256-thread CTA 与 producer/tail/writer 描述。其他 APPT
grouped、resident、warp/radix8、data-time 变体则明确登记为
[缺独立 liveness-validated descriptor](../src/ntt_template_candidates.hpp#L57)。
因此“源码可执行”“可显式重放”和“自动搜索可达”仍需逐变体核对。

## Coefficient-supply capability boundary

系数供给目前是两层能力，不能由字段同名推断为同一接口。公共组织模板只
要求算子提供 `apply(stage, offset, left, right)`，并把系数寻址和算术留给
算子对象。它不会保存、缓存或预取系数。对应的实际消费者如下：

| 算子/路径 | 公共阶段系数读取 | 显式系数复用 | 辅助系数 | 当前可作为公共 supply 候选 |
| --- | --- | --- | --- | --- |
| FFT `FftOperator` / `LowFftOperator` | `twiddles[stage, offset]` in `apply` | 否；仅由具体 FFT provider 的寄存器或 recurrence 策略决定 | 否 | 否 |
| FWHT / zeta | 无外部系数表 | 不适用 | 否 | 否 |
| Structured 2x2 | `matrices[stage]` in `apply` | `broadcast` 和 `per-stage-table` 是独立 generated codelet 选项 | 否 | 仅限其现有 codelet |
| NTT `NttStageOperatorT` | `twiddles` and `twiddles_shoup` in `apply` | `apply_coefficient` plus cached/reused register stages in `ntt.cu` | Shoup table and modulus | 仅限 NTT descriptor |
| xD resident | per-axis twiddle arrays | provider-local table/recurrence | NTT axes may have Shoup arrays | 否，尚无共同 lowering |

因此当前没有可以由 FFT、NTT、FWHT 和 zeta 共同消费的
`coefficient-supply` 执行模块。`coefficient_reuse_stages` 只存在于
`NttSubgraphMapping`，不能投影到 `ExecutionGroup`；把它复制到公共 IR 会
使 FFT/FWHT 的 lowering 接受一个没有消费者的参数。公共模板继续通过
`Operator::apply` 保持语义通用，NTT 的 `apply_coefficient` 和 Shoup 辅助表
继续由 NTT descriptor 约束。后续只有在第二个算子有真实的缓存/供给消费者，
并能描述系数所有权、生命周期、寻址和资源占用后，才抽取公共接口。

当前专用模板工厂的 [支持域](../src/ntt_template_candidates.hpp#L16) 包含
logN=20、Shoup、自然输入输出顺序及 word32/64 等条件。这些条件描述现有
候选工厂，不定义统一方法的理论空间。公共 SharedIterative 还明确
[拒绝专用 subgraph mappings](../src/ntt.cu#L9212)，两类后端尚不能任意混合。

本次补充了最小 descriptor contract。`ntt_template_descriptor_reason` 集中
校验 register-tail 的 7+7+6 阶段划分、三个 256-thread/8-unit/16-space/8-time
角色、两条双槽 ring 边界、双 pipeline buffer，以及 producer/tail/writer
权重、fragment 和 data-time mask。候选生成只登记通过该校验的基础变体，
而设备相关的线程数、共享内存和编译资源仍由 `Plan::validate_config` 检查。
`ntt_template_candidate_contract` 还对该候选做 `serialize_mapping` 往返和
破坏 ring 槽数的负例验证，确保字段不会在公共缓存或重放路径中静默丢失。

## Role 与 data-time 的二次核对

当前不能把 `ExecutionGroup.data_time` 直接当作 APPT 的
`data_time_role_mask`。公共 SharedIterative 路径使用 `data_time` 表示一个
物理组内的批次或 butterfly 遍历次数，并由
[`mixed_dataflow.cpp`](../src/mixed_dataflow.cpp#L451) 投影到统一执行组。NTT
APPT kernel 则分别读取 producer、tail、writer 的角色 mask，并在
[`ntt.cu`](../src/ntt.cu#L1891) 中改变任务解码、publication 和 transform
队列顺序。两者数值相同也不意味着执行协议相同。

本轮没有新增一个只保存这些字段的“公共 role 模块”。这样做会让 mapping
看似统一，却无法由 FFT 或 FWHT 的 lowering 消费，也无法验证 buffer 所有权。
在细粒度 readiness、role service rate 和 ring 生命周期具备跨算子执行者之前，
专用字段继续留在 NTT contract 中，公共 IR 对非 BatchPipeline ring 继续明确
拒绝。下一步只有在第二个算子实际消费同一 publication 协议后，才抽取公共
role traversal 模块。

## 最小后续三步

1. 先补已有变体的 descriptor 与 search。复用现有 kernels，逐个补齐角色、
   task 分组、publication、buffer 生命周期、资源约束和支持域；贯通候选、
   分派、实际 mapping、缓存身份与重放。把未描述、未枚举、未编译、未测量
   分开登记，保留现有公共基准 APPT 候选。
2. 再提取通用组织模块。从现有实现抽出独立行分组、角色服务率、任务遍历、
   系数供给及细粒度 readiness 协议，通过算术 trait/codelet 接入多个算子。
   继续复用 SharedIterative、shared_resident 和 BatchPipeline；模块验收以
   真实消费者和实际参数消费为准，不以新增公共字段代替执行迁移。
3. 最后验证跨算子数值、生命周期与性能。整数/NTT 精确比较，浮点采用同精度
   CPU 参考并报告误差；覆盖方向、排列、变输入、恢复输入、buffer 所有权和
   CUDA Graph 重放。性能采用固定核心的组织对照及相同预算重搜索，记录完整
   mapping、硬件/构建身份、资源、流量和负结果，再判断可复用模块的收益域。

这三步是后续实现与验证顺序。本次已完成基础 register-tail descriptor
contract 和 CPU 序列化负例，尚未把其余专用变体迁移为公共执行模块，也没有
新增 GPU 性能结论。既有跨算子组织实验继续作为其原构建和命名范围内的证据，
不扩大为所有 NTT 专用机制的泛化证明。
