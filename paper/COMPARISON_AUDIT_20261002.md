# 论文对照与缺口审计

本审计把 APPT26 的 Hermes 原稿、UnionSparse（arXiv:2608.09291）和当前工作稿与代表性算子/编译系统论文放在同一证据框架下。它只判断论文是否把主张说清楚，不把外部论文的性能数字并入 cuButterfly 结果。

## 已经形成的主线

cuButterfly 已经比 APPT26 更适合作为 arch/sys 论文。APPT26 证明了 NTT 的 stage/data 混合组织、roofline 动机、bank/offset 写回和硬件利用率。当前论文把这些专用设计提升为依赖驱动的蝶形处理范式，并加入算子语义、负载域、硬件能力分离，闭合子图、后继对齐写回、物理服务校准和自动选择。

UnionSparse 提供了很好的证据组织范式：先给总体分布，再做受控消融，最后用 trace 指标解释原因。当前论文已经补上跨 FFT/NTT/FWHT 的绝对吞吐箱型总览、固定核心×组织消融和部分 trace；端到端结果和 RC2 资源计数仍不足。

Welder、ROLLER、Ansor 和 TensorIR 说明层次映射、硬件对齐 tile、进化搜索和核心/外层组织分离都已有先例。cuButterfly 的贡献不能写成“首次解耦”或“首次搜索”，而应写成蝶形 stage-bit 依赖如何构造闭合子图、边界包和后继布局，并把这些对象接入可执行 lowering、校准服务和跨算子选择。

FlashFFTConv、GPU-NTT、VkFFT 和 HadaCore 说明专用分解、核心实现和完整库基线都可以很强。它们要求本文明确区分组织收益、核心收益、接口适配成本和应用收益。

## 与代表性工作的对照

| 对照对象 | 对方已经说清楚的内容 | cuButterfly 当前还需补强 |
| --- | --- | --- |
| APPT26 / Hermes | NTT 的 stage/pipeline 分类、roofline、bank/offset、硬件资源与多长度吞吐 | 将 NTT 专用图推广到一般蝶形；加入闭合子图和边界构造的可验证规则；给出 GPU 的寄存器、shared、HBM 与 launch 代价 |
| UnionSparse | 箱型分布、固定条件消融、profile 指标和机制解释的顺序 | 用绝对 transforms/s 与 butterfly-pairs/s 做主图；每个关键机制补匹配 trace；报告方差、尾部和负结果 |
| Welder / ROLLER | tile/连接/liveness、硬件对齐映射和局部性能模型 | 给出候选数量、剪枝规则、lowering 缺口、编译/校准/搜索成本和 break-even |
| Ansor / TensorIR | 计算与调度分离、搜索预算、局部 block/core 与外层变换 | 说明 stage-bit 依赖和 arithmetic labels 如何进入核心接口；增加预算—质量曲线和跨负载 holdout |
| FlashFFTConv | FFT 分解、容量、融合和端到端长序列收益 | 增加真实应用；区分 C2C 与 real FFT；报告融合/适配/核心/组织四类消融 |
| GPU-NTT / VkFFT / HadaCore | 面向单一算子的高性能核心或完整实现 | 统一长度、batch、精度、方向、布局、计时边界；绝对吞吐与资源归一化；不猜测专有库内部实现 |

## 建议补齐的图

1. **绝对吞吐主图。** 已新增 RC2 FFT/NTT/FWHT 的统一箱型图，按算子和精度分组，展示 transforms/s 和 effective butterfly-pairs/s，并叠加每个采样点。逐 workload 的 FFT 配对图保留为细节证据；旧比例箱型图降为历史审计材料。
2. **设计空间与选择图。** 需要一张从 stage-bit 分段树、闭合子图、边界包到最终 mapping 的示例，标出合法、容量淘汰、无 lowering 和实测 finalist 四类候选。
3. **机制 trace 图。** 至少给出一个正收益和一个负收益 case 的 kernel 时间、launch 间隔、HBM/L2 流量、寄存器、occupancy 和 stall 分类。当前 pipeline overlap 表只能证明 launch fragmentation，不能定位完整资源原因。
4. **跨规模图。** 对固定语义绘制 $N\times B$ 的绝对吞吐和有效 butterfly/s，分 FP32/FP64、forward/inverse，并保留未测点为空。
5. **端到端图。** 已加入 FNO、QuaRot 和 HEonGPU 的统一计时边界协议图。真实结果阶段还需给 baseline/treatment 的端到端时间、蝶形占比、适配开销和峰值显存；在此之前不画空柱状图。

## 建议补齐的表

1. **平台与合同表。** GPU 型号、显存、SM、编译模式、UUID、输入/输出顺序、精度、warmup/repeat/trials、是否包含 plan/适配器。
2. **支持矩阵。** FFT、NTT、FWHT、zeta、structured-$2\times2$ 的 CT/GS、正逆、布局、精度和 lowering 状态，明确 `supported`、`reference-only`、`unavailable` 和 `blocked`。
3. **选择成本表。** 编译、校准、搜索、首次调用、稳态 kernel，以及 break-even 调用次数；不能把准备成本混进 kernel 加速比。
4. **资源表。** 代表性设计点的寄存器、shared memory、occupancy、launch 数、HBM/L2 字节和有效带宽。
5. **端到端合同/结果表。** 每个应用的公开 commit、模型/数据 hash、替换边界、质量误差、应用总时间和 butterfly fraction。

## 目前论证仍不充分的文字

- **“分层”为什么必要。** 需要明确它首先扩展可表达的逻辑 stage 深度，其次才是硬件性能选择；更长的 $N$ 可以递归增加边界，不能暗示所有中间量都驻留片上。
- **“解耦”带来什么证据。** 目前有固定核心组织消融，但还缺少相同组织下的核心替换、相同核心下的组织替换、编译复用和迁移成本四项并列说明。
- **“最优”口径。** 搜索器只能在已生成、已校准的有限候选池中选择最快确认点；应统一写成 measured competitive 或 finite-pool best，不能写全局最优。
- **“互相掩盖”的条件。** 需要把依赖完整性、buffer lifetime、lookahead、资源竞争和 fill/drain 写成条件，不能由 batch 增大直接推出 overlap 或隐藏 HBM 延迟。
- **外部基线公平性。** NTT 的完整矩阵仍有明确边界，FWHT DAO 含接口开销，FFT 的 VkFFT 只覆盖 FP32；这些限制应出现在图注和主文而不是只放脚注。
- **应用价值。** 单算子加速不能替代应用收益。至少要给出实际应用中蝶形占比和适配成本，才能把方法主张推进到系统价值。

## 当前优先级

最高优先级是完成端到端三应用的真实测量和资源 trace。其次是把设计空间/支持矩阵/准备成本表放入正文或补充材料。最后再扩展同版本 V100 和 GS lowering。这样论文的主证据链会从“算子接近或超过基线”推进到“方法如何工作、何时有效、为何能产生应用收益”。
