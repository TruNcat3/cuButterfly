# 候选约束流程核查

2026-10-24。范围是 FFT 预取实验暴露的粒度问题，以及公共流程在不同
蝶形算子中的复用情况。本页是代码审计，不是新性能结果或统一重构完成声明。

## 论文、实现与实验的区别

APPT26 `samplepaper.tex` 的混合数据流描述区分 `Npart`、`Spart` 和 `p`。
`Efficient Dataflow` 描述每周期读取和写回 `2p` 个系数，子图跨多个周期
连续消费。本文方法不要求每个 worker 预存整次变换。原 FPGA 实例还使用
URAM 保存其支持范围内的中间状态，因此“实例选择片上存放”与“理论必须
全量片上存放”也必须区分。cuButterfly 的方法章节进一步明确，父区域可以
细分，跨区域中间数据可以放在更高存储层。

当前 `src/fft_factor_streamed.cuh` 把输入环形缓冲单位绑定为
`局部 FFT 点数 × factor_columns`。每次先把这一组输入搬入寄存器，再对
释放的输入槽提交未来组的 `cp.async`，执行局部核心，最后复用核心 shared
做输出布局并写回。它不预取完整 N 或 batch；但也没有独立于核心分组的
搬运 chunk 参数，局部 `FFT().execute()` 是整组核心调用。

以 FP64 局部 1024 点为例，下表只列动态 shared 的下界，未假装包含编译后
静态 shared、寄存器或超过输入大小的核心 scratch。

| CTA 内局部 FFT 数 | 输入环深度 | 输入环下界 | 核心/输出复用区下界 | 合计下界 |
| ---: | ---: | ---: | ---: | ---: |
| 4 | 0 | 0 KiB | 64 KiB | 64 KiB |
| 4 | 1 | 64 KiB | 64 KiB | 128 KiB |
| 4 | 2 | 128 KiB | 64 KiB | 192 KiB |
| 2 | 2 | 64 KiB | 32 KiB | 96 KiB |
| 1 | 2 | 32 KiB | 16 KiB | 48 KiB |

`depth=2` 被口头称为“双缓冲”不准确，它有两组输入环和另一个工作区。
A100 本次 NCU 的设备属性为每 CTA opt-in shared 上限 166912 字节。
只有第三行已由下界证明不满足此容量约束。其他行仍要通过核心、线程及
编译资源校验，不由此表保证高效或可执行。固定局部形状时，增加 N 或 batch
只增加任务总数，不按比例增加该 CTA 的缓冲容量。

最初受控实验固定列数四、分段和 EPT 后增加预取深度，未在编译前进行
联合约束筛选。超限项来自这个手工实验，不是公共搜索选出的赢家。直接
跳过超限项只能避免启动失败，不能补足对其他合法 tile/缓冲组合的观察。
该旧矩阵不应作为混合数据流空间已得到完整验证的证据。

## 已有共同层与待统一处

| 层次 | 已有实现 | 当前边界 |
| --- | --- | --- |
| 公共表示 | `MixedDataflowPlan`、图语义、物理执行组与边界 | 公共表示存在不代表所有候选工厂已按同一流程实现 |
| 共享组织生成 | `shared_organization_candidates` 同时被 NTT 和 Butterfly 候选使用 | 只覆盖 shared 组织，并非所有 lowering 的总生成器 |
| 公共资源检查 | `estimate_resources(plan, hardware)`、`validate_plan(plan, hardware)` | 已接入 Butterfly/NTT rank-one 生产构造；未知资源不等于编译后完整验证，原后端检查保留 |
| 特定核心投影 | FFT factor、register、NTT 等声明或推导核心形状 | 当前仍有重复的线程/shared筛选与设备查询；部分通用组织参数绑定特定 lowering |
| CPU 实验准备 | provider 数据库与 JIT source renderer 可离线检查 | 脚本未统一调用公共硬件约束流程，手工组合可能遗漏预检查 |

公共 FFT factor 搜索在 `src/plan_registry.cpp` 联合枚举 columns、EPT、
depth、分段和连续组数，并按硬件容量筛选。它不是固定四列后禁用预取。
本次修改后，其公共硬件查询可消费同一静态硬件快照；CUDA 查询仍用于
设备身份校验及缺少快照时的显式回退。纯 CPU loader 可独立检查快照，但
生产候选工厂仍在内部查询设备身份，尚不是完整离线生成入口。
Python `calibration_space` 提供资源投影，数值是估计，不提供
覆盖所有 lowering 的硬件合法性证书。

本轮修改前，进一步核查得到三处生产链路缺口。

1. `src/mixed_dataflow.cpp` 的 `validate_plan` 尚未被生产构造调用。
   `ButterflyPlan` 与 NTT `Plan` 仍分别校验线程、shared 和 grid。显式
   mapping 跳过枚举和选择，但仍经过这些实际构造检查，不是跳过所有校验。
2. 未命中精确 registry 时，默认 selector 存在按平台先验顺序返回首个
   可执行候选的回退，注释是 `unmeasured-feasible`、`no local timing estimate`。
   C API 却把它归入 `SELECTION_STATIC_MODEL`。这一路径不是统一模型的完整
   成本排名；显式测量策略才在其测量集合内按时间选择。
3. JIT 把编译资源写进逐执行组描述。通用 estimator 目前只有 factor 分支
   逐组消费这些信息，其他后端仍读取可能为零的顶层寄存器字段。统一的
   编译后资源反馈尚未闭环，不能用其估计值替代真实驻留指标。

对应代码为 `src/mapping_selector.cpp`、`src/plan_registry.cpp` 的 portable
candidate/selector、`src/c_api.cpp` 的选择 provenance，以及
`src/jit_module.cpp` 和 `src/mixed_dataflow.cpp`。必要的 NTT 模数/根、
structured 矩阵、FFT 精度与核心支持域分支应保留。统一的目标是共用
组织和约束流程，并非删除算术语义差异。

CPU 投影与模型还发现两项独立缺陷：部分非 shared 投影未检查分段总和，
以及驻留数为零时模型用 `max(1, ...)` 仍给出有限分数。这两项已修复：
分段须正值、覆盖全部逻辑阶段并满足物理边界；已知资源超过 per-CTA 或
per-SM 容量时成本为不可行，不再造出一个可驻留 CTA。未知编译资源仍标为
未知。三项生产链路的后续修改见下节，不能与这两项修复相互替代。

统一方向是复用现有公共表示和资源检查，让各核心提供能力与资源需求，
由共同流程联合生成、筛选、编译补充资源、测量和选择。不能继续把独立
FFT 实验生成器当作这个统一入口，也不能为了接口形式统一而声称每个核心
都能使用每种机制。当前公共流程及合法缓冲组合的运行结果见
[公共回归报告](common_planner_validation_20261024.md)。

## 静态平台输入

[ADR 0001](adr/0001-static-profile-driven-planning.md) 固化所有算子的共同合同。
`hardware_model.json` 描述型号、CC、容量和资源约束；性能采样与模型文件
独立维护，核心能力和编译资源另有来源。不能把 SM 总容量当成 CTA 上限，
也不能把硬件寄存器数量当成核心实际寄存器需求。

已接通默认复用、显式刷新、手动采集导出及公共硬件读取。显式文件优先，
然后本地完整文件，再使用匹配的仓库种子；只有部分旧校准时明确保留部分
状态。现存坏文件直接报错，不静默替换。仓库种子安装仅补缺失文件，不覆盖
用户数据。首次或显式刷新调用 `--hardware-live` 获取真实属性，再让后续
候选和测量子进程绑定同一个静态文件。

A100 80GB PCIe 种子来自本机已有校准与 NCU 设备属性，包含容量和历史实测
速率；没有增加 GPU 测量。旧 `physical-stage-composition-v6` 成本模型带
`calibrated-local-warning`，且 `usable_for_unmeasured_ranking=false`，因此
没有将其发布为已验证的新模型。容量完整、服务层验证与整个模型排序有效
是不同的结论。

CPU 回归覆盖静态文件选择/导出、模拟安装复用/刷新、安装不覆盖、本轮
分段投影和成本拒绝。C++ loader 可独立运行。实际构建、运行结果与静态
接口完成度分别记录，不由接口测试推断 GPU 正确性或性能。

## 生产链路补齐

`ButterflyPlan` 和 NTT `Plan` 已接入共同 `validate_plan`，在执行前复核
JIT 回填的实际资源。通用 estimator 逐执行组检查线程、shared、寄存器和
驻留约束，不再仅在 factor 路径读取逐组编译信息。shared 默认分段也已
按构造器的 local/flow stage 大小展开，避免将整个 N 误投影成单个片上 tile。

registry 精确命中仍优先。未命中时，公共 rank-one selector 对完整投影集合
调用已有 stage cost model 的 CPU 排名，编译有限候选后再排名。预算默认四个
可执行候选，并记录 projected/ranked/unranked/rejected 分区和实际预算。
`CUBUTTERFLY_MODEL_FINALISTS` 可调整预算。无可靠估计时保留可执行回退，
C API 使用独立 `UNMEASURED_FALLBACK` 类型，避免冒充静态模型选优。

模型自身验证和请求兼容性分开检查。只有版本、真实 validation 字段、build、
硬件和计时协议均匹配，且候选 primitive/overlap 范围受覆盖，才使用拟合模型。
旧模型的 warning 不会被忽略；可用硬件实测速率单独标为 `hardware-rate-prior`。
缺少物理启动描述的候选保持 unranked，而非把缺失开销估为零。

[物理启动描述审查](physical_launch_descriptor_audit_20261024.md) 记录了新的
遗留问题：部分旧 NTT 后端将多个真实 launch 表成一个逻辑组；Baseline/Tile
还有 copy/scale 辅助 launch。当前成本合同不能把这些零蝶形阶段当作一个
蝶形阶段。该缺口不代表这些后端不支持执行，但不允许冒充完整静态排名。
此范围与 xD 独立选择器均不能由本轮 rank-one 补丁宣称完全统一。

`scripts/check_common_planner.py` 固定本轮 27 个正确性组合与 7 个性能点，
默认只生成命令；`--run --gpu-uuid GPU-...` 才执行。覆盖 FFT FP32/FP64、
FWHT、subset/superset/xor、structured-2x2、NTT word32/word64 的显式 shared
和 AUTO 路径。记录原始 CSV、全部 batch 验证、实际设计点与模型来源；
浮点仍使用已有 bench 误差检查，不把 max_error 冒充完整误差分布。
GPU 被其他进程占用时保存中断状态，仅停止自身进程组，支持已有等待器恢复。
这组有限回归不是论文全矩阵或整套理论空间的验收。

## 本轮状态

- 等待器已完成本轮任务并正常退出；旧固定矩阵仍保持暂停。
- 重建和公共 GPU 回归已完成。27 个正确性条目与 7 个性能条目由 29 次
  独立执行覆盖，5 个同形状正确性条目复用性能测量的完整验证。所有请求的
  batch 均通过原 bench 检查。普通机制、Graph 和公共 planner/replay 测试通过。
- 三个合法 tile/并行度/缓冲组合均已运行。固定核心和其他参数的预取对照
  得到 1.4473× 提升；更大且超限的组合在 CPU 检查拒绝，未启动 GPU。
- 本轮 AUTO 性能回归使用空 registry 和历史硬件速率先验，验证公共选择路径。
  它不是完成新一轮校准与 tuning 后的最优性能横评。缺少物理描述、编译资源
  或模型覆盖的路径仍分别记录，不以这组回归宣称全部补齐。
- FFT-L 请求准备、tcFFT 和低精度 cuFFT 独立桥接已完成 CPU 阶段；实际
  GPU 正确性、Graph 和性能仍待验证，不是已完成的性能对照。
- 独立 FFT 联合组合原型未作为生产入口；本轮使用生产构造、公共约束与
  实际 mapping 验证合法组合，避免将专用实验脚本误当成统一流程的修复。
