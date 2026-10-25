# 论文实验执行目录

全仓库、全蝶形的设计与验证口径以
[项目研究原则](../../docs/research_principles.md)为准。重点区分组件消融、
关闭组件后重搜，以及“固定核心、更换组织并跨核心重复”的消融二；
已有 core×overlap 不能单独替代完整的模板可迁移性验证。
本轮三类表与 batch 执行见
[统一实验合同](unified_evaluation_contract_20261012.md)。

2026-10-12 的正文已按五个研究问题重组，当前补测以
[有限补测合同](principle_gap_campaign_20261012.md) 和
[剩余证据清单](principle_gap_campaign_20261012_followup.md) 为准。
它们复用已完成的 FFT 固定核心消融和 FWHT/NTT 局部分段控制，新增满 warp
布局配对，并明确受限重搜及第二可移植核心的实现依赖。下方旧 E 编号保持为
历史脚本标识，不根据旧“当前顺序”重新启动一轮全面测试。

这里把论文问题、输入、脚本和证据标准放在一起。数学和性能论点见
[EXPERIMENTS.md](../EXPERIMENTS.md)；机器可读清单是 [plan.json](plan.json)。
每张测试卡只维护一份 [target 配置](targets/a100-80gb-gpu1.json)，路径相对仓库根目录。
新增平台复制并修改 target 中的型号、显存、UUID、构建及本卡校准路径即可，
不需要另写一套实验脚本。

## 当前执行顺序

1. E1 继续已有综合验收，复用其已完成 trial、编译缓存和检查点。
2. 同时用 CPU 生成其他实验的配置、预编译请求和执行命令，并汇总已有证据。
3. E2 的严格 FFT 核心×组织四格和 E3 的 FWHT/word64 NTT 跨算术控制已在
   A100 80GB 完成；E4/E5 的机制与 profiler 记录沿用各自独立 cohort。
4. 冻结 E6 的整负载留出与 E9 的独立候选池，补齐 E7 跨平台和 E10 扩展成本。
   它们的前置条件在生成的协议文件中明确记录。
5. E11 扩展精度、正逆和算法形式；E12 单独记录编译、校准、搜索、首次调用和
   稳态 kernel 的准备成本。两者均不改变 E1 的冻结 cohort。
6. E13 对 FFT/FWHT/NTT 各选一个开源应用。其 [来源、负载与适配合同](../END_TO_END.md)
   和 [机器清单](end_to_end.json) 独立于原队列，先完成 CPU 准备，再开发和验证应用后端。
   `python paper/tools/prepare_end_to_end.py` 检查准备状态，不执行应用或占用 GPU。

论文三类主证据的最小 A100 cohort 见
[targeted evaluation plan](targeted_evaluation_plan_20261011.md)。它将 E1 算子横比、
E2 五个机制 profile case 和 E3 六个端到端 payload 与全面验收队列分开，避免等待
全面覆盖队列后才开始论文实验。历史结果只能在 build、GPU、语义和计时协议完全
匹配时复用；用户预期的 FFT 持平或小幅领先以及其他算子的领先均是待验证假设。

已有 E1 验收不读取这个目录的新配置；其构建、搜索代码和计时协议保持原样。
2026-09-17 的 CPU 准备、预编译和验收快照见
[本轮记录](../../results/paper_preparation_20260917/a100-80gb-gpu1/report.md)。

## CPU 准备入口

在仓库根目录运行：

```bash
CUDA_VISIBLE_DEVICES='' /home/wt/yes/envs/cubutterfly/bin/python \
  paper/tools/prepare_experiments.py \
  --target paper/experiments/targets/a100-80gb-gpu1.json \
  --output-dir results/paper_preparation_20260917/a100-80gb-gpu1
```

这个入口不执行命令，也不查询 GPU。E11/E12/E13 的条目可被登记，但仍使用各自的
独立入口，不因此自动加入原有测量队列。输出包括：

| 文件 | 用途 |
| --- | --- |
| `execution_plan.json`、`RUNBOOK.md` | 带 CPU/GPU 标记、环境、参数和前置条件的执行清单 |
| `workloads_e2.json`、`workloads_e3.json`、`workloads_e7.json` | 6 个 FFT 负载、4 个 FWHT/NTT 负载及其并集 |
| `factorial_requirements.json` | 10 个负载各自的四格映射待绑定表；未绑定不算可执行 |
| `controlled_layout.json` | FFT/FWHT/NTT 的同核心 writer layout 对照 |
| `controlled_partition.json` | 同核心 `[3,3,3,3]` 与 `[6,6]` 分段对照 |
| `controlled_batch.json` | 同核心、固定 N=4096，batch 1/4/16/64，bulk 与不同 tile 的流水对照 |
| `control_precompile/modules.json` | 仅上述显式配置的去重模块请求；不是整个理论设计空间 |
| `heldout_protocol.json` | E6 的训练/测试拆分和服务清单冻结要求 |
| `search_pool_protocol.json` | E9 的独立池、先验预测、4/8/16 总预算和随机种子要求 |
| `adaptation_audit.json` | E10 的新增形状/语义、复用路径和手写修改审计表 |

E11 的准备入口为：

```bash
CUDA_VISIBLE_DEVICES='' /home/wt/yes/envs/cubutterfly/bin/python \
  paper/tools/prepare_extended_semantics.py \
  --output results/paper_preparation_20260918/extended_semantics_prepared.json
```

它输出当前 CT lowering 的显式命令和 GS reference-only 记录。GS 只有在独立的
stage direction、twiddle 和输入/输出顺序 lowering 通过正确性后才进入 GPU timing；
设计空间中的 `algorithm_form={ct,gs}` 是可表达性，不是支持声明。

18 组受控 case 是机制实验的具体起点。分段配置是这些对照的固定控制条件，
不限制库的理论空间或自动搜索分段范围。NTT 使用 word64、模数
`576460756061519873` 和 natural 输出。运行时仍须检查实际 lowering、资源、
正确性和映射差异；两个配置若被解析成同一实现，就记为别名而非有效消融。

同一准备目录可以重复生成相同配置；改变计划、target 或生成逻辑后使用新的
准备目录。准备入口拒绝写入当前 acceptance 目录。需要继续某个已经开始的
GPU 测量时，使用该测量的 resume 入口，而不是重新生成其输入。

## 提前编译和配对计时

只编译本轮新增的显式控制请求：

```bash
CUDA_VISIBLE_DEVICES='' CUBUTTERFLY_COMPILE_MODE=research \
/home/wt/yes/envs/cubutterfly/bin/python scripts/precompile_research.py \
  --mode compile --jobs 2 \
  --output-dir results/paper_preparation_20260917/a100-80gb-gpu1/control_precompile
```

复用原有编译器和缓存锁，不重建主库。已有综合预编译的 118 个 FP64 cuFFTDx
不支持请求属于另一份清单，不随此命令重试。CPU 编译成功只表示模块准备成功，
不能替代目标 GPU 上的执行验证。

[通用受控运行器](../../scripts/run_controlled_experiment.py) 接受显式 study：
默认 `--mode prepare` 只检查配置和生成命令，`--mode run` 才执行 GPU 测量。
使用生成的 `RUNBOOK.md` 中的命令，先做 CPU prepare，再在独占窗口执行 run。
本轮三个 study 已完成 prepare；重复准备需加 `--resume`，且只能复用输入一致的
prepared 记录。已开始测量的 journal 只能通过 run 的 resume 继续，不能重新 prepare。
运行器按实际设备和构建绑定检查点，采用三次 trial、每次 100 warmups/100 repeats，
反序交替各 variant，并保留正确性和不可用结果。完整语义检查、初始化和编译不计入
CUDA-event kernel time。恢复时复用已有 trial，不混合变更过的研究输入。

严格四格需要在同一语义下绑定普通/优化核心和基础/混合组织。普通核心必须是正常
实现；不能用关闭优化、加入延迟或更换精度制造慢基线。当前固定 scalar/radix2
组织对照只证明该核心下的组织效应，不自动满足四格或外部库核心改造的要求。

E6 的 `heldout_protocol.py --mode prepare --candidate-points FILE` 可以冻结显式
有限候选集；默认仍保留原四候选实验。显式集合要求每个已选负载至少两个不同映射，
并检查候选语义与负载一致、记录输入哈希。已校准配置的新 batch 留出实验应在
`--mode predict` 时加 `--require-covered`：先保存预测，若存在缺口则退出并阻止
后续测量。结构冷启动的缺口实验可保留原默认流程，不能把 fallback 排序当成
校准模型的预测。2026-09-17 两类实验的边界见
[留出审计](../../results/paper_experiments_20260917/analysis/heldout_prior_gap_REPORT.md)。

现有 [factor-streamed 消融入口](../../scripts/benchmark_factor_streamed.py) 继续用于
FFT 的 data tiles、prefetch、factor slices 对照；它采用自己的 20 warmups/50 repeats
协议，结果单独成组，不直接塞进本轮 100/100 协议的四格。
[stage-overlap 入口](../../scripts/benchmark_stage_overlap.py) 保留历史专用实验用途。
NCU/NSYS 脚本按已确认的完整 mapping 绑定后再运行，不直接使用历史脚本中的默认
尺寸和核心参数；profile 时延不并入无 profiler 的延迟表。

补齐普通核心的组织消融可使用
[prepare_scalar_organization.py](../tools/prepare_scalar_organization.py)：

```bash
CUDA_VISIBLE_DEVICES='' /home/wt/yes/envs/cubutterfly/bin/python \
  paper/tools/prepare_scalar_organization.py \
  --requirements results/paper_preparation_20260917/a100-80gb-gpu1/factorial_requirements.json \
  --target paper/experiments/targets/a100-40gb-gpu0.json \
  --output-dir results/paper_experiments_20260917/scalar_organization
```

它读取原定的 10 个 E2/E3 负载，每点固定 scalar/radix2 核心、线程数和布局，
比较每个逻辑 stage 都写回 HBM 与最多六个 stage 在 CTA 内驻留。生成的 study
和模块清单仍交给上述公共预编译及受控运行器。该实验回答普通核心能否受益于
组织变化；FFT 的 native/imported factor 核心交叉实验则单独回答核心与调度的
作用，两者不合并冒充已具备所有核心的严格“无混合/有混合”四格。

此处的逐 stage 变体仍使用通用 `shared_iterative_kernel`。单 stage 的 CTA
只有一个 butterfly 计算线程，六 stage 时有 32 个，而两者均请求 128 个线程。
因此这一控制也改变了 CTA 内有效并行度，不能把其全部收益归因于 HBM 流量，
也不能将它当作已优化的传统逐 stage GPU 实现。对外库比较和同 factor 核心的
预取/调度比较各自报告，不用这一退化分组对照替代。

两张独占 GPU 可分别运行上述实验，每个进程通过 UUID 绑定本卡。SM80 编译模块
可以复用，校准曲线、timing journal 和性能汇总分别存储。

本轮新增的跨算术 E3 运行入口为
[`paper/tools/run_e3_cross_operator.py`](../tools/run_e3_cross_operator.py)。它的
`compile` 模式只做 CPU/NVCC 预编译，`run --target a100-80gb` 才执行固定的
24 单元 A100 80GB cohort；结果和哈希保存在
`results/paper_completion_20260918/e3_cross_operator/`。该入口不触碰 E7
迁移队列。

[run_prepared_queue.py](../tools/run_prepared_queue.py) 可按 `--command-id` 的
明确顺序执行准备清单，并以 `--after-session`、`--after-journal` 等待本卡已有
writer 正常完成。它记录命令和 wall time，检查 GPU UUID 与独占状态；前置任务
中断、等待到期或某个命令失败时停止，不自动重新校准或重试。每个队列使用独立
的 output journal。`completed-commands` 只表示这些命令已经结束，论文覆盖与
正确性仍以各实验自己的 journal 为准。

## CPU 证据汇总

```bash
CUDA_VISIBLE_DEVICES='' /home/wt/yes/envs/cubutterfly/bin/python \
  paper/tools/summarize_experiments.py \
  --acceptance-dir results/comprehensive_20260917/a100-80gb-gpu1 \
  --output results/paper_preparation_20260917/a100-80gb-gpu1/evidence_status.json
```

汇总保留来源哈希及完整设备/协议身份，以一次读取的 acceptance 快照为准。
E1 的完成度与局部性能、E6 的确认候选排序、E9 的现有搜索轨迹、E10 的已重放
覆盖分别报告。补测前后的排序必须区分，不能把 test-driven service extension 后的
结果称为零补测泛化；现有自适应搜索产生的池也不是独立的等预算算法对照。

`heldout_protocol.json` 与 `search_pool_protocol.json` 中未绑定字段需在对应实验
开始前填入并冻结。目前没有把它们标记为已完成的冷启动测试或等预算对比。
本文冻结结果仍由 [export_evidence.py](../tools/export_evidence.py) 的独立清单管理；
上述实时汇总不会自动改动正文图表。

E12 的准备成本记录必须把 `compile_seconds`、`calibration_seconds`、
`search_seconds`、`first_use_seconds` 与稳态 `kernel_ms` 分栏。cuFFT/VkFFT 等
库的 plan/setup 与 cuButterfly 的编译/校准/搜索不是同一种工作，正文只在明确的
协议和 break-even 定义下并列展示。固定 cohort 见
[`preparation_cost.json`](preparation_cost.json)；当前只冻结协议，尚未把未完成
的成本实验写成结果。

## 本轮验证边界

CPU 测试覆盖准备入口隔离、控制变量、整数模数、来源身份、恢复和汇总口径。
新增 runner 的目标 GPU 执行验证在 E1 之后的独占窗口进行。本轮脚本准备与预编译
不算新增 GPU 性能证据，也不将论文中的 pending 实验改成完成。

## RC2 统一箱型图与硬件 profile 队列

`end_to_end_queue.json` 冻结 E13 的三条端到端执行顺序。它复用
`end_to_end.json` 的来源和质量合同，只在质量检查通过后进入 GPU 计时。FNO、QuaRot
和 HEonGPU 的准备、模型/数据获取、适配和完整应用运行分别记录；任何 blocker 都
保持 `blocked`，不能用替代负载填入性能图。

`profile_rc2.json` 冻结 E14 的 A100 80GB 机制 profile cohort。它包含固定核心、
驻留组织、overlap 负例、FWHT 寄存器驻留和 NTT word64 大尺寸案例。正常 timing 与
NCU/NSYS 诊断使用相同 workload、mapping、GPU UUID 和正确性身份，profile 时延不
替代正常性能结果。CPU-only 准备命令为：

```bash
python paper/tools/prepare_profile_cohort.py \
  --manifest paper/experiments/profile_rc2.json \
  --output-dir results/paper_profile_rc2/a100-80gb
```

所有 profiler 结果必须先归一化为 `profile_metrics.csv`，再用
`paper/tools/render_profile_evidence.py` 生成硬件证据图。原始 NCU/NSYS 文件不能
直接进入正文，避免漏掉版本、UUID、mapping 或单位身份。

端到端实测完成后，统一 CSV 必须包含两个 variant 的完整 application rows，随后用
`paper/tools/render_end_to_end_results.py` 生成性能图。脚本会拒绝 `pending`、缺失
baseline/cuButterfly 配对或不完整质量记录，因此端到端协议图不会被误当成性能结果。
