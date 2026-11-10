# 工程地图

本页用于从研究对象定位实现入口，不是支持矩阵或源码穷尽审计。方法定义见
[设计空间](../butterfly_design_space.md)，实现与证据边界见
[研究原则](../research_principles.md)、[统一 planner](../unified_planner.md)和
[支持矩阵](../support_matrix.md)。同一接口不意味着每个算子已经实现全部组织、核心或 lowering。

## 从研究对象到代码

| 研究对象 | 关键入口 | 阅读目的 |
| --- | --- | --- |
| 算子语义与负载 | [C descriptor](../../include/cubutterfly/cubutterfly.h)、[C++ Transform / PlanOptions](../../include/cubutterfly/plan.hpp)、[语义检查](../../src/semantic_contract.cpp)、[Python 前端](../../python/cubutterfly/__init__.py) | 区分运算、精度、方向、排列与 rank、shape、batch、stride；合同以[统一语义](../unified_semantics.md)为准。 |
| 公共组织与约束 | [mapping](../../include/cubutterfly/mapping.hpp)、[shared 组织枚举](../../src/shared_organization_candidates.hpp)、[驻留 mapping](../../src/resident_mapping.hpp)、[硬件资源模型](../../src/hardware_model.hpp) | 从分段、物理执行组、线程与驻留组合定位合法性；算子能力不能替代公共资源约束。 |
| 可替换 core 与 lowering | [FFT core 合同](../../src/fft_local_core_contract.hpp)、[cuFFTDx provider](../../src/cufftdx_provider.hpp)、[算术 traits](../../src/operator_traits.cuh)、[shared lowering](../../src/shared_iterative.cuh)、[NTT 候选](../../src/ntt_template_candidates.hpp)、[xD axis](../../src/xd_axis_fft.cu) | 支持域、数值合同、组织接口与实际执行分别核对；见[核心集成策略](../core_integration_strategy.md)和[xD lowering](../xd_lowering.md)。 |
| 静态硬件事实与服务 profile | [硬件 seed](../../config/hardware/README.md)、[静态 profile 工具](../../scripts/static_hardware_profile.py)、[校准入口](../../scripts/calibrate_hardware.py)、[阶段服务模型](../../scripts/stage_service_model.py) | 容量事实、测量样本、拟合服务与可执行覆盖分开；消费策略以[ADR 0001](../adr/0001-static-profile-driven-planning.md)及[硬件校准指南](../hardware_profile_install.md)为准。 |
| selection、编译缓存与公共调用 | [公共 C API / planning](../../src/c_api.cpp)、[registry](../../src/plan_registry.cpp)、[mapping selector](../../src/mapping_selector.cpp)、[静态排序](../../src/static_candidate_ranking.hpp)、[JIT module](../../src/jit_module.cpp)、[module API](../../include/cubutterfly/module.h) | 区分注册结果、模型估计、测量选择与编译产物；缓存存在不代表语义、构建和策略兼容。 |
| benchmark 与论文冻结证据 | [benchmark 合同](../../benchmark/README.md)、[scope](../../benchmark/scopes/a100_80gb_research_v1.json)、[报告合同](../reporting_contract.md)、[论文证据入口](../../paper/evidence/README.md)、[冻结快照](../../paper/generated/current_refresh/snapshot.json) | 完整调用对照、同核心组织消融和应用证据分别取证；派生快照不代替原始测量。 |

rank-one 与 xD 当前仍有不同的选择路径；公共接口收敛不代表内部已成为单一实现。
`src/c_api.cpp` 同时承担前端、候选选择与部分 xD 编排，是后续解耦的实际边界；
本地图只标明入口，不据此修改实现或推断路径已经统一。

## 顶层目录职责

| 目录 | 职责 |
| --- | --- |
| `include/` | 对外头文件和底层 C++ 配置；`cubutterfly/` 是公共 API，`cuntt/` 保留既有核心与执行配置。 |
| `src/` | CUDA 核心、lowering、语义验证、plan、registry、硬件约束和 JIT 实现。 |
| `python/` | Python 包、公共计划封装及 Python 集成辅助路径。 |
| `scripts/` | 安装、校准、调优、实验编排、分析、绘图及仓库检查；不是第二套公共算子 API。 |
| `config/` | 声明式空间、负载、平台规则、硬件 seed 与发布工件登记；有日期的配置保留所属实验身份。 |
| `benchmark/` | 研究评测合同、scope、manifest、runner 注册及 Python 评测流程。 |
| `benchmarks/` | 外部基线、语义桥接和专用 CUDA 对比程序；注意与单数目录职责不同。 |
| `apps/` | 可执行探针、微基准及实验程序，例如 [plan probe](../../apps/cubutterfly_plan_probe.cpp)；应用集成也可能位于 adapters 和 scripts。 |
| `adapters/` | 面向外部应用的集成和补丁，例如 [HEonGPU NTT adapter](../../adapters/heongpu/ntt_adapter.hpp)；需计入适配边界成本。 |
| `tests/` | CPU 合同测试、构建/缓存检查及需要设备的正确性测试；按改动选择验证，文档修改不触发 GPU 测量。 |
| `docs/`、`paper/`、`results/` | 稳定指南与报告、论文 source/冻结证据、本地测量工件；维护方式见[文档约定](documentation.md)。 |

## 扩展时接到哪里

- **新增算子**：先按[统一语义](../unified_semantics.md)补 descriptor、语义验证和参考校验，再接算术 traits / provider 支持域与公共组织候选，贯通 lowering、实际 mapping、registry/cache 身份和公共调用。缺失 lowering 明确报告；不要为算子名称另建搜索流程。
- **新增核心**：按[核心模板合同](../core_template_contract_20261013.md)及[集成策略](../core_integration_strategy.md)声明精度、支持域、资源与原始未分段执行点，再接候选、显式 mapping、编译分派和缓存/服务身份；验证核心替换与固定核心组织收益时分别取证。
- **新增硬件**：按[安装与校准指南](../hardware_profile_install.md)生成匹配设备类别的容量与服务文件，接既有 profile 消费和资源门禁；运行时设备安全检查保留。不要复制另一型号的测量或把型号分支散入算子实现。

扩展的验证和报告沿[研究原则](../research_principles.md)与[报告合同](../reporting_contract.md)，本页不另设验收或审批流程。
