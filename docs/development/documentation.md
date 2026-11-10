# 文档维护约定

文档按任务导航、稳定指南、实验协议与结果、历史归档分层。这个分层借鉴
Ascend-FFT 的[任务导航与指南分层](https://github.com/TruNcat3/ascend-fft/tree/8b0b19c92025b9fb58e2b8bc8e2c6c2c75fc1acc/docs)，
参考版本为 `8b0b19c`。本仓库保留已有路径、实验身份与报告合同。
入口是 [docs/README](../README.md)，工程定位见[工程地图](repository-layout.md)。

## 面向读者的内容边界

首页和稳定指南介绍研究问题、方法、能力、安装与通用使用流程，不放当前任务分工、
等待 GPU、某台服务器路径、平台实验交接或本轮测试数量。平台专用工具链和实际
能力限制写在对应硬件参考页；实验日期、样本数和测量身份写在结果报告中。删除
开发过程叙述不等于删除数值合同、已知限制或负结果。

## 放在哪里，维护什么

| 层次 | 本仓库入口与维护方式 |
| --- | --- |
| 任务导航 | [文档索引](../README.md)回答“我要做什么、先读哪页”；导航只给链接和简短范围，不复制支持表或实验数值。 |
| 稳定指南 | [研究原则](../research_principles.md)、[统一语义](../unified_semantics.md)、[统一 planner](../unified_planner.md)、[硬件校准](../hardware_profile_install.md)各维护自己的合同；修改行为时同步受影响的 canonical 页面。 |
| 实验协议与结果 | [benchmark 合同](../../benchmark/README.md)和[reporting contract](../reporting_contract.md)规定计时、范围、状态与来源；专题报告按既有合同组织，不新增竞争模板。 |
| 历史归档 | [归档索引](../archive/README.md)保留可追溯的旧版本与报告入口；历史路径不因导航重整而批量移动。 |

日期报告描述指定构建、硬件、范围和协议下的一次证据，不能当作实时支持状态。
稳定指南引用报告时说明其证明的范围；报告失效或勘误时保留原 cohort，并链接修订。
当前工作进度指向[框架进度](../framework_refactor_progress.md)对应阶段，能力查
[支持矩阵](../support_matrix.md)，不从某份日期最新的报告推断全部能力。

作者与软件引用统一维护在 [about](../about.md) 和 [CITATION.cff](../../CITATION.cff)。
新页面引用这些入口，不复制会逐渐分叉的作者列表；报告中的实际贡献者和匿名稿
仍遵循[报告合同的署名规则](../reporting_contract.md#署名与引用)。

## 数据、报告与论文分开

- 原始测量保留在各自结果目录，并绑定命令、scope、build、硬件与计时协议；不覆盖旧 trial，也不把诊断数据改称正式结果。
- 派生表格和图由有记录的脚本从明确输入生成，保留源路径与 provenance；不要手改生成表来修正文案或替换缺测值。
- 论文 source 独立维护，不纳入软件仓库。实验入口见 experiments/README.md，公开快照见 [current_refresh](../../experiments/snapshots/current_refresh/snapshot.json)；运行中的目录不自动成为论文证据。

状态标签、完整报告字段、负结果、unsupported 和复现要求均以
[reporting contract](../reporting_contract.md)为准。源码发布未附带的历史工件使用
[明确路径清单](../../config/repository_local_artifacts.json)说明；登记不代表内容已验证。

## 修改与检查

新增指南先查已有合同并链接 canonical 页面；只在承担该合同的页面定义术语和规则。
更新入口时保留旧链接兼容，必要时在原位置留迁移说明，不批量移动报告、生成物或数据。
相对路径以当前文件为基准；外部资料应指向明确版本，不能用外部平台的结果支持本平台结论。

文档改动检查相对链接、引用路径与涉及的命令；根入口和递归 `docs/` 链接检查使用
`python scripts/check_repository.py`。该检查会单独报告已登记但未附带的本地工件，
不会证明实验数值有效。仅改文档不需要安装、重新校准或执行 GPU benchmark。

首页概念图的可编辑源是 [Graphviz DOT](../../figures/cubutterfly_concept.dot)，
对应 [SVG](../../figures/cubutterfly_concept.svg)。使用图稿工具 Graphviz 重新生成后，
按首页实际显示宽度检查文字和连线；Graphviz 不属于算子运行依赖。

```bash
dot -Tsvg figures/cubutterfly_concept.dot -o figures/cubutterfly_concept.svg
```
