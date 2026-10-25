# 研究报告与证据合同

本页把 cuButterfly 的实验记录整理为一个稳定的阅读和复现入口。它借鉴了
[Ascend-FFT 的文档索引](https://github.com/TruNcat3/ascend-fft/blob/master/docs/README.md)、
[测量协议](https://github.com/TruNcat3/ascend-fft/blob/master/docs/benchmarks/methodology.md)
和[项目与作者页](https://github.com/TruNcat3/ascend-fft/blob/master/docs/about.md)的报告分层方式，
但不把不同硬件、不同构建或不同计时协议拼成一个新结果。
当前范围仍以 [A100 80GB research scope](../benchmark/scopes/a100_80gb_research_v1.json)
为准；旧 V100、A100 40GB 和当前 A100 80GB 记录按各自 cohort 分开。

## 推荐阅读顺序

1. [实验原则](research_principles.md)：明确语义轴、可替换核心、校准、搜索和两类消融。
2. [benchmark contract](../benchmark/README.md)：确认 scope、计时边界、baseline 和缺口政策。
3. 专题报告：先看范围与协议，再看结果表、trace/profile 归因和限制。
4. [Reproducibility](reproducibility.md)：使用报告中绑定的命令和原始数据路径重跑。
5. [论文证据索引](../paper/evidence/README.md)：查看论文当前接纳的冻结快照。

## 结果目录与状态

新实验使用带日期和目标硬件的独立目录，例如
`results/e2e_current_20261017/`。报告不覆盖旧目录；修复或换协议必须创建新 cohort，
并在 scope 中写明 `revision_of` 或 `result_ref`。

每个结果按下面的状态标识：

| 标签 | 含义 |
|---|---|
| `measured` | 按声明命令直接在指定硬件上测得，含原始 trial 或 counter 记录。 |
| `derived` | 从 measured 数据按公开公式确定性计算出的统计量。 |
| `external` | 固定版本的第三方实现，在匹配语义和时间边界下运行。 |
| `unsupported` | 当前 lowering 或 provider 没有实现，必须保留为缺口。 |
| `diagnostic` | 用于定位机制或工程问题，不进入 headline 几何均值。 |
| `placeholder` | 仅表示未来范围或图表位置，没有性能结论。 |

`unsupported`、`missing-baseline` 和 `failed-correctness` 都要出现在报告中，不能通过换精度、
缩小 batch、flatten rank 或替换算子代跑后从分母删除。

## 标准报告结构

每个可公开引用的报告按以下顺序组织。短报告可以压缩段落，但不能删除元数据或限制。

### 1. Scope and question

给出报告 ID、版本、状态、研究问题、硬件型号和显存、GPU UUID、CUDA/驱动、源码与库 hash，
以及算子、精度/模数、rank、长度、batch、方向、布局、输入输出顺序、placement 和归一化。

### 2. Protocol

明确 baseline、计时边界、warmup、repetitions、trial 数、统计归约、Graph/eager 模式、
编译和 plan 构造是否排除，以及转换、拷贝、scale、padding 是否包含。安装和 tuning 产生的
时间单独报告，不能混入 steady-state latency。

### 3. Result

先给完整调用的绝对 latency 或 throughput，再给 paired speedup。主表至少保留 workload、
operator/core、baseline、batch、latency、speedup、correctness/status 和 source path。
箱型图中的重复调用必须说明是 process replicates、steady-state samples 还是 trial range，
不能把同一进程的调用样本写成独立置信区间。

#### 设计点概要

每次性能汇报用下面的顺序描述实际执行点，适用于所有蝶形算子。多个
阶段的参数写成有序列表，尺寸不同时不压缩成一个含糊的 tile 数字。

`a 个大阶段 [划分] — tile 点数 [t1,…,ta] — 每 CTA 独立子图数 [c1,…,ca] — 线程数 [T1,…,Ta] — 每 CTA 连续 tile 组数 [q1,…,qa] — 流水范围/深度 — 核心 [core1,…,corea] — 布局交接`

主表可用设计点 ID 引用一张紧邻的概要表，明细仍保存完整实际 mapping。
同时注明 N、B、精度、方向、硬件、计时模式和选择来源。应用表保持
「场景｜类型｜对照｜加速比」，另附内部蝶形请求及其设计点 ID。
FFT 的类型列使用 `fft-h` 或 `fft-l`，精度仍单列。低精度输入、内部 FP32
计算的路径属于 `fft-l` 并显式标明；高低精度分组不替代存储/乘法/累加
精度合同，也不把同一组内不同精度的外部实现互相替代。

- 大阶段数来自实际执行组；例如 `factor-streamed` 的大阶段由
  `factor_partition` 决定，不能将其 `stage_partition=[logN]` 误报为一次
  无中间交换的单阶段执行。
- `t` 是一个独立局部子变换的点数，`c*t` 才是一组 CTA 工作的元素数；
  `q` 是同 CTA 连续处理的组数。外部 batch 切片另列。
- launch CTA 总数可按实现确定性推导并标为 derived；实际同时驻留的
  CTA/warp 数及吞吐须来自 profile，不能用总 CTA 数或线程数替代。
  方法层面的 `Us/Ud/Ts/Td` 有明确映射证据时另列，不能从线程数猜测。
- 分别说明阶段内预取、子图阶段角色流水、跨大阶段 overlap 和跨 batch
  overlap。`stage_overlap=false` 只表明对应调度选项关闭。输入预取不等于
  写回也已异步化，tile 内数据驻留不等于 CTA 跨 tile 持久执行。
- 记录写回到消费者布局的位置、独立 transpose/scale 的实际次数，以及
  寄存器/shared/全局暂存边界。黑盒核心或库内部配置不可见时写未知。

例如本轮长一维 `N=65536, B=16` register-radix4 点为
`2 阶段 [6,10] — tile [64,1024] — 子图/CTA [4,4] — 线程 [32,256] — 连续组 [1,1] — 无显式跨 tile 预取 — 核心 [register-radix4,cuFFTDx-block] — prefix 直接写 suffix 布局`。
对应 launch CTA 数是 `[4096,256]`，不是实测并发 CTA 数。

### 4. Attribution and limitations

用 kernel/trace/profile 解释差异，但把 profiler 计数与完整调用时间分开。记录未覆盖组合、负点、
适配路径的真实开销、错误分布和不支持原因。机制消融只改变一个因素；核心替换和数据流替换
分别报告，不能把核心收益写成组织收益。

### 5. Reproduction and provenance

给出复现命令、输入/权重 seed、原始 JSON/CSV、派生表格和图表 hash。报告应能从结果文件回溯到
scope、manifest、build identity 和生成脚本。论文只读取已经接纳的 evidence snapshot，不读取
正在运行的目录或手工改写的生成表。

## 署名与引用

软件和实验工件的署名以 [项目与作者](about.md) 和 `CITATION.cff` 为准；报告中的
`maintainer`、`experiment owner` 和 `verifier` 只能填写有记录的实际贡献者，不能根据运行主机或
脚本作者推断姓名。第三方库保留其原始作者、版本和许可证。论文匿名稿使用独立的匿名化作者行，
不能把内部作者信息写回匿名 PDF。

## 当前端到端例子

`docs/e2e_b16_trace_20261017.md` 是第一份按此合同整理的完整案例。它保留了 B16 静态 fallback、
显式 2 GiB tuning 和自适应默认三种状态，明确记录构建 hash、GPU UUID、候选来源、Graph 计时、
kernel trace、负结果及复现路径。该案例的自适应结果可以作为论文的独立 2D Poisson endpoint，
但不能覆盖旧的 9 点 image-filter cohort。
