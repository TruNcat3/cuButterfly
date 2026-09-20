# 全应用方法对应、工程贯通与性能提升审计

> 说明：标注为 `local artifact` 的路径是本地实验记录，未随源码发布。

本文件保留重构前的审计状态。后续实现及测量证据见
[当前工程进展](framework_refactor_progress.md)，请勿将下文的历史限制当成最新能力清单。

审计日期：2026-09-12。对象是当前工作区和公共 conda 环境中的 cuButterfly，包含未提交的 v0.9 改动。

结论：**六类数学应用已有实现，混合数据流的若干关键机制也已验证，但全应用、全入口的可执行设计空间与自动优化流程尚未完成对应。仍有明确的提升空间，目前不能认定为理论上限。**

这里的六类应用是 FFT、NTT、FWHT、subset zeta、superset zeta、structured 2×2。`xor-zeta` 保留的是 subset-zeta 的历史兼容语义，不应计为一种已经新增的 XOR 卷积算法。

本次新增审计报告、诊断程序和证据，没有修改 kernel、校准表或已安装的库。基于论文、源代码、已有性能记录和官方资料作出判断；没有把这次的小规模正确性运行当成综合性能复测。

## 1. 方法应如何落到 GPU 上

依据 APPT26 论文源码（local artifact: `../../APPT26/samplepaper.tex`; 本地实验记录，未随源码发布），尤其是 hybrid dataflow、architecture、Algorithm 1 和 dataflow discussion：

| 方法要求 | GPU 工程中需要表达的内容 | 当前对应程度 |
|:--|:--|:--|
| stage/data 各自进行时空展开 | 每一层的阶段组、独立数据数、顺序迭代次数、线程/warp/CTA 归属 | JSON 描述比较丰富；实际参数与约束仍依赖各 backend |
| 多个子图复用有限的计算资源 | 可替换的小计算单元，独立于它的输入输出所有权和调度 | NTT 有较多实现；FFT 新 register-tile 已验证；其他应用尚未共享同样的组合体系 |
| 中间结果按生命周期驻留 | 对依赖闭合的数据域计算寄存器/shared/L2/global 的容量、复用与同步 | 存在可执行局部驻留；还没有统一的生命周期与交换规划器 |
| 写回对齐下一阶段读取 | writer 和 reader 共用索引契约，并按真实访存指令验证事务与端口 | NTT APPT、FWHT XOR、FFT 新前后缀、shared-iterative 分别实现；未统一生成和验证 |
| 独立数据持续供给流水 | 生产者/消费者的服务速率、可并发资源、队列深度和填充排空 | NTT 有细粒度路径；FFT 有两流 overlap；其他应用没有同等范围的端到端搜索 |
| 根据目标硬件寻找优良实现 | 编译可用性、实际资源、数值正确性、成本预测、实测与选择闭环 | 各部分存在，但存在明显断点 |

论文的 bank/offset 公式为 `bank=(i xor floor(i/Npart)) mod (2p)`、`offset=floor(i/(2p))`。它对应的是特定的 URAM 布局和访问调度。GPU 上应保留“生产者写到消费者需要的位置”这一原则，同时考虑元素字节宽度、warp 指令中的线程集合、向量事务和 shared bank。索引是双射只证明无丢失/覆盖，不能单独证明无 bank conflict。

论文在 U280 上以最大 N=2^16、64 位系数验证了设计。A100 上一个 N=2^20 的 FP32 复数数组是 8 MiB，超过单 CTA 的 shared 容量。小计算子图完全可以驻留；跨子图依赖需要多大的存活数据域，必须另行计算。GPU 所有 SM 的 SRAM 总和不能直接视为一个 CTA 的 shared 数组。A100 的容量、资源和异步复制机制见 [NVIDIA Ampere Tuning Guide](https://docs.nvidia.com/cuda/archive/12.6.3/ampere-tuning-guide/index.html)。

batch 增大确实增加独立工作，有利于隐藏延迟；达到驻留、发射或带宽饱和后，收益会趋缓。当前生产者与消费者仍共享 SM、寄存器、shared 和访存资源，不能直接套用 FPGA 独立流水级的 II。应估计 `T ≈ Tfill + (tiles−1)×II_effective + Tdrain`，其中 II 要受各阶段服务速率、共享资源竞争和队列约束共同限制。

因此，使用必要的 global 边界并不否定方法；强制分段、强制不分段或强制 overlap 都不是完整搜索。论文也没有证明有限 GPU 模板集必然得到所有规模、所有数值语义上的全局最优实现。

## 2. 各应用的实际状态与提升方向

| 应用 | 已经实现并能运行 | 尚未完整对应的部分 | 在当前方法内的主要机会 |
|:--|:--|:--|:--|
| FFT | scalar、thread/block/direct codelet、在线分解、局部 shared 迭代、两流 overlap；新 register-prefix + grouped-suffix 路径 | 新路径仅 FP32、两物理组；prefix logN=6/8/10、suffix=10/12 且总 logN≤20；不能与 overlap 任意组合。FP32 多段路径存在，安装搜索未覆盖；FP64 在线路径限两段 | 把寄存器所有权、分组输出、全局布局作为独立组合项；扩展 FP64 与未覆盖形状；定位后缀 shared 交换及系数生成开销 |
| NTT | 32/64 位精确模算术；Hybrid2D、resident、warp-role、ring/packet、cooperative、多子图路径；APPT writer mapping 最丰富 | 与通用 Butterfly 的参数/选择仍分离；APPT-online 有 N=2^20、7+7+6、256 线程等特化约束；安装只有两个固定 smoke 映射 | 参数化子图、发布粒度、data-time、系数共享和模乘策略；把现有细粒度调度变成可复用 lowering，而不是继续增加特定尺寸枚举名 |
| FWHT | 通用 scalar 和 shared 路径；FP32 register/warp/shared 层级实现支持 logN=3..15 | 快速 register 家族线程数与每线程数据量由尺寸固定；低精度/FP64、大于 2^15 的组合能力不对称；`tensor-hadamard` 仍主要是描述 | 小尺寸以相同语义的 copy 为吞吐参照；大尺寸组合 register 子图与合并边界；低精度探索 packed/Tensor Core 并保持明确的舍入契约 |
| subset/superset zeta | uint32 模 2^32 的单侧加减；正逆、stride、in/out-of-place；共享通用调度 | 安装搜索覆盖窄；未形成对应的高效寄存器/warp 子图家族；无足够外部高性能基线 | 生成保留单侧更新语义的 register/shuffle codelet，减少 shared 往返、barrier 和大尺寸 global passes |
| structured 2×2 | 广播/逐阶段矩阵；可逆时支持逆变换；低精度、FP32/FP64；FP32 warp-register 路径 | 搜索主要覆盖一个广播矩阵、一个大小；没有系数供应与局部形状的统一搜索 | 广播系数保留寄存器，逐阶段矩阵分层预取；把矩阵依赖、FMA/向量化和展开程度纳入资源搜索 |
| 复合应用：rank-2、非二次幂、embedding | C API 有轴计划、transpose 和 Bluestein/direct 路径 | 轴间 layout/工作区没有共同最优化；任意长度 FFT 的 cuFFT/Bluestein-cuFFT 核心不能算作自有 butterfly lowering 的性能证明 | 统一轴/阶段的布局契约，避免可消除的格式转换；对完整复合计划搜索并报告核心来源 |

通用算子的 `shared-iterative` 是真实可执行路径：一个依赖闭合阶段组在双缓冲 shared 中迭代，writer 按下一阶段索引写入。但目前限 scalar radix2、最多两个物理组；循环中的每个完整阶段仍有 CTA barrier。它没有实现所有 stage-space/data-time 组合，也不是任意跨 CTA 的片上驻留。

目前 `stage_overlap` 的实际两流调度仅接受 FFT 的两个 shared-iterative 或 cuFFTDx-block 物理组。NTT 自己的细粒度流水另有实现。这两种机制不能合称为“所有应用已经有同一套完整流水”。代码依据：[shared_iterative.cuh](../src/shared_iterative.cuh)、[batch_pipeline.cuh](../src/batch_pipeline.cuh)、[butterfly.cu](../src/butterfly.cu)、[ntt.cu](../src/ntt.cu)。

## 3. 本次确认的工程断点

### A. 用户入口没有共用同一套选择结果

使用公共 conda 环境中安装的 headers/static library 编译诊断程序，在 A100 80GB 上创建同语义的 N=2^20、batch=4、FP32、forward、out-of-place FFT 计划：

| 入口 | 实际结果 |
|:--|:--|
| C++ `ButterflyPlan(auto_select=true)` | `online-reorder/register-tile` |
| C API 默认算法 | `hierarchical-radix4`，selection source 为 static model |

这是实际运行得到的差异，见 api_probe.cpp（local artifact: `../results/method_audit_20260912/api_probe.cpp`; 本地实验记录，未随源码发布）与 evidence.json（local artifact: `../results/method_audit_20260912/evidence.json`; 本地实验记录，未随源码发布）。C API 的 `measured_v100_shape/axis` 仍以旧 V100 profile 决定是否调用自动选择；A100 本地表没有完整接到这里。

C API 的 `MEASURE` 也使用另一套候选名字：若干 scalar/warp 家族加 cuFFT，没有新 register-tile。它用零输入、两次 warmup、五次重复测速，不能等价于安装搜索的正确性确认协议。FFT 可能选择整库 cuFFT，因此必须保留实际核心来源，不能把这样的速度计成自有实现胜出。依据：[c_api.cpp](../src/c_api.cpp)，`measured_v100_shape`、`tune_butterfly`、`choose_butterfly_mapping`。

### B. 本地校准表是单硬件、单批次覆盖，而非可累积注册表

当前 build 和安装库哈希相同。嵌入表只有 A100 80GB 的六个新 FFT 工作负载；此前 A100 40GB 表保存在实验备份中，没有与当前表共存。对 FWHT、subset/superset、structured 的诊断请求，C++ auto 因没有本地点而拒绝；显式实现仍可运行。C API 对它们使用自己的静态 fallback。

`local_selector_data.write_header()` 每次从输入结果构造一个设备的整张头文件，缺少跨工作负载增量合并、跨设备并存的存储层。NTT 的本地映射应用仍固定为 Hybrid2D/radix4/fused，并未像 Butterfly 一样序列化完整子图与边界配置；扩大 NTT 搜索前必须先打通此处。依据：[mapping_selector.cpp](../src/mapping_selector.cpp)、[local_selector_data.py](../scripts/local_selector_data.py)。

### C. 可执行设计点与描述文件再次发生漂移

通用架构 schema 自检通过，但它不包含已经编译的新 `register-tile` core、`shared-iterative` runtime binding，也没有覆盖公共 API 已支持的全部 FP16/BF16 语义。FFT 架构文件里的硬件 profile 仍只有 V100。

这是“描述文件彼此一致”与“描述对应真实工程”的差别。`check_v09_architecture.py` 验证层名及路径存在，不能证明源码的依赖方向或可执行覆盖。当前 unified IR 主要由已经选定的 backend/config 投影出来，还不能反向统一生成所有执行路径。

另有两处 IR 单位问题：NTT radix4 被填入每一个逻辑 `stage_arities`，而通用 Butterfly 保持逻辑二叉图并单列 processing radix；NTT 的 shared 估算在投影和估算器两处乘 `data_space`。探针中 data_space=2 得到 16,384 字节字段、32,768 字节估算。应统一这些字段的单位并与实际 kernel 分配核对；此处没有把估算差异宣称为已证明的运行时错误。

### D. 安装枚举不是完整空间，非 FFT 存在实际漏枚举

`butterfly_candidates()` 构造了 `(6,8,10)`，随后使用 `stages[0]`，没有遍历阶段数。实际每个非 FFT 工作负载只得到 18 或 19 个候选；hierarchical/online 的 local_stages 全是 6。也没有枚举 shared-iterative、stage-pipeline、writer-aligned、独立组映射等。

FFT 的编译库存组合较丰富，但安装主搜索只覆盖两段 online 家族；whole-transform/direct 等部分点来自单独 smoke/incumbent，不是共同枚举。目前一个 FP32 N=2^16、batch=64 单元有 20,956 个枚举候选，默认筛选预算 64，仅约 0.31%；N=2^20、batch=16 为 5,325 个。以上包含需要后续合法性/正确性检查的点，不是全部可执行点数。

`--search-budget 0` 只穷尽该枚举器的产物，不能补回未枚举的家族或没有实现的 lowering。依据：[calibration_space.py](../scripts/calibration_space.py) 和本次枚举证据（local artifact: `../results/method_audit_20260912/evidence.json`; 本地实验记录，未随源码发布）。

### E. “现场编译”开关尚未连接到自动搜索

当前 CMakeCache 确实设置 `CUBUTTERFLY_FFT_CODEGEN_MODE=on-demand`。但源码中该变量只声明、校验和由安装脚本传入；没有驱动搜索遇到缺失候选时编译。

`compile_fft_variant.py` 可以手动扩大 cuFFTDx online spec，然后重新配置/构建整个工程。它是有用的显式构建工具，但不是自动 JIT 补全，也不覆盖 NTT/其他通用算子。已有 build 的安装脚本还可能在没有重新配置分支时忽略新传入的模式/架构选项。依据：[CMakeLists.txt](../CMakeLists.txt)、[安装脚本](../scripts/install_with_hardware_profile.sh)、[编译工具](../scripts/compile_fft_variant.py)。

### F. 硬件测量、成本模型、搜索决策尚未形成闭环

安装已测 global/shared/barrier、一个等效 butterfly 核及 stage-pipeline 的若干 Us，并按设备识别生成数据。这些是有效的硬件事实，但还不足以区分复杂乘法、精确模乘、无系数加减和矩阵更新。

当前回归在候选测量后拟合，不负责主搜索排序/剪枝。本次直接调用 `feature_vector()` 确认：只改变 direct boundary 的 transpose、element stride、direction、per-group core、linear→writer-aligned，会得到相同特征；尚无 register-tile 家族特征。合成特征探针只证明模型区分能力不足，不证明每个合成组合都合法。

历史 A100 成本模型仍有 `calibrated-local-warning`。例如 `local_install_calibration_a100_gpu0_v7` 的留一中位相对误差约 45.3%；winner accuracy 在用于拟合的数据上计算，不能替代留出形状/设备的排序验证。模型也不能因为能预测总毫秒数，就证明已找到正确的驻留层次或流水 II。

还需要按计算核心/位宽标定指令吞吐、寄存器压力、shuffle/shared 事务、同步和系数供应，并显式区分 global 地址流量、L2 命中与 HBM 流量。依据：[initialize_hardware_profile.py](../scripts/initialize_hardware_profile.py)、[fit_local_cost_model.py](../scripts/fit_local_cost_model.py)。

### G. 安装后的独立校准工具不完整

公共环境执行 `bin/calibrate_local_hardware.py --help` 直接报 `ModuleNotFoundError: No module named 'calibration_space'`。CMake 安装了四个入口脚本，没有安装其全部辅助模块、配置和所需 benchmark 工具。仓库内安装流程可用与脱离源码后的迁移校准可用是两个验收项。

支持文档还有旧结论，例如“自动校准只有 V100”；其表述需要由实际 capability 和 selector inventory 生成，减少人工同步遗漏。上述 G 及 A/B/C 是影响交付的工程问题，不需要改动蝶形数学方法。

## 4. 性能证据能支持多大的结论

| 证据 | 可以得出的结论 | 不应扩大的范围 |
|:--|:--|:--|
| A100 80GB 最新 FFT，N=2^18/2^20、batch=1/4/16，FP32 forward contiguous out-of-place | 固定参数规则的平均 `TcuFFT/TcuButterfly` 为 97.63%；两种已冻结映射中择优为 99.50%；严格正确性通过 | 只有六个单元；不是所有 FFT，更不是所有算子、A100 40GB 或所有 API 的自动性能 |
| 相同 FFT 前缀调度下替换 native 与 cuFFTDx Thread 算术 | 后者在两个控制点略快；不能把瓶颈归咎于 cuFFTDx 整体不适配 | 控制点使用相同旧后缀；不能把不同分段/后缀的变化归为纯算术收益 |
| A100 40GB 历史 NTT 对 GPU-NTT，两点 | N16/b64 natural：0.235039 vs 0.320563 ms，约 1.36×；N20/b4 bit-reversed：0.254464 vs 0.324823 ms，约 1.28× | 两种输出语义不同，分别配对；未覆盖全部模数、位宽、方向及新版本基线 |
| A100 40GB 历史 FWHT 对 Dao，FP32 两点 | N8/b16384 接近；N15/b128 有优势记录 | 部分 trial 标为 unstable；没有低精度/FP64/更大尺寸的全面外部结论 |
| zeta / structured | 已有正确性及内部路径差距证据 | 当前表格许多 `1.0×` 的 basis 是内部最好实现，不能理解为追平外部最优库 |
| FP64 FFT | 后续校准已找到优于早期 scalar 的 cuFFTDx 配置，N16/b64 的一组校准中位数为 0.310989 ms | 早期 0.394× cuFFT 不是最新配置的完整评价；需要同场配对复测，不能跨运行直接拼接比值 |

来源：FFT 最终报告（local artifact: `../results/fft_register_tile_20260912/report.md`; 本地实验记录，未随源码发布）、最终目标计算（local artifact: `../results/fft_register_tile_20260912/final_api/goal_gate.json`; 本地实验记录，未随源码发布）、A100 40GB 外部配对表（local artifact: `../results/a100_c0_rebenchmark_20260910_external_tables.md`; 本地实验记录，未随源码发布）、综合内部表（local artifact: `../results/a100_c0_rebenchmark_20260910_tables.md`; 本地实验记录，未随源码发布）、后续校准（local artifact: `../results/theory_alignment_a100_20260910/report.md`; 本地实验记录，未随源码发布）。

最新 NCU 显示 FFT 新 prefix 的 global sector efficiency 为 100%/100%，long scoreboard 约 9.7%；grouped suffix 的 SM 吞吐约 61.6%、DRAM 吞吐约 69.9%，仍有大量 shared store bank conflicts。值得继续做地址/指令级归因，但不能用 `100%-SM吞吐` 直接计算可实现的加速比例，也不能仅凭 conflict 总数认定它就是最大瓶颈。先区分 adapter 交换与 imported codelet 内部交换，再做保持调度不变的消融。

## 5. 调研后建议的提升路线

优先把 FFT 这次成功的经验抽象为“计算单元 + 所有权 + 布局 + 生命周期 + 调度”的共同接口。库导入核心和自己生成核心都可以作为候选；框架应控制组合。cuFFTDx 官方提供 Thread/Block、寄存器/shared 输入等执行接口，Block shared 的复用还有明确同步要求，适合作为受契约约束的组件，而非天然固定的整阶段布局：[NVIDIA cuFFTDx execution methods](https://docs.nvidia.com/cuda/cufftdx/api/methods.html)。

FWHT 的性能路线要分数值契约。Dao 的官方实现覆盖 FP32/FP16/BF16、最大 32768，并把同形状 memcpy 作为吞吐参照：[Dao FHT](https://github.com/Dao-AILab/fast-hadamard-transform)。低精度还应纳入 HadaCore 的研究比较；官方实现使用 MMA 与异步加载，当前入口限 FP16/BF16、二次幂不超过 32768：[PyTorch HadaCore 源码](https://github.com/meta-pytorch/applied-ai/tree/main/kernels/cuda/inference/hadamard_transform)。这些提供可实现方向，不构成对本库未测精度/尺寸的速度承诺。尤其本库低精度按逻辑蝶形舍入，不能无说明地改成整块 FP32 累加后一次舍入。

NTT 后续比较应固定基线提交、模数、32/64 位、正逆和输出顺序。GPU-NTT 官方当前同时提供 Merge/4-Step、32/64 位和 Barrett 路线；历史两点赢某一个 baseline 不能代替这组覆盖：[GPU-NTT 官方仓库](https://github.com/Alisah-Ozcan/GPU-NTT)。对 zeta/structured，应同时使用精确 reference、优化过的相同操作图基线和带宽/指令上界；稠密矩阵乘不是相同计算复杂度的唯一参照。

| 次序 | 工作包 | 完成标准 |
|:--|:--|:--|
| P0：先交付现有性能 | C/C++/benchmark/安装共用候选记录和 selector；按硬件与语义增量合并；修复独立安装工具 | 同请求各入口解析出相同完整映射；新增 FFT 不删除其他应用/设备记录；脱离源码仍能查询、校准和回放 |
| P1：统一可执行描述 | schema 对接 codegen inventory 和各 operator lowering；计算语义与处理 radix 分离；布局、数据宽度、资源单位一致 | 每个设计点可追踪 described→feasible→compiled→executable→correct→measured；报告拒绝原因；新 kernel 必须出现在同一库存 |
| P2：补齐搜索与编译流程 | 修复非 FFT 阶段枚举；统一 whole/two/multi-group 与 NTT；缺失的合法点按研究/安装编译预算构建并缓存 | on-demand 实际触发并复用编译产物；编译时间独立于执行时间；预算遗漏与 lowering 缺失分开统计 |
| P3：复用高吞吐 lowering | 抽取寄存器 codelet、writer-aligned 交换、grouped 输出骨架；先扩 FFT FP64、大 FWHT、zeta，再参数化 NTT 专用调度 | 跨至少两类算子复用同一所有权/布局机制；独立改变 core、tile、分段和 schedule 时都能验证与测量 |
| P4：模型引导优化 | 分层服务速率/资源/传输校准；用误差和不确定性分配测量预算 | 留出尺寸、batch、数值语义与 GPU 验证排序；报告相对已测最好点的 regret，不能只报训练命中率 |
| P5：综合性能验收 | 固定版本、同 GPU/显存、同语义，随机交错测量与正确性前置；覆盖所有用户入口 | 按应用/精度/大小分层报告 geomean、最差点及低于阈值比例；显式映射与默认调用各报告一份 |

P3 的局部实验可以与 P1/P2 并行推进，但新胜出点必须进入共同库存与回放流程。阶段流水、批量 overlap、async copy 都应作为可比较的选项；当额外队列、shared 缓冲或寄存器压力降低吞吐时，应允许搜索选择更简单的执行计划。

这里不预设能再提高某个百分比。较确定的是 P0/P2 可以恢复已有实现和未搜索参数的价值；新 core、FP64、低精度、复杂流水的潜力则需要逐项消融确认。

## 6. 本次验证和复现

硬件：A100 80GB PCIe，UUID `GPU-0eae5f07-33de-321b-70a9-ffeae49309e6`。公共环境 `/home/wt/yes/envs/cubutterfly`。

- 20/20 显式正确性探针通过：N=4096、batch=3；FFT/FWHT FP32/FP64、subset/superset/legacy zeta uint32、structured FP32，shared-iterative 6+6/writer-aligned 的正逆；另含 NTT Hybrid2D 的 32/64 位正逆。每次均使用已有 benchmark 的 `--verify`，这些运行只验证指定契约。
- 43 个 host tests、61 个 subtests 通过，涉及架构描述、安装搜索、成本模型和硬件/application profile。描述自检通过而跨层审计发现漂移，表明现有测试仍需补充端到端契约检查。
- 实际运行了安装库的 C/C++ 选择探针、编译库存查询、安装校准命令的 `--help`；成功和失败结果均保留。
- 已安装与 build 的静态库 SHA256 相同：`92bf141e34a651e12feb24404d8e739965c882d2d23e02fb6ef9df03ac65b0fa`。源文件哈希及完整命令记录在 evidence.json。

复现诊断程序：

```bash
g++ -std=c++17 results/method_audit_20260912/api_probe.cpp \
  -I/home/wt/yes/envs/cubutterfly/include -I/usr/local/cuda-12.4/include \
  -L/home/wt/yes/envs/cubutterfly/lib -lcuntt \
  -L/usr/local/cuda-12.4/lib64 -Wl,-rpath,/usr/local/cuda-12.4/lib64 \
  -lcufft -lcudart -lcudadevrt -lpthread -ldl \
  -o /tmp/cubutterfly-method-audit-api
/home/wt/yes/envs/cubutterfly/bin/python results/method_audit_20260912/collect.py \
  --build-dir /home/wt/git/cuButterfly/build-a100-cufftdx \
  --prefix /home/wt/yes/envs/cubutterfly \
  --api-probe /tmp/cubutterfly-method-audit-api \
  --device GPU-0eae5f07-33de-321b-70a9-ffeae49309e6 \
  --check-lowerings --output /tmp/cubutterfly-method-audit-rerun.json
```

历史 [2026-09-10 审计](theory_alignment_audit.md) 保留其时间点的发现；本文反映 register-tile 合入后的状态，不将历史缺失项与后来已完成的局部实现混为一谈。
