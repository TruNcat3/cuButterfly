# CKKS 完整调用与 NTT 兼容适配

此入口比较 pinned HEonGPU 与使用 cuButterfly NTT 的同一 CKKS 调用。
覆盖密钥切换、平方与重线性化和 rescale、平方后密钥切换、槽旋转。
两种 key-switch 方法由 `--aux-primes 1|2` 选择。它们是完整 FHE 操作和
短计算链的集成测试，不代表完整 bootstrap 或部署中的 FHE 服务。

## 语义与计时

输入、密钥及非蝶形计算由 HEonGPU 提供。适配器处理 word64 的多个 RNS
模数、`X^N+1`、根差异、位反转顺序、逆变换归一化和 caller stream。
默认 `--ntt-execution native-rns` 使用公共 C++ `cuntt::RnsPlan`，在单个
请求内合并多个 RNS 模数及多项式，直接读写 caller 的物理偏移。根、
negacyclic twist、位反转和逆变换归一化融合在两遍变换内，无逐模数
pack/unpack。`--ntt-execution grouped` 保留原有按模数组包、调用
`cuntt::Plan`、解包的路径，供相同输入和计时口径下对照。GPU 上的全部
转换均计入完整调用；新路径的性能结论需由相应运行记录给出。

冷缓存准备阶段验证根表和逆变换缩放，再按真实别名关系调用
`RnsPlan::tune(RnsPlacement::InPlace|Separate, stream)`。
调优在该请求的合法候选中搜索两遍局部 logN 均为 6..10 的拆分、每块
1/2/4/8/16/32 行、64/128/256 线程、radix 2/4/8、第一/第二遍 cross twiddle，
以及位反转边界的逻辑/物理地址遍历。行数仍受设备共享内存容量约束，
自然序双边界不枚举无效的物理遍历开关。negacyclic 请求还枚举融合根
CT/GS 核心，其局部计算核心为 radix2/4、局部 logN 为 2..10、行分组扩展
至 128，并逐遍检查独立 tile 数、共享内存和线程约束。保留两遍共用参数的
候选，另枚举两遍 tile 字数相同、行数随局部长度调整、线程数独立选择的
候选。该有界集合不等于独立逐遍参数的全笛卡尔积，显式 mapping 可重放
其他合法组合。根已嵌入蝶形，无独立 twist/cross 操作。
GPU-NTT-compatible Merge/Barrett provider 也进入同一请求级搜索，
支持范围和参数见 [RNS 接口说明](../../docs/unified_planner.md#explicit-root-rns-ntt-integration)。
合法候选共同排序并复测前三名。
候选使用准备好的 CUDA Graph 计时，首次图上传也在计时外。
日志记录最终映射、候选数、确认延迟、置信度和选择原因。`second_rows_per_block`
和 `second_threads_per_block` 按物理执行顺序指定第二遍参数，0 表示继承；
逆变换也采用此口径。新增字段改变 C++ 结构布局，运行前须重编调用端。
`selection_calibrated=0` 对应 `selection().calibrated=false`，表示未使用
安装级成本模型；`confidence=measured-rns-request` 表示执行了请求级实测
调优。两者可以同时成立，也不证明该候选族之外的全局最优性。
`grouped` 的日志保留原 `cuntt::Plan` 选择依据；旧日志字段名为 `calibrated`。

位反转边界在此次优化前就已直接消费/产生，没有独立的全局排序 kernel。
`physical_boundary_traversal` 只调整线程访问顺序，使相邻线程尽量读写
连续物理地址；它不改变请求的顺序，不增加排序调用，也不改变算术核心。
旧的逻辑遍历保留为可测候选，因为全局合并访存与共享内存冲突可能存在取舍。

每个场景都逐元素比较原生和适配后的整数密文，并解密全部 slots，与
CPU double 明文运算比较 max-absolute 和 relative-L2 误差。浮点误差
只报告，不使用未获用户确认的阈值。独立 NTT 回归另对照 CPU 精确模运算。

`batch` 表示不同输入的独立密文批量提交，不是重复同一密文。默认公共
API 逐密文调用；`native-rns` 合并每次 NTT 请求内的 RNS 工作，`grouped`
按模数分别执行。两种模式都未合并不同密文的公共调用，不能用算子级
大 batch 的收益代替本结果。新增 `--stage-batch` 显式启用下述跨密文合批。

Graph 模式捕获整批完整公共调用，使用 CUDA event 测量。两后端均使用
同一套 Graph 内存兼容层：初始化原内存池后记录分配，再用相同地址捕获，
直到图销毁才归还所有临时存储。它同时避开上游每次访问内存池时的
`cudaFree(nullptr)`。这些改动不改变算术 kernel。测试会更换加密输入、
恢复输入、对照原生 eager 重算，并检查另一条图不会破坏已有输出。

eager 模式另报公共调用到设备完成的主机墙钟时间。上下文、密钥生成、
加解密、计划建立、请求级调优、JIT、Graph 准备和正确性检查均在稳态计时外。
Graph 与 eager 的延迟不能混算。

## 同阶段跨密文合批

`--graph --stage-batch --ntt-execution native-rns` 比较四条完整执行路径：
原生 NTT 循环、cuButterfly NTT 循环、原生 NTT 同阶段合批和 cuButterfly
NTT 同阶段合批。每个密文仍执行上游原始的 keyswitch、平方/重线性化/rescale
或槽旋转流程。Graph lowering 在每次 NTT 处记录完整语义与实际指针，
再合并不同密文中的对应阶段。非 NTT 节点及依赖边原样保留，各密文独占
临时存储。显式阶段签名不匹配、输出重叠、跨请求输入输出依赖或未知图
节点类型都会失败，不能靠猜测 kernel 名称决定合批。

两种合批路径都计入 gather、NTT、scatter 和所有外围计算。cuButterfly
在合批后的真实 RNS 请求上重新调优，日志记录选中映射；原生控制调用
pinned GPU-NTT 的 modulus-ordered 公共接口。两者按 `(q, psi)` 去重根表，
使用相同打包顺序。整数密文验证覆盖四种执行路径、更换输入、恢复输入
和输出存活期，新增独立测试再对照 CPU 检查混合模数与稀疏/别名地址。
原生核心的合批控制也由本轮 Graph lowering 构造，不表示 HEonGPU 已提供
这套跨密文 batch API。相对逐密文循环的收益同时包含 NTT 合批和去除
独立密文之间的多余串行关系，不能全部归于 NTT 局部核心。

CSV 每个配置有三行。`HEonGPU_native` 比较双方循环；
`HEonGPU_native_loop` 比较原生循环与 cuButterfly 合批；
`HEonGPU_native_stage_batch` 比较双方合批。后两行使用同一个 treatment
实测值，不能把三行当作三个独立场景取平均。第三行控制了跨密文组织，
仍包含 NTT 映射/核心选择差异，不属于固定 mapping 的单变量消融。
`--graph-replays K` 将 K 次完整图重放放入一次 event 样本并除以 K，
`--repetitions` 控制样本数，四条路径按轮换顺序计时。准备、编译、调优和
校验仍在计时外，图内必需的打包与回写始终计时。

此选项保留 B=1 负结果。合批是显式可测候选，当前还未自动选择应用层的
循环/合批策略，也不支持任意不同长度或不同 stage 序列的混合密文任务。

## 复现

重用公共 conda 环境和 pinned 依赖构建；原 HEonGPU checkout 和 archive
保持原样。构建入口复制 archive，按符号精确移除 CKKS operator 成员，
再链接含 NTT 替换入口的同一份上游 operator 源码。依赖位置沿用已有
`results/e13_assets` 与 `/tmp/heongpu-e13-*`，缺失会显式失败，不自动下载。

```bash
/home/wt/yes/envs/cubutterfly/bin/python experiments/workloads/run_heongpu_ckks_e2e.py \
  --build-only --output results/ckks_replay/method1_native_rns.csv

CUDA_VISIBLE_DEVICES=GPU-2971e500-d3f6-d7e6-e5f2-cb67a06fff02 \
  results/e13_assets/heongpu-build-ckks-runner/test_heongpu_ntt_adapter

CUDA_VISIBLE_DEVICES=GPU-2971e500-d3f6-d7e6-e5f2-cb67a06fff02 \
  /home/wt/yes/envs/cubutterfly/bin/python experiments/workloads/run_heongpu_ckks_e2e.py \
  --skip-build --graph --rotate --sizes 4096,16384,65536 --batches 1,8,64 \
  --warmup 5 --repetitions 20 --ntt-execution native-rns \
  --output results/ckks_replay/method1_native_rns.csv
```

第二种方法增加 `--aux-primes 2`，输出到单独文件；普通调用移除 `--graph`。
合批实验增加 `--stage-batch --graph-replays 10`，并执行构建产物中的
`test_heongpu_stage_batch`。Graph DAG 重建使用共享 CUDA runtime，以保证
读取、克隆来自动态库的 kernel 节点参数。
长时间调优后可加 `--thermal-guard`：准备完成后在计时外降温、预热；
每轮选拔和独立测量前再次检查，必要时等待温度降至 75°C 或以下且无热降频标志。
完整 batch、轮次和每样本回放数不变，轮间等待不计入 CUDA event 时间。
metadata 的 `thermal_protocol` 标记本轮冷却协议，与旧协议结果分开记录。
每个计时样本结束后检查
NVML 热降频状态，检测到则使该次运行失败；不会静默删除慢样本。
默认降温等待上限为 120 秒，日志记录设备 PCI 身份、温度、时钟和标志。
该流程不改变显卡时钟或功耗设置，抽样不能排除所有瞬时变化。
分组对照使用 `--ntt-execution grouped`，输出到
`results/ckks_replay/method1_grouped.csv` 等新文件，保留旧结果。
每次保留 CSV、完整运行日志与 `.metadata.json`，记录源码、二进制、依赖
hash、GPU UUID、执行参数和返回码。`--skip-build` 的源码/二进制对应关系
标为未自动验证；不能用运行时源码 hash 掩盖旧二进制。
CSV 的 `mode` 区分 Graph/eager，`ntt_execution` 区分原生 RNS/分组执行。
汇总脚本将缺少 `ntt_execution` 列的旧 CSV 视为 `grouped`，并按两种模式
分别统计；新路径不能继承旧 CSV 的测量结论。

测试使用 Q=[50,40,40]、P=[50] 或 [50,50]、scale=2^40、`sec_level=none`。
这些参数用于适配验证与集成性能比较，尚未作生产安全参数验证。

## 当前测量结果

完整同阶段合批接入及四路对照见
[2026-10-19 接入报告](../../results/ckks_stage_batch_20261019/README.md)。
该报告区分原生循环与原生核心合批控制，保留 B=1 负结果和首次热降频
诊断。以下数据是此前逐密文循环路径的历史记录。

短阶段及逐遍映射补齐后，A100 40GB 和 80GB 各完成原范围的 52 点复测。
方法 I/II 几何平均加速比在 40GB 为 0.969x/0.925x，在 80GB 为
0.972x/0.924x；方法 I 的 N=65536 子集分别为 1.050x/1.055x，两卡均为
12 项测得领先，N=16384 仍偏弱。全部整数密文一致。见
[当前优化报告](../../results/e2e_rns_workload_20261018/stage_mapping_optimization.md)
及 [80GB 完整表](../../results/e2e_rns_workload_20261018/ckks_stage_mapping_80gb_comparison.md)、
[40GB 完整表](../../results/e2e_rns_workload_20261018/ckks_stage_mapping_full_comparison.md)。
80GB 曾有一次尝试检测到外部任务，该数据仍排除；上述结果来自后续
带占用监测的干净复跑。不同物理 GPU 的时间不相除估算版本收益。

实验入口要求显式单 GPU UUID，启动前和结束后检查占用，运行中约每秒
采样。发现冲突或查询失败只停止本次测量，记录性能不可用；汇总也检查
旧 metadata 的目标卡占用。抽样不能保证捕获所有瞬时任务。已有数据
不覆盖，复跑使用新输出路径；`--build-only` 不需要 GPU UUID。

以下保留前一阶段的有效同卡结果。

融合根核心新增 radix4 后，同一张 80GB 卡完成 52 组完整调用复测，
方法 I／II 的几何平均加速比为 0.961x／0.927x。方法 I 的 N=65536
子集为 1.016x，36 项中有 9 项胜出；方法 II 的 16 项尚未胜出。
整数密文全部一致。范围、固定映射消融和新 trace 见
[radix4 优化报告](../../results/e2e_rns_workload_20261018/optimization.md) 及
[逐点表](../../results/e2e_rns_workload_20261018/ckks_comparison.md)。

以下记录保留上一轮构建的结果。

后续加入物理边界遍历及融合根 CT/GS 核心后，在 A100 80GB
`GPU-0eae5f07-33de-321b-70a9-ffeae49309e6` 完成新的 52 组配对测量，
整数密文全部一致。方法 I／II 相对原库的几何平均加速比为
0.919x／0.899x，仍未胜出。新版本与旧适配版本、不同物理 GPU 的数据
分别保留，不混合计算版本收益。见
[最新完整结果](../../results/e2e_rns_coalesced_20261018/ckks_comparison.md) 和
[机制诊断](../../results/e2e_rns_coalesced_20261018/README.md)。

2026-10-18 的 A100 80GB 测量完成 52 组 `native-rns` 与 52 组 `grouped`
配对比较，整数密文均与原库完全一致。方法 I 覆盖 N=4096/16384/65536、
B=1/8/64，方法 II 覆盖 N=4096/16384、B=1/8。新路径相对原库的几何平均
加速比分别为 0.777x 和 0.810x，尚无胜出点；相对自身 grouped 路径则
分别提升 2.741x 和 3.359x。后者同时包含 RNS 批处理、边界融合与映射
调优的收益，不能作为单一机制的独立消融。

全部原始数据及逐场景表格见
[本轮结果](../../results/e2e_rns_20261018/ckks_comparison.md)。四份性能文件
使用同一份保存的库和测试二进制；它们的路径及哈希保存在 metadata 中。

独立的 N=16384、B=8 Nsight Systems 诊断显示，额外 GPU kernel 时间的
97.4–97.7% 来自替换后的两遍 NTT。NTT 启动数量已与原库相同，非 NTT
kernel 的名称及启动配置也一致。两遍 NTT 合计耗时仍为原核心的
1.46–1.57 倍。这个结果定位了直接耗时来源，尚不能区分算术效率、
边界地址、访存和映射选择的贡献，也不能据此判定理论存在限制。
[Trace 分析](../../results/e2e_rns_20261018/trace_analysis.md) 包含分解表和
只读复现命令；诊断数据不计入上面的性能汇总。

## 完整 Graph 自动选择与瓶颈采集

在原有 `--graph --stage-batch` 后增加 `--auto-schedule`，会先独立测量循环与
合批候选，再对选中路径执行正确性和性能复测。默认选拔为 7 轮、每轮每候选
5 次完整重放，可用 `--schedule-rounds` 和 `--schedule-replays` 设置预算。
计时包含打包、回写和非 NTT 工作，选拔在稳态性能计时之外。候选都先验证，
选中路径继续更换/恢复输入验证；不以固定 batch 阈值决定赢家。

同名 `.csv.selection.csv` 保存每轮选拔时间和结果，`.csv.samples.csv`
保存五条执行路径的独立复测样本。汇总器检查样本身份、完整性、哈希和主表
一致性，报告相对自身两个固定候选的复测 regret，不把选拔中的最小值当最终性能。
当前只在对象生命周期内复用选择，不冒充安装校准或跨进程调优缓存。

`--scenes square_rescale_keyswitch` 可以只运行指定真实场景，完整工作量不变。
Nsight 诊断使用单一 scene/N/batch 和 `--trace-measurement`，在准备及验证后
才调用 `cudaProfilerStart/Stop`，并以 `ckks_measure/场景/路径/轮次` 标记测量。
配合 `nsys profile --capture-range=cudaProfilerApi --cuda-graph-trace=node`
采集；`experiments/workloads/analyze_ckks_schedule_trace.py` 分解真实 Graph 回放。
带该开关的 metadata 标为 `profiler-diagnostic`，汇总器拒绝作为普通性能结果。

## 适用边界

公共核心位于 `include/cuntt/rns_ntt.hpp`、`src/rns_ntt.cu` 和
`src/ntt_hybrid2d.cuh`。这是新增的 C++ API，尚未纳入公共 C ABI 或通用
planner 的全局 RNS 选择。当前编译候选族支持 logN=12..20、word64、
小于 2^63 的素数模数；HEonGPU 适配器保留小于 2^62 的现有模数限制。
`core_family` 明确区分 Hybrid2D 与 FusedCTGS。后者目前仅支持 negacyclic，
采用成熟 GPU-NTT 的融合根 CT/GS 策略及 radix2/4 核心，由本项目的 stage interval 和
独立行分组模板执行。它不调用外部 GPU-NTT 完整变换。正逆方向分别展开，
确定为 2 的幂的索引用移位/掩码计算。显式 mapping 中与该核心冲突的
radix、cross placement 或遍历开关会被拒绝，不静默忽略。

缓存按执行模式、设备、stream、表指针、排列指针、方向和形状区分。上下文中的表
必须保持不变；所有相关 GPU 工作及 Graph 完成后、销毁上下文前，调用
`clear_cache()`。不支持的 ABI 请求显式失败，不静默回退至原库。

TFHE 的 standalone key switching 是 torus32 分解、查表与累加，不含 NTT。
TFHE bootstrap 内的融合 NTT 仍需要单独的设备接口。CKKS 全 bootstrap 的
长模数链、其他调用变体与安全参数也需分别验证，本入口不替代这些工作。

## Bootstrap profiles and memory boundary

The complete v2 profile is selected with `--parameter-set bootstrap-v2` and
uses the pinned N=65536 Q/P chain. Its preparation can exhaust an A100 80 GB
plan-cache budget before steady-state timing. `bootstrap-v2-small` is an
explicit diagnostic profile that keeps the Q/P chain but changes N to 16384;
the root condition alone does not make the upstream encoding matrices a valid
reduced-N application configuration, so its correctness result must be
reported separately.

For a valid smaller application, `--parameter-set bootstrap-v1-small` uses the
pinned `example/bootstrapping/1_ckks_regular_bootstrapping.cpp` contract at
N=4096, Q bit sizes 60+30×50, P bit sizes 3×60, scale 2^50, and
`(CtoS=3, StoC=3, Taylor=11, less_key_mode=true)`. It is a reduced v1
workload, not an extrapolation of v2. The runner includes complete public-call
timing and exact ciphertext validation for this profile.
