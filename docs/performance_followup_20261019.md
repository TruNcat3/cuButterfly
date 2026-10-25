# 性能弱点复核与诊断记录，2026-10-19

本记录汇总已保存的自动选择复测及两组 trace，保留早期合批负结果和诊断计划
的历史边界。36 点独立普通复测已完成；B32 trace 本轮不追加。
结论遵循
[研究原则](research_principles.md)：核心、映射、应用组织分别归因，旧构建
trace 只解释其自身测量，profiler 时间不替代普通计时。

## 2026-10-19 自动选择复测状态

`GraphSchedule` 已接入完整 CUDA Graph 调用。它在同一显式 stream 上轮换候选
顺序，用未计时 warmup 和 CUDA event 中位数选择 `loop` 或 `stage_batch`，再用
独立复测报告自动路径；选择时间单列，不混入稳态延迟。该选择是当前请求级
实测，不是安装级持久缓存或全局最优声明。

| 请求 | 状态 | 自动选择 | 相对原生逐密文调用 | 相对原生阶段合批 |
|---|---|---|---:|---:|
| 方法 I，N=65536，B=1，4 场景 | 有效，20/20 配对，整数密文精确一致 | 4 个均为 `loop` | 几何均值 1.074× | 几何均值 1.305× |
| 方法 I，N=65536，B=8，4 场景 | 有效，20/20 配对，整数密文精确一致 | 4 个均为 `stage_batch` | 几何均值 1.205× | 几何均值 1.154× |
| 方法 I，N=65536，B=32 cooled，4 场景 | 有效，新构建与逐轮冷却协议单列，整数密文精确一致 | 旋转 `loop`，其余 3 个 `stage_batch` | 几何均值 1.165× | 几何均值 1.134× |

B=1 的自动加速范围为 1.033–1.145×，B=8 为 1.151–1.273×；独立复测相对
同请求固定候选的 regret 为约 -0.016%–0.798%。B=32 的旧运行在 84°C、SM
1305 MHz、`throttle_flags=0x20` 时被保护器拒绝，CSV 和 metadata 保留为
`performance_eligible=false`，不作为性能数据。完整逐点表见
[B=1](../results/ckks_auto_schedule_20261019/method1_n16_b1_comparison.md)
和 [B=8](../results/ckks_auto_schedule_20261019/method1_n16_b8_comparison.md)，
五组逐点表见 [comparison.md](../results/ckks_auto_schedule_20261019/comparison.md)，
选择证据在各 CSV 的同名 `.selection.csv` / `.samples.csv`。

B32 cooled 已由一个进程完成四场景，metadata 为 `completed`、returncode 0、
`ordinary`、`performance_eligible=true`。协议为
`cool-before-selection-and-measurement-rounds-v2`：选择和 holdout 每轮在 CUDA
event 外等待温度至 75°C 或以下且无热降频标志，仍保留 7×5 选择及 20×10
holdout。新二进制含 stream-forward 修复，身份与旧四组单列，不将跨协议差值
解释为版本加速。对原生 loop 的范围为 1.122065–1.220350×，对原生阶段合批为
1.089846–1.178421×。regret 均值为 0.5337%，范围 -0.6355%–2.8004%；旋转
选择 loop 后保留 2.8004% 的 holdout 残差，不追加实验或宣布最优。
该点 holdout loop/stage_batch/auto 均值分别为 5.165460/5.532764/5.310116 ms，
选择与候选排序一致；残差是选中路径单独复测的时差，其执行上下文与测量变化
尚未隔离，不能直接归为选错候选、selector 遗漏或纯随机噪声。

该复测支持“批量大小改变组织选择”的结论，但不能证明所有 N、方法或 GPU
均能由这两个候选达到全局最优。N=16384 的方法 I/II 自动选择结果也已保存
在同目录；它们与 N=65536 分开统计，避免把不同尺寸混成一个性能点。
完整五 cohort 统计与七个 N16384/B1 负点见
[自动选择证据汇总](../results/ckks_auto_schedule_20261019/README.md)。当前普通计时
完成 36/36 个配置，汇总核验 180 条配对结果、504 条选择样本和 3600 条
holdout 样本，全部整数密文精确一致。全矩阵选择为 13 loop、23 stage_batch；
相对原生 loop 有 29/36 点延迟更低，相对原生阶段合批为 36/36，均为点估计，
不作显著性结论。五组分别报告几何均值，失败与 profiler 记录继续排除。
本轮测试及有限归因可收敛，保留负点和选择残差；B32 没有 trace，不追加采集。

## 历史 stage-batch 负结果

本节依据为自动选择复测之前的 [合批报告](../results/ckks_stage_batch_20261019/README.md)、
[完整逐点表](../results/ckks_stage_batch_20261019/comparison.md) 和三份
`*guarded.csv`。三组均为 A100 80GB PCIe、完整 CUDA Graph、20 轮轮换、
每样本 10 次回放、整数密文精确一致；GPU、源码、二进制身份由同名
`.metadata.json` 保留。温控检查通过仍不等于锁频，约 1% 的差距须保留
动态时钟和样本离散度的不确定性。以下延迟属于该轮固定路径实验；后续自动
路径状态见上一节，当前 trace 的归因结果见后文，不能跨轮次相减。

| 优先级 | 请求与比较 | 已测事实 | 可以确认的边界 |
|---|---|---|---|
| P1 | 方法 I，N=16384，B=1，长链，自身循环/原生循环 | 原生 148.987 us，自身 163.052 us；0.91374x，增加 14.065 us | 两者都是 34 个 kernel；后续独立 trace 已定位到 NTT 请求，不能将其占比套用本轮完整延迟 |
| P1 | 方法 II，同请求 | 原生 170.342 us，自身 191.421 us；0.88988x，增加 21.079 us；35/35 kernel | 方法 II 保留更大残差，不能由方法 I 的旧 trace 定量外推 |
| P1 | N=16384，B=1，其余循环场景 | 方法 I keyswitch/平方链为 0.96973x/0.96152x，旋转 1.00717x；方法 II 为 0.95872x/0.96049x/0.98992x | 单密文并非所有场景都落后；长链优先于近似持平旋转 |
| P2 | 方法 I，N=65536，旋转 B=8，合批/自身循环 | 循环 1351.311 us，合批 1366.707 us；多 15.396 us，约 1.139%；88→40 kernel | kernel 更少仍略慢；合批对原生循环仍有 1.13788x 收益 |
| P2 | 同上，B=32 | 循环 5359.237 us，合批 5490.555 us；多 131.318 us，约 2.450%；352→112 kernel | 合批对原生循环仍为 1.15045x，不是对原库的性能失败 |
| 已有调度处理 | 所有 B=1 强制合批 | 12 个配置均慢于原生循环；N=16384 长链方法 I 合批 225.423 us、II 253.686 us | 这包括新增传输与合批请求映射变化，不能把全部差值叫 gather/scatter 时间 |

原始 CSV：
[方法 I/N14](../results/ckks_stage_batch_20261019/method1_n14_guarded.csv)、
[方法 I/N16](../results/ckks_stage_batch_20261019/method1_n16_guarded.csv)、
[方法 II/N14](../results/ckks_stage_batch_20261019/method2_n14_guarded.csv)。
表中差值直接来自这三份文件，不将三种对照视角算成独立配置。

## 归因状态与证据边界

| 层次 | 当前已证实 | 当前假设与缺失证据 |
|---|---|---|
| 请求级 NTT 核心/映射 | N14/B1 长链独立 trace 已将循环差距定位到 NTT，并恢复同序 11 个请求；详见后文 | 分段、算术和线程组织同时不同，尚未隔离因果贡献；不能直接断言 selector regret 或某个核心失败 |
| pack/scatter | [实现](../adapters/heongpu/rns_stage_batch.cu) 每阶段执行 `transfer<true>`、NTT、`transfer<false>`；N14/B1 与 N16/B8 trace 已分别量出传输时间 | 双方合批共享传输，但其时间不能由完整调用相减唯一恢复；B32 尚无 trace |
| 非 NTT 组织 | N16/B8 trace 已观察到阶段合批的真实 kernel 并发，分别报告非 NTT 总时长、并集 busy 与 GPU span | 节点相同不保证持续时间/关键路径相同；该 trace 不能解释其他 batch 或证明某种依赖改写的独立贡献 |

传输的算法流量可估为 `4 * sizeof(uint64_t) * N * P`，P 为整图所有合批阶段
处理的多项式总数：gather 与 scatter 各读写一次。旋转每密文 P=32，因此
N65536、B8/32 分别约 512 MiB/2 GiB。此值是源码推导的读写量，不是实测
DRAM 流量或传输延迟，不能据此证明带宽瓶颈。

已有 trace 的可复用范围：

- [stage mapping trace 审计](../results/e2e_rns_workload_20261018/stage_mapping_trace/README.md)
  及 [汇总](../results/e2e_rns_workload_20261018/trace_summary_stage_mapping.json)
  为 N16384、外层 B8、A100 40GB 的旧循环构建。四场景新增 kernel 时间的
  85.7%–93.3% 来自 NTT；长链 B2 逆/正向请求合计占其 NTT 增量约 44.8%。
- [radix4 trace 审计](../results/e2e_rns_workload_20261018/postchange_trace/README.md)
  同样只覆盖旧循环；对应四场景的 NTT 增量占比为 90.1%–95.7%。
- 两组均早于当前原位/放置调优及 stage-batch。它们支持“优先追踪小请求”
  的诊断顺序，不能给新 B1 残差或 N65536 合批负收益分配百分比。
- [历史瓶颈文档](fhe_e2e_bottleneck_20261019.md) 已声明上述限制。
  [核心对齐文档](rns_gpu_ntt_core_20261018.md) 中 Barrett PTX 和单独删除尾段
  barrier 均无稳定收益；不应把这两项作为已确定的下一步优化。

原 `results/ckks_stage_batch_20261019/` 未采集合批 trace。已完成的独立稳态
trace 位于 `results/ckks_auto_schedule_20261019/`，覆盖 N14/B1 长链与 N16/B8
旋转；它们不回写为原温控轮次的采集证据。B32 trace 尚未做，本轮不追加。

## 新 trace：方法 I/N16384/B1 长链

[原始 SQLite](../results/ckks_auto_schedule_20261019/trace_m1_n14_b1.sqlite)、
[可重复分析 JSON](../results/ckks_auto_schedule_20261019/trace_m1_n14_b1.analysis.json)、
[CSV](../results/ckks_auto_schedule_20261019/trace_m1_n14_b1.csv) 及
[metadata](../results/ckks_auto_schedule_20261019/trace_m1_n14_b1.metadata.json)
覆盖 `square_rescale_keyswitch`，四条路径各 3 轮、每轮 2 次回放。
metadata 为 `completed`、`measurement_kind=profiler-diagnostic`、
`performance_eligible=false`，这是有意排除 profiler 主性能声明的状态。
整数密文仍精确一致。以下微秒数都是 kernel 诊断，不替代该构建的普通计时。

解析器在真实 schema 上找到 12 个 NVTX 测量范围和 24 次 GraphLaunch，
全部关联，未分配 GraphLaunch 为 0。`NVTX_EVENTS` 同时有 `text/textId` 与
`globalTid`，kernel 通过 runtime correlation ID 关联。真实传输符号为
`cubutterfly_heongpu::<unnamed>::transfer<(bool)1/0>`，分别识别为 gather/scatter。
两条循环路径每次回放均为 22 个 NTT + 12 个非 NTT；两条阶段合批路径均为
22 NTT + 11 gather + 11 scatter + 12 非 NTT，节点数分别为 34 和 56，
与 CSV 完全一致。所有回放的 kernel 时间和等于并集 busy，本 trace 未观察到
kernel 时间区间重叠；这不证明其他负载也没有并发。

下表先对各轮的两次回放取均值，再对三轮取均值，单位 us。

| 路径 | NTT | gather | scatter | 非 NTT | kernel 并集 busy | 每回放 GPU span |
|---|---:|---:|---:|---:|---:|---:|
| native_loop | 119.145 | 0 | 0 | 37.210 | 156.354 | 157.921 |
| cu_loop | 141.467 | 0 | 0 | 37.526 | 178.993 | 180.614 |
| native_stage_batch | 131.066 | 25.349 | 22.885 | 37.546 | 216.846 | 219.441 |
| cu_stage_batch | 149.748 | 26.688 | 23.803 | 38.111 | 238.349 | 241.063 |

三项差分提供相互独立的定位：

- 两条循环相比，cu 多 22.638 us kernel 时间，其中 NTT 多 22.322 us，
  占 98.60%；非 NTT 多 0.316 us。三轮 NTT 均值分别为 cu
  141.615/141.536/141.250 us，原生 119.246/119.072/119.116 us。
  当前 B1 长链的剩余 kernel 劣势因此已定位到 NTT 路径；下节进一步定位
  实际请求，仍不能仅据此断言某个候选遗漏。
- cu 合批比自身循环多 59.356 us，其中 gather/scatter 合计 50.490 us，
  占 85.06%；NTT 多 8.281 us，占 13.95%；非 NTT 多 0.585 us。
  这直接支持单密文保留循环候选，且说明强制合批损失不全是数据搬运。
- 双方合批相比，cu 多 21.503 us，其中 NTT 多 18.682 us，占 86.88%；
  gather/scatter 多 2.256 us，非 NTT 多 0.565 us。共享传输源码并不使其
  实际执行时间严格相等；本表保持各类实测差分。

普通 CUDA event CSV 的原生循环/cu循环为 198.437/224.357 us，而逐回放
kernel span 均值为 157.921/180.614 us。二者边界不同，差额包含事件窗口内的
回放间隙等因素；不能将上述占比套到 CSV 完整延迟差值，或将 profiler
绝对时间与旧温控表相减。N65536/B8 旋转的独立 trace 见后文；方法 II 与
N65536/B32 尚无对应 trace，不以本节代替。

### 同序的 11 个 NTT 请求

分析 JSON 的 `loop_ntt_requests` 复用既有 `fused_request/read_plans` 解码，
逐 GraphLaunch 按实际 GPU 顺序将 22 个 NTT kernel 配成 11 对。6 次回放的
每个 ordinal 的符号、grid/block、寄存器/shared 签名均一致；原生/cu 每对的
正逆方向一致，各 pass 恢复的 cu 多项式数一致，总数为 47，与 CSV 一致。
原生的 `PolyOrdered/ModulusOrdered` 符号保留其顺序类别。以下均为每次图
回放的六次样本均值，单位 us；序号从 1 开始，B 为请求内多项式数。

| 序号 | B/方向/原生顺序 | 原生 NTT | cu NTT | cu 增量 |
|---|---|---:|---:|---:|
| 1 | 3/I/contiguous | 11.115 | 13.109 | 1.995 |
| 2 | 12/F/modulus | 15.984 | 15.920 | -0.064 |
| 3 | 2/I/polynomial | 11.189 | 12.560 | 1.371 |
| 4 | 6/F/contiguous | 10.485 | 13.195 | 2.710 |
| 5 | 2/I/polynomial | 10.090 | 12.208 | 2.118 |
| 6 | 4/F/contiguous | 10.053 | 13.099 | 3.045 |
| 7 | 4/I/contiguous | 11.599 | 13.349 | 1.750 |
| 8 | 6/F/modulus | 10.635 | 12.784 | 2.149 |
| 9 | 2/I/polynomial | 9.813 | 12.405 | 2.592 |
| 10 | 4/F/contiguous | 10.219 | 11.104 | 0.885 |
| 11 | 2/F/contiguous | 7.963 | 11.734 | 3.771 |

增量最大的三个 ordinal 是 11、6、4，合计 9.526 us，占 NTT 增量 42.67%。
全部均为正向，两个 pass 都有差距，第二 pass 分别贡献
2.171/1.776/1.563 us。它们不是“B2 逆向单一瓶颈”；三次 B2 逆向合计另占
约 27.24%。第 6 和第 10 请求具有相同可见几何，却有不同应用位置下的
kernel 时间，说明应用上下文影响也须保留，不能仅按 mapping 名字预测成本。

三个最重请求的原生符号均为 `gpuntt::ForwardCore<unsigned long>`；P1 使用
grid=(32,1,B)、block=(16,16,1)，P2 使用 grid=(1,32,B)、block=(256,1,1)，
两个 pass 均为 32 registers/thread、4096 B dynamic shared、0 static shared。
对应 GPU-NTT 的 5+9 分段。cu 符号为
`cuntt::rns_fused_detail::stage_kernel<L,R,I,radix_log,barrett,compact>`，下表
省略符号里的显式类型转换，完整符号保存在 JSON 的 `stages` 和 `kernels`。
cu 所列 block 均为 (256,1,1)，grid 后两维均为 1，static shared 均为 0。

| 序号 | cu P1/P2 模板参数 | cu gridX P1/P2 | cu registers/thread P1/P2 | cu dynamic shared P1/P2 | P1 原生→cu us | P2 原生→cu us |
|---|---|---|---|---|---|---|
| 11 | (6,8,0,2,0,0)/(8,2,0,2,0,0)，radix4/Shoup | 64/64 | 40/40 | 4096/4096 B | 3.451→5.051 | 4.512→6.683 |
| 6 | (6,4,0,1,0,0)/(8,1,0,1,0,0)，radix2/Shoup | 256/256 | 28/28 | 2048/2048 B | 4.389→5.659 | 5.664→7.440 |
| 4 | (6,8,0,2,0,0)/(8,2,0,2,0,0)，radix4/Shoup | 192/192 | 40/40 | 4096/4096 B | 4.608→5.755 | 5.877→7.440 |

准备日志的可见语义匹配如下。第 11 请求唯一匹配 `ntt_adapter_plan` 第 33 行，
第 6 唯一匹配第 21 行，均为原位，domains 为
`(q,psi)=(1125899903991809,125929543876)` 和
`(1099508121601,13296178)`。第 4 请求匹配第 15 和 27 行，两者前两个域
相同，第三个分别为 `(1099510054913,42618759)` 与
`(1125899904679937,184459094098)`；trace 符号/几何不能区分这两个域集合。
三个 B2 逆向请求也有两个不同的可见域集合，不能凭准备日志行序强行分配。
每项完整候选、域及原日志行号均保存在 JSON。即使只有一个可见语义匹配，
日志仍缺运行时 offset/ordered-index/pointer 身份，不能称为完整 plan ID 还原。

这组证据将下一步范围收敛为 B2/B4/B6 正向请求的阶段映射与核心协作。
当前 cu 是 6+8、Shoup 与一维 block，原生是 5+9、Barrett 与二维 ownership；
分段、算术和线程组织同时不同，因此尚不能把差值单独归给某一项。优先用
相同完整语义重放已有合法候选、核对两 pass 的几何，保持原位和域契约；
本轮没有新证据要求另造算术核心，也没有证明安装模型候选遗漏。

## 历史最小诊断集合与采集方案

最初方案列出三个配置，沿用相同数据、完整图和四路路径：方法 I/N16384/B1，
方法 I/N65536/B8，以及方法 I/N65536/B32。现已完成前两组，B32 trace 未执行，
本轮不追加。以下保留采集时的方案与环境检查，不作为待执行清单。runner 新增 `--scenes` 和
`--trace-measurement`，分别只采长链、旋转、旋转；每进程限定单一场景/N/B。
方法 II/N16384/B1 只在首轮不能解释更大残差或修改涉及其域布局时追加。
主 Agent 已安排的普通独立计时直接复用；profiler 内计时另标诊断。

2026-10-19 本机只读环境检查：`nsys --version` 返回 2023.4.4.54；
`nsys status --environment` 显示 perf_event_open 失败、kernel paranoid=4，
所以关闭 CPU 采样及上下文切换采集。`/proc/driver/nvidia/params` 中
`RmProfilingAdminOnly=0`，`/dev/nvidiactl`、`/dev/nvidia-uvm`、`/dev/nvidia0`
均对当时用户可读写。这是采集前检查；之后两组 CUDA Graph node trace 已完成，
其实际关联与解析结果分别记录于对应章节。
本机帮助确认支持 `--cuda-graph-trace=node`，必须采 node 才能分离传输/NTT。

以下命令为历史采集模板，输出使用新的诊断目录。执行时先按项目规则确认目标 UUID
空闲并保存构建身份；若主 Agent 已更换构建，替换 `--build-dir` 与
`--cuntt-library` 为独立复测采用的同一版本，并保留同名 metadata。

```bash
CUDA_VISIBLE_DEVICES=GPU-2971e500-d3f6-d7e6-e5f2-cb67a06fff02 \
nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  --capture-range=cudaProfilerApi --cuda-graph-trace=node \
  --output=results/ckks_auto_schedule_20261019/trace_m1_n14_b1 \
  /home/wt/yes/envs/cubutterfly/bin/python paper/experiments/run_heongpu_ckks_e2e.py \
  --build-dir build-fhe-inplace-20261019 --skip-build \
  --cuntt-library build-contract-20261017/libcuntt.so \
  --sizes 16384 --batches 1 --aux-primes 1 \
  --graph --stage-batch --thermal-guard --trace-measurement \
  --scenes square_rescale_keyswitch \
  --warmup 2 --repetitions 3 --graph-replays 2 \
  --output results/ckks_auto_schedule_20261019/trace_m1_n14_b1.csv

nsys export --type=sqlite \
  --output=results/ckks_auto_schedule_20261019/trace_m1_n14_b1.sqlite \
  results/ckks_auto_schedule_20261019/trace_m1_n14_b1.nsys-rep

/home/wt/yes/envs/cubutterfly/bin/python \
  paper/experiments/analyze_ckks_schedule_trace.py \
  results/ckks_auto_schedule_20261019/trace_m1_n14_b1.sqlite \
  --ntt-size 16384 \
  --plan-log results/ckks_auto_schedule_20261019/trace_m1_n14_b1.log \
  --output results/ckks_auto_schedule_20261019/trace_m1_n14_b1.analysis.json
```

采集前创建新的输出目录；不覆盖既有报告。N65536 两配置分别使用 `--batches 8`
或 `32`、`--scenes rotate_rows_shift1 --rotate`，选择原 N16 GPU UUID，更新
输出名为 `trace_m1_n16_b8` 或 `trace_m1_n16_b32`。三组均可添加
`--auto-schedule` 同时记录第五条自动路径。

采集区间由 `cudaProfilerStart/Stop` 限定为稳态测量。每轮四/五条路径各自用
`ckks_measure/{scene}/{variant}/{round}` 标记；variant 为 `native_loop`、
`cu_loop`、`native_stage_batch`、`cu_stage_batch`，以及可选 `cu_auto`。
[新 CPU 解析器](../paper/experiments/analyze_ckks_schedule_trace.py) 复用已有
解析器的 SQLite 只读和 GraphLaunch correlation 关联方式，移除旧 N14/B8
节点数硬编码，按 NVTX 同线程范围找到 API，再按 correlation ID 找 kernel。
GPU 时间戳无需落在 host NVTX 范围内；异步工作不能按 host 区间裁切。

输出逐 variant/round 的四类 kernel 数/时间、回放数、每次回放统计及 kernel
符号；并列报告 kernel 时间和、全部 kernel 时间区间并集 busy、首末 kernel
span。每次回放的均值由各回放独立统计后取平均；整轮 span 保留回放间空隙。
这些都为 profiler 诊断，不称为普通完整延迟。`transfer<true/false>` 分别归为
gather/scatter，缺 NVTX、重复身份、关联不唯一或缺 node 活动均显式拒绝。
8 个 CPU fixture 测试覆盖并发区间、异步超出 host 范围、其他线程排除，
以及两 pass 几何解码、准备语义匹配与正逆序列不一致拒绝。

若 N14 长链增量主要来自 NTT，只对实际贡献最大的请求固定完整语义与
alias/offset/root/order，复放当前 winner 和已有合法候选；检查候选是否进入、
是否测量、完整应用中是否仍获胜，再决定是否存在选择 regret。首轮无需
NCU；仅当某阶段稳定贡献主要差距时才采该阶段的访存、等待和资源指标。
若 N16 的 pack/scatter 主导，先考察兼容布局下消除拷贝或生产者直写消费者
布局；若非 NTT span 主导，则考察合法依赖组织。两者都需保持完整密文输出。

### N=65536、B=8 旋转诊断

新增 trace `trace_m1_n16_b8_v2` 覆盖四条固定路径和自动路径，metadata 明确为
`measurement_kind=profiler-diagnostic`、`performance_eligible=false`。
[分析 JSON](../results/ckks_auto_schedule_20261019/trace_m1_n16_b8_v2.analysis.json)
含 15 个范围、30 次 Graph replay，未分配 GraphLaunch 为 0。以下均为每次
完整图回放的六次样本均值，单位 us；分类列是 kernel 总时长，busy 是区间并集，
span 为该图首个 kernel 开始至末个 kernel 结束。整轮有两次回放，不能将整轮
`kernel_busy_us` 与每图 `kernel_us_per_graph` 混用。

| 路径 | NTT | gather+scatter | 非 NTT | kernel 总时长 | 并集 busy | 每图 span |
|---|---:|---:|---:|---:|---:|---:|
| native_loop | 1226.913 | 0 | 259.754 | 1486.667 | 1486.667 | 1490.617 |
| cu_loop | 1043.968 | 0 | 255.190 | 1299.158 | 1299.158 | 1303.121 |
| native_stage_batch | 1015.729 | 317.734 | 372.886 | 1706.349 | 1532.788 | 1540.713 |
| cu_stage_batch | 777.294 | 322.577 | 374.219 | 1474.090 | 1298.594 | 1306.534 |
| cu_auto（选 loop） | 1043.456 | 0 | 255.160 | 1298.615 | 1298.615 | 1302.631 |

循环路径每图 88 kernel，阶段合批为 40。循环未观察到 kernel 重叠；阶段合批
有真实并发。逐图扫描 kernel 时间区间，原生/cu 阶段合批至少两个 kernel 同时
执行的时间均值分别为 139.496/140.211 us，最多并发 4/5 个。kernel 总时长减
busy 分别为 173.561/175.496 us；并发超过两个时该差值累计额外执行区间，
不能直接叫作重叠的墙钟时长。span 还包含 kernel 间隙，不等于 busy。

在双方同为阶段合批时，cu 的 NTT kernel 总时长少 238.435 us，传输多
4.843 us、非 NTT 多 1.333 us，kernel 总时长净少 232.260 us；busy 比为
1.180345×。同为循环时，NTT 少 182.945 us，占 kernel 总时长降幅 97.57%，
busy 比为 1.144332×。这支持当前 trace 的 NTT 路径是双方差距的主要来源，
并未单独隔离局部核心与请求映射的贡献。

cu 自身合批比循环的 NTT 少 266.674 us，但传输新增 322.577 us、非 NTT
多 119.029 us，kernel 总时长反而多 174.932 us。并发使 busy 近似持平
（合批少 0.564 us），每图 span 则多 3.413 us。因此不能说“NTT 节省单独
覆盖了传输成本”，也不能将分类总时长直接相加推导完整应用加速。

诊断进程用 3 轮、每轮 2 次回放选择 `loop`，普通进程用 7 轮、每轮 5 次回放
选择 `stage_batch`，随后各自独立计时。采集条件、选择预算和观测均不同，
当前证据不能将 winner 差异单独归因于 profiler 开销或低重复数。
普通 B8 旋转的 auto 为 1345.971 us，两个固定候选为 loop 1344.384 us、
stage_batch 1343.309 us；0.1982% holdout regret 表明该点接近持平。
保留两组观测及各自边界，不用 profiler 替代普通性能结论。

## FFT/FWHT 只读优先级筛查

算子数据取自 [安装横评说明](install_comparison_results_20261017.md) 与
[统一 CSV](../results/install_compare_20261013/cpu_external_v2/unified_comparison.csv)，
只保留匹配 GPU 对照；浮点状态仍为 error-report-only，不改写为数值验收通过。
该表有自己的冻结构建身份，不能与 CKKS 当前构建混算。

| 优先级 | 弱点或缺口 | 已保存结果与下一步边界 |
|---|---|---|
| P2 | FFT complex64，N256/B1，正/逆，对 VkFFT | `e1-00/01-fft-fp32` 为 0.76471x/0.80882x，cu 均 3.482 us；检查短调用边界及已有核心支持域，先不展开新核心重构 |
| P3 | FFT 其他近似持平负点 | N4096/B32 逆向对 VkFFT 0.99259x；N65536/B1、B4 正向对 cuFFT 0.99412x/0.99197x；32³/B4 逆向对 VkFFT 0.98343x；先独立确认离散度，避免对亚百分比残差过度优化 |
| P3 | FWHT FP32 xD 对 Dao | `extension-fwht-rank2` 128²/B4 正向 0.99091x；rank4 8⁴/B4 正向 0.99083x；其余有匹配 Dao 结果的当前冻结请求未显示负点，缺失 GPU 对照仍保留 |
| P2 验证缺口 | QuaRot B1/prefill128/decode8，FP32 composed，Graph | [最新 Graph 记录](../results/e2e_current_20261018/quarot_b1_p128_d8_graph_fp32_v2.json) 为 `blocked_runtime_execution`，原 token Graph 数值验证失败；没有可使用的 Graph 延迟结论 |
| P3 | 同 QuaRot 普通调用 | [eager 记录](../results/e2e_current_20261018/quarot_b1_p128_d8_eager_fp32.json) 完整序列 689.485/677.323 ms，0.98236x；两个样本、treatment-first、数值未设阈值。四个 FWHT 计划均为 static_model；先核对调优接入与 trace 占比，不据此断言 FWHT 核心是全部差距 |
| 降级为历史 | Poisson 2048²/B16 原 Graph 负结果 | [后续独立自适应结果](e2e_b16_trace_20261017.md) 对 Torch/CuPy 为 1.159x/1.186x；当前 Torch 控制 1.177x，旧 summary 的 0.84x 不能当现有未修弱点。2048²/B4 eager 仍约 0.536x，Graph 1.159x，属于普通调用提交边界诊断 |

最新固定滤波器 Graph 结果也不是当前弱点：
[1024²/B16](../results/e2e_current_20261018/fft_n1024_b16/image1024_filter31_batch16.current_graph_fast.json)
为 cu 1406.293 us、CuPy 1500.160 us、同 padding Torch 1532.587 us；
[2048²/B16](../results/e2e_current_20261018/fft_n2048_b16/image2048_filter31_batch16.current_graph_fast.json)
为 5730.816/5903.872/6007.040 us。均保留完整输出和原有误差分布，不能扩大
为全部 FFT 应用领先。更早相位相关和 FNO 普通调用负点保持其原构建记录；
本轮不把它们加入新的复测矩阵。

可复用优化候选按证据触发：调用级候选选择、真实 alias/offset 契约下的请求
映射、合法边界直写以减少搬运、安装/请求调优结果的公共路径复用。每项先用
上面的有限诊断确定贡献；当前证据不足以指定一个新的算术核心为确定赢家。

## 2026-10-19 端到端对齐收口

本轮新的 A100 80 GB 结果统一放在
[`results/e2e_alignment_20261019/`](../results/e2e_alignment_20261019/)。统一表固定为
“场景、类型、对照、batch、完整调用延迟、加速比、验证状态”，CUDA event 是主口径，
墙钟只作诊断。FFT 滤波的完整调用包含输入刷新、padding、正逆变换、裁剪和独立输出
拷贝；QuaRot 和 TFHE 保留 Graph 的改输入、恢复输入、输出生命周期及精确结果检查。

完整 CKKS bootstrap-v2 使用固定的 N=65536、Q/P 模数链。原生-rns 试跑暴露了同一
输入对象在完整 bootstrap 中被原地消费，基准循环因此曾让后端比较不同输入；runner 已
在每个 eager 后端测量前恢复完整输入。修正后准备阶段仍会积累大量不同 RNS 计划并耗尽
A100 80 GB 的计划缓存，未产生可用的性能数字。该资源边界与早期 exact-check 诊断均
保留为失败记录，不转换成加速比，也不把短链 CKKS 矩阵混入该结果。

为响应显存边界，新增了两个明确分离的缩小 profile。复用 v2 Q/P 链但改 N=16384
的 `bootstrap-v2-small` 虽通过静态根条件检查，实际 native 解码误差达到
8.7×10¹²¹ 量级，且 native/cuButterfly 密文不一致，因此只作为 reduced-N 功能失败记录。
有效的缩小应用采用上游 `1_ckks_regular_bootstrapping.cpp` 的 N=4096、31 个 Q 位宽、
3 个 P 位宽和 `(3,3,11,true)` legacy 配置。该配置 batch=1 的完整 eager 调用逐元素
一致，native 18.823 ms、cuButterfly 18.599 ms，加速比 1.012×。这只是 v1 reduced-N
证据，不能外推到 N=65536 v2；对应 Graph 重跑因选定 GPU 出现外部 Python 进程被所有权
监控器终止，没有性能数字。
