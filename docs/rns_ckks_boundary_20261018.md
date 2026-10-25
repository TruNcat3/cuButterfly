# CKKS RNS NTT 的顺序支持与访存效率

本轮修复针对已支持的位反转边界的执行效率。cuButterfly 原有路径已经
直接消费和产生位反转序，既不要求用户预排序，也没有额外的全局排序
kernel。此前把直接顺序支持与高效访问混在一起的解释不准确。

## 历史横评与 CKKS 的差别

历史 RC2 的 1.472x 是 214 个请求的几何均值，覆盖 word32/word64、正逆、
不同环和 batch。CKKS 当前请求是 word64、多个 40/50-bit RNS 模数、
negacyclic，正向 natural→bit-reversed，逆向 bit-reversed→natural。
历史自然序双边界对照需要计入 GPU-NTT 的额外排列，当前 CKKS 则直接
消费其原生顺序。这解释了为何两个完整调用结果不能直接互相替代，
但尚不能量化排列在历史优势中占多少。

另有两项边界差别。历史 GPU-NTT revision 为 `d03c5eae`，HEonGPU 源码
固定的 revision 为 `8a4daf11`。历史 RNS 对照逐模数调用，当前原库一次
请求包含多个模数。外层密文 batch 也不等于单次 NTT 的多项式 batch。
新运行元数据补记实际链接的 GPU-NTT archive 哈希和 checkout revision，
仍如实保留源码与二进制对应关系未经自动证明的状态。

历史来源详见 [NTT 审计](ntt_comparison_history_20261017.md)。修复前的
[CKKS trace](../results/e2e_rns_20261018/trace_analysis.md) 与本轮修复后的
[性能及机制报告](../results/e2e_rns_coalesced_20261018/README.md) 分别保留。

## 已定位的实现差别

原生 GPU-NTT merge 使用融合根的 CT/GS，直接满足 negacyclic 的自然序/
位反转序契约。当前 RNS Hybrid2D 保留普通局部 NTT、边界 twist 和分段
cross 系数。因此即使都只发射两个 kernel，执行的模乘、系数读取和地址
模式也不相同。Shoup 与 Barrett 的差别本身不能证明哪方更快。

本轮还把 pinned GPU-NTT Merge 作为 `RnsCoreFamily::GpuNttMerge` 模板接入。
它要求 N 对齐且互不重叠的物理行、显式的每多项式 domain/offset 序列和
radix2 Barrett，因此可以消费 HEonGPU 按 modulus 分组或按 polynomial
选择造成的逻辑行置换；输入输出别名且
行位置相同时才采用原地路径。它已通过 12 个 HEonGPU adapter 精确用例，
但当前 A100 受控测量仍慢于 Shoup-CTGS，所以搜索器只把它作为候选，不把
GPU-NTT 或 HEonGPU 的外围 RNS 优势误写成核心加速。

本轮先保持算术核心不变，处理两项可单独验证的映射限制。

| 参数 | 原候选 | 新增候选 | 保持不变 |
| --- | --- | --- | --- |
| 独立行分组 | 每块 1/2/4 行 | 8/16/32 行，受资源约束 | 局部蝶形算术 |
| 位反转边界遍历 | 按逻辑系数索引分配线程 | 按连续物理地址分配线程 | 输入输出顺序、根及归一化 |

令 N=N1×N2，rev 为相应长度的位反转。第一遍的逻辑输入索引是
`n2*N1+n1`。对于位反转输入，线程遍历连续位置 p，令 `n2=rev(p)`，
其物理地址即 `rev(n1)*N2+p`，共享内存目的位置也连续。

第二遍逻辑输出是 `k1*N2+k2`。对于位反转输出，令 `k1=rev(p)`，
物理地址成为 `rev(k2)*N1+p`。全局写入连续，但共享读取仍可能冲突，
所以保留原遍历作为候选，由同请求实测选择，不按应用或 GPU 型号写赢家。

## 融合根核心候选

`RnsCoreFamily::FusedCTGS` 将成熟 GPU-NTT 的融合根 CT/GS 策略参数化为
stage interval、独立行分组和线程配置，使用本项目已有 Shoup 模乘。
每级执行明确的 CTA 同步，避免依赖隐式 warp 同步。根表、模数、方向、
物理偏移和输入输出顺序由同一个 RNS 请求传入，未为 CKKS 固定赢家。

该核心将 negacyclic 根直接放入蝶形，省去独立的逐系数 twist 和分段
cross 模乘。它与 Hybrid2D 共同参与调优。后续补入寄存器融合的 radix4，
相邻两级在四个寄存器中执行后再交换，奇数级余数使用 radix2；原 radix2
保留为候选。尚未实现的 radix8 或根放置选项会被拒绝。核心正逆方向分别编译，展开局部级循环，
将已知为二次幂的除法和取模写成移位与掩码。

radix4 的新增 20 例将精确验证总数扩至 124。负载分布及本轮结果见
[后续优化报告](../results/e2e_rns_workload_20261018/optimization.md)；
下述 104 例为前一轮已保留的验证范围。

此后短阶段及逐遍映射补齐将精确用例扩至 148：FusedCTGS 的局部级数
扩展到 2..10，行分组到 128，第二遍 rows/threads 可独立选择，均受资源
和 tile 数约束。候选集合保留耦合参数并增加等 tile 字数的逐遍组合。
原生 3+9、5+9 等组织由公共参数表达，不按应用或 GPU 写赢家。
新增字段改变 C++ ABI，调用端须重编。最新实测与负结果见
[阶段映射报告](../results/e2e_rns_workload_20261018/stage_mapping_optimization.md)。

## 验证与归因口径

`tests/test_rns_ntt.cu` 保留原有 36 例，增加 48 例独立 CPU 精确对照，
覆盖两种环、双方向、四种边界顺序、较大行分组、不均衡分段、原地执行、
空洞保护和 Graph 输入变更。融合根核心再增加 20 例，共 104 例，另测
不支持的环及冲突映射参数。共享内存超限与无效参数显式拒绝。

`apps/cuntt_rns_mapping_bench.cu` 在同一请求中比较原候选子集、仅切换
边界遍历、扩大行分组的子集、融合根核心子集及完整调优结果。只有第二项与第一项构成
单变量对照；重新搜索的子集结果另行标注。普通计时使用预热后的 Graph，
独立 CPU 校验、编译、调优和图创建均在计时外。

NCU 使用显式 mapping 重放，避免两次 profiling 重新调优出不同核心或
分段后错误归因。应用性能继续比较完整 CKKS 调用；诊断 kernel 时间
不冒充端到端速度。完整 bootstrap、跨密文阶段批处理和全局 RNS planner
接入仍属于独立的未完成工作。
