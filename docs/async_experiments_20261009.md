# 异步实验队列

实验工具版本为 `0.9.3.4-async-experiments.1`。算子库仍为 v0.9.3.4，
kernel、合法设计空间和两张 A100 80GB 的独立校准保持不变。
本次修改解决实验调度中的 CPU 串行等待，不是新的算子性能优化。

## 工作划分

| 工作 | 执行位置 | 交付物与约束 |
|---|---|---|
| 确定性输入、最终存储量化、精确参考结果、逻辑地址 | CPU 后台进程池 | 每个语义用例的校验缓存；不导入 Torch，不初始化 CUDA |
| 模板编译与已有模块验证 | 独立 CPU 编译队列 | 按源码、编译器、SM、mapping 标识的 ready/failed 记录 |
| 硬件相关计划构建、候选实测、完整算子和基线计时、输出回传 | 每张 GPU 一个消费者 | 同一 GPU 串行提交；可在其他用例等待 CPU 验证时运行就绪用例 |
| 输出误差、整数逐元素检查、padding/输入保持检查、压缩误差文件 | CPU 验证进程池 | 在全部验证结束后才提交终态结果 |
| 历史证据核验、失败隔离、缩放输入准备 | CPU 控制器 | 保留原失败记录，单独生成带输入 profile 和原 mapping 的控制实验 |

固定计时协议仍为 10 次预热、20 次图内调用、7 次回放；原地执行仍为
7 次独立调用，重置在事件计时之外。同一后端的七次样本连续完成，期间
不插入其他用例的 kernel，避免异步调度改变缓存复用。每次样本的完整
输出立即保存，之后统一交给 CPU 检查。首次正确性检查和 mapping 回放
检查仍在计时前完成。编译、计划构建、参考计算和误差统计不计入算子时间。

浮点结果继续标记 `accuracy-review-required`，不新增或放宽阈值。
整数保持精确比较。完整顺序/布局转换仍纳入计时，原生与适配结果分别记录。

## 编译与搜索的边界

不能通过切换到 `precompiled` 策略来制造就绪状态，这会改变 research
模式下的搜索范围。GPU 消费者使用 `CUBUTTERFLY_JIT_PYTHON` 接入编译代理。
代理只查询已验证的模块，或把缺少的 mapping 交给后台编译器，不现场编译。

原 C++ 搜索器会捕获候选构建异常并继续筛选。因此，只要代理产生一次
pending、running 或 failed 事件，外层就丢弃这次完整选型。所有请求就绪后，
使用原 descriptor 和全新的 Plan 重做选型。失败候选不会被静默删去。
编译失败、身份不符和预算用尽是可调查的停止状态，不是性能结论。

已知 mapping 可在 GPU 空闲前准备。尚未发现的自适应候选仍需硬件相关
计划构建触发；这部分无法在当前接口下声称完全脱离 GPU。发现缺项后，
消费者切换其他就绪用例，CPU 编译器持续工作。没有就绪任务时仍可能空闲，
状态文件会明确区分等待输入、等待编译和等待验证。

## 入口

复用 `/home/wt/yes/envs/cubutterfly/bin/python`。以下目录应使用绝对路径。

1. `semantic_preparation_queue.py --manifest MANIFEST --directory PREPARED --jobs 6`
   提前准备整个范围，已有完整且身份一致的缓存可恢复复用。
2. `semantic_compile_broker.py --init --broker BROKER --root REPO --mathdx MATHDX --nvcc NVCC --sm 80 --cache CACHE`
   固定编译身份。随后运行
   `semantic_compile_broker.py --watch --broker BROKER --jobs 6 --seed-cache-since LIBRARY --timeout 0`。
3. 在停止旧消费者并保存检查点后，运行
   `prepare_async_semantic_handoff.py --source-root OLD_ROOT --output HANDOFF`。
   它核验历史行、误差文件、协议、库与每卡校准，再生成剩余任务清单。
4. 每张卡各运行一个
   `run_async_semantic_queue.py --handoff HANDOFF/handoff.json --output GPU_OUTPUT --preparation PREPARED --compile-broker BROKER --slot 1`，
   第二张卡用 `--slot 2`。使用现有 `supervise_semantic_queue.py` 托管可记录退出原因。

底层 `run_async_semantic_campaign.py` 支持独立 manifest、GPU UUID、每卡校准、
`--cpu-workers` 和 `--in-flight`，可用于其他平台的同类部署。上层交接入口
对应本次两张 A100 的既有任务分片；它不替代通用算子或硬件接口。

GPU1 先完成剩余 conformance，再运行 timing shard 1；缩放控制的 CPU
准备与 timing 重叠，GPU 控制实验随后执行。GPU2 运行 timing shard 2。
已核验的历史结果只作为独立历史证据保留，不改写成新测量。
确认的正确性失败隔离后可推进其他用例；运行、编译或基线错误停止调查。
不会自动停止其他用户任务，也不恢复过期的 GPU 等待期限。

## 状态与恢复

本次目录为 `results/async_experiments_20261009_v0934_1/`。

- `prepared/records/`：CPU 输入与参考结果的逐用例状态。
- `compile/producer_status.json`：编译器存活、排队、就绪和失败数量。
- `gpu1/queue_status.json`、`gpu2/queue_status.json`：阶段及失败隔离记录。
- 每个 `measurement/results.json`：已完成测量和等待原因；
  `compile_events/` 保留完整选型被退回准备的原因。
- `handoff/handoff.json`：原任务进度、历史证据和剩余范围。

缓存是本机生成的可信 pickle 文件，只能由本工具写入并在校验后加载；
不能当作外部数据导入格式。CPU 并发与 GPU 在途用例有界，不按全矩阵
同时分配显存。原始结果和误差文件不删除。

中断后先检查具体停止原因。CPU 缓存和编译队列可恢复；GPU consumer 的
`--resume` 只接受相同身份且没有待调查错误的检查点。已记录的失败不能
靠重启改写成通过，部分或身份不明的队列状态也不会自动视为完成。

本次检查包含 CPU 队列/身份/损坏/中断测试、六类算子的真实 GPU 执行、
冷缓存缺项退回与重新选型、独立缩放输入原 mapping 回放和完整计时联调。
这些验证证明调度路径可执行，不等于完整性能矩阵已完成，也不解除既有
校准覆盖边界或浮点精度待审状态。

本次最终验证为 119 项 CPU 测试通过。A100 40GB 的 12 用例执行检查、
A100 80GB 的冷缓存和最终就绪缓存各 12 用例计时检查均完成，覆盖六类
operator；每组为 3 项整数精确通过、9 项浮点待审，没有运行或数值失败。
最终计时检查用时 85.99 秒，所有用例只需一次准备尝试。冷缓存检查中有
6 个用例因缺编译被完整退回并重新选型。该耗时仅描述这个小型联调用例集，
不外推完整矩阵吞吐。另一个原始 FP16 失败用例保持原 mapping 的独立
缩放控制完成，结果为浮点待审。外部库未覆盖的组合仍保留缺口标记。
