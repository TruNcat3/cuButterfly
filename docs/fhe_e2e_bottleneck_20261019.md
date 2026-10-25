# FHE/CKKS 端到端劣势归因与下一步边界

后续已将同阶段跨密文合批接入完整 CKKS Graph，见
[接入、四路对照与温控复测](../results/ckks_stage_batch_20261019/README.md)。
本文保留逐密文循环路径的历史归因，不能将这些旧结果当作合批后的结论。

本记录只回答一个问题：为什么当前 CKKS/FHE 完整调用仍落后于
HEonGPU 的原生 GPU-NTT 路径。它不把一次受控修复结果写成全面性能结论，
也不把 FHE 中非蝶形的工作错误归入 NTT。

后续的原位安全修复、放置感知调优、20 个完整调用及真实合批控制见
[本轮完整结果](../results/rns_placement_batch_20261019/README.md)。
下述分阶段映射 trace 保留其原始构建身份。

## 结论

当前劣势有三个相互叠加的来源，其中第三项在 2026-10-20 的完整阶段图
profile 中被定量确认。

1. **应用内的 NTT 请求是小批次**。CKKS runner 的外层 `batch` 是独立密文
   的公共 API 循环。它不会扩大单个 NTT 的 grid。每次上层操作仍按自己的
   多项式集合调用一次 adapter，`RnsPlan` 在每次调用内做两次 pass。
2. **小批次下的核心组织仍弱于原生 GPU-NTT**。HEonGPU 使用原位、按
   RNS 行交错的原生 Merge 组织，当前 cuButterfly provider 需要同时处理
   显式 domain/offset，通用 fused kernel 的线程协作和边界路径更重。现有
   分阶段映射版本的 trace 把新增时间的 85.7--93.3% 定位到 NTT kernel，NCU 没有显示一个可以
   单独解释全部差距的 occupancy 或 shared-bank 指标。

   证据支持继续检查**低批次映射、选择时的调用契约及核心实现**。
   已测 CKKS 请求保持整数精确，非 NTT 部分没有出现同等规模的增量。
   这些结果不足以证明理论最优，也未把全部残差归因到某一条指令。

3. **阶段边界复制没有进入核心服务模型**。当前 `RnsStageBatch` 的每个
   阶段仍执行 gather、transform、scatter。B=8 的 516 阶段图中，核心探针
   `confirmed_kernel_ms` 求和为 44.4016 ms，而实际 cuButterfly 阶段图为
   114.8129 ms；B=4 的对应比值为 2.5645x。两次边界复制合计约 16.30 GB
   的 device 数据移动。因而 B=8 的 cu/native 阶段比只有 0.9465x，不能
   直接解释为核心线程映射失败。完整 profile 和同一 packed arena 的控制
   见
   [`results/e2e_alignment_20261020/ckks/kernel_profile_20261020/README.md`](../results/e2e_alignment_20261020/ckks/kernel_profile_20261020/README.md)。

## 证据链

分阶段映射版本的 CKKS trace（N=16384，外层 batch=8）中，替换 NTT 相对原库的新增
kernel 时间占比如下。数据来自
`results/e2e_rns_workload_20261018/stage_mapping_optimization.md` 的 A100 40GB trace。
它早于本文后面的原位与调优修复，不能直接充当修复后 trace。

| 场景 | 原库 NTT us | 替换 NTT us | NTT 占新增 kernel 时间 |
| --- | ---: | ---: | ---: |
| Keyswitch | 456.340 | 492.343 | 90.8% |
| 平方/relinearize/rescale | 529.462 | 580.831 | 92.1% |
| 上述计算后 keyswitch | 932.548 | 1158.838 | 93.3% |
| 槽旋转 | 424.643 | 436.953 | 85.7% |

同一 trace 的 kernel 数量、非 NTT kernel 名称、grid、block、shared 和
寄存器签名均匹配。弱链中 B=2 逆向贡献 68.956us、正向贡献 32.528us，
两者合计约 44.8% 的剩余 NTT 增量，
   而 B=12 的正向请求贡献很小。外层 batch 从 1 增加到 8 或 64 时，调用数
   增加，单次请求的多项式数不增加，所以没有得到理想的单 NTT 扩展收益。

较早的孤立 provider 控制使用 A100 40GB、N=16384、B=2 的
同语义控制中，pinned GPU-NTT 的 forward/inverse 分别为 0.018880/0.021146
ms，cuButterfly 的 GPU-NTT-compatible provider 分别为 0.021965/0.023744
ms。其中仅 cuButterfly 记录了逐元素校验；原生控制是 timing-only，
且两者独立进程、动态时钟，不能据此精确量化核心差距。本轮新增的
同进程四路 stage-batch 控制对两个后端分别进行 CPU 精确校验。

## 已修复的工程缺口

adapter 将不同模数的行按 modulus 分组后，`input_offset == output_offset`
但物理行顺序可能不等于逻辑多项式顺序。此前 `fused_alias_eligible()` 只
接受严格的 `index*N` 布局，合法的原位重排行因此被迫使用独立 scratch
路径。现在 fused stage kernel 在 `in_place` 时保留每个多项式的显式
offset，两遍都在对应物理行上读写。资格检查保留构造函数已有的输出
区间不重叠约束。本轮进一步要求正向 natural→bit-reversed、逆向
bit-reversed→natural 原生边界；其他顺序仍通过两遍 scratch 路径支持，
避免全局位反转跨 CTA 覆盖尚未读取的数据。

修改位置：

- `src/rns_fused_ctgs.cuh`
- `src/rns_ntt.cu`

上一轮 155 个 RNS 精确用例通过，但追加的两对稀疏排列测试因奇偶索引和
反转开关抵消，实际均走独立缓冲区。本轮已改为显式放置组合，新增的
覆盖及验证结果见本轮记录，不能把旧通过数当作原位组合的完整证据。
上一轮在 A100 80GB GPU1 上运行了 N=16384、batch=8、CUDA Graph、
五次交替测量的受控 CKKS 调用。结果保存在
`results/e2e_rns_workload_20261018/ckks_inplace_fix_gpu1.csv`，密文逐元素
一致，kernel node 数一致：

| 场景 | 原库 ms | cuButterfly ms | 加速比 |
| --- | ---: | ---: | ---: |
| Keyswitch | 0.555827 | 0.580608 | 0.957x |
| 平方/relinearize/rescale | 0.709427 | 0.737894 | 0.961x |
| 上述计算后 keyswitch | 1.190502 | 1.357005 | 0.877x |
| 槽旋转 | 0.520397 | 0.516096 | 1.008x |

这是修复后的独立受控样本，不与不同 GPU、不同搜索状态的历史绝对时间
直接相减，也不作为全面 CKKS 结论。它说明原位重排行缺口确实已经进入
真实应用路径，但剩余弱点仍主要在低批次 NTT 核心和请求组织。

## 不能直接采用的修复

wrapper 不能把多个 `transform()` 调用静默延迟到后面再合并。HEonGPU 的
每次公共调用可能马上消费输出，keyswitch、mod-up/down 和后续乘法之间
存在依赖。正确的跨密文合并必须由上层显式声明同一阶段、同一 N、方向、
ring、顺序、root/domain、offset 和 stream，并把独立 ciphertext 行放入
连续 device arena，或提供受控的 pointer-array/gather 路径。当前
`RnsPlanConfig.polynomials` 已能表达一个 packed arena，但独立 Cipher 的
地址不能用简单 offset 假定连续。

## 后续实现顺序

1. 修正原位安全边界及 placement-aware 调优，再使用已有
   `RnsPlanConfig.polynomials` 表达显式同阶段 packed batch。无需新增
   重复的库接口；应用必须声明独立性和 arena 生命周期，不能延迟已有调用。
2. 在同一 CKKS 阶段中比较 packed batch 与独立公共 API，分别报告 pack、
   NTT、unpack 和完整调用时间，验证外层 batch 是否真正放大 NTT grid。
3. 将 GPU-NTT 的 2D/block-z 行交错和 Barrett/CTGS 协作作为独立 core
   provider 候选接入搜索，先做相同 root/order/alias 契约下的单核心控制，
   再放入 CKKS Graph。不能用调参结果代替核心实现对齐。
4. 在 stage service/cost model 中显式加入 boundary bytes 和 gather/scatter
   服务曲线，再搜索能保持兼容布局的 resident stage 链或边界融合路径。
   `RnsStageBatch::boundary_copy_bytes()` 已提供每次执行的精确搬运量；只有
   在该成本被单独建模后，才把 provider core 的残差作为完整图结论。
5. 重新跑完整 CKKS 矩阵；TFHE bootstrap 仍需单独处理，因为 HEonGPU 的
   `SmallNTT` 是 bootstrap 内部 helper，不能用 standalone RNS NTT 直接
   替换其融合调用。

## 复现

```bash
CUDA_VISIBLE_DEVICES=GPU-2971e500-d3f6-d7e6-e5f2-cb67a06fff02 \
python paper/experiments/run_heongpu_ckks_e2e.py \
  --cuntt-library build-contract-20261017/libcuntt.so \
  --build-dir build-fhe-inplace-20261019 \
  --sizes 16384 --batches 8 --warmup 1 --repetitions 5 \
  --rotate --graph \
  --output results/e2e_rns_workload_20261018/ckks_inplace_fix_gpu1.csv
```
