# Operator Coverage Contract (2026-10-06)

本文是当前 checkout 的有限覆盖盘点，不是“所有合法组合均已验证”、全硬件迁移或全局最优声明。它把三类事实分开：

* **当前同源测试入口/证据**：当前 source 中存在可复现入口；只有用同一 source/build 重新得到正确性结果，才升级为 current pass。
* **API 合法但当前矩阵缺行**：头文件和 `src/c_api.cpp` 允许，然而当前 workload/test 没有该精确组合。
* **当前实现明确拒绝**：应保留为显式 `NOT_SUPPORTED`/negative test；不能把它误报为已测成功。

任何旧 GPU CSV、旧 journal、旧 worktree 或旧映射都只是其命名 build/hardware/protocol 的历史证据。只有重新构建后的同源二进制、当前目标 GPU、当前输入语义和当前 protocol 才能计入验收。

## 1. Contract Surface

公开 C API 在 [`cubutterfly.h`](../include/cubutterfly/cubutterfly.h#L37) 声明六个 operator：FFT、NTT、FWHT、subset zeta、superset zeta、structured 2x2；storage enum 还声明 FP16/BF16/FP32/FP64、复数 FP16/BF16/FP32/FP64 及 UINT32/UINT64（同文件 [L46-L65](../include/cubutterfly/cubutterfly.h#L46)）。方向、in/out-of-place、zero-extended embedding、C2C/R2C/C2R、NTT natural/APPT/bit-reversed layout 和 rank 1--3 descriptor 的入口在 [L67-L105](../include/cubutterfly/cubutterfly.h#L67)。

合法性不是由 workload 名字推断，而是由 [`validate_storage`](../src/c_api.cpp#L295) 和 `cubutterflyCreatePlan` 执行：

| 维度 | 当前可声明的表面 | 当前实现的关键边界 |
| --- | --- | --- |
| rank | public descriptor rank 1--3（[`cubutterflySetShape`](../include/cubutterfly/cubutterfly.h#L177)） | `EXECUTION_XD` 要求 rank >= 2；rank > 3 不是 public contract。 |
| C2C FFT | complex storage；FP16/BF16/FP32/FP64 enum 存在 | standard arbitrary length 仅 complex FP32/FP64；power-of-two common path 的低精度入口由低层 butterfly 支持，但没有 public C-ABI 低精度执行证据。 |
| R2C/C2R | real storage，R2C forward、C2R inverse | 当前可执行路径是 rank1/2/3 的 standard、OOP、packed、FP32/FP64；同源 `check_xd.py` 已验证 rank1 odd/even 以及 rank2/3。compute 必须 native 或匹配 storage；explicit stride、in-place、embedding 被拒绝（[`src/c_api.cpp`](../src/c_api.cpp#L2433)）。最后轴输出 `N/2+1`。 |
| NTT | UINT32/UINT64，modulus、word bits、input/output order 可设 | standard arbitrary 仅 uint64/word64；bit-reversed output 仅 rank1 power-of-two + natural input；APPT static 仅 rank1 `N=2^20`、standard、OOP（[`src/c_api.cpp`](../src/c_api.cpp#L2794)）。直接 rank1 NTT 要求 packed contiguous batches（[`src/c_api.cpp`](../src/c_api.cpp#L3151)）。 |
| FWHT/zeta/structured | FWHT/structured 用 real floating；zeta 只 UINT32 | zero-extended embedding 要 OOP；rank1 common path 要求 input/output stride 完全相同（[`src/c_api.cpp`](../src/c_api.cpp#L3240)）。 |
| decimation selector | 无 `DIT`/`DIF`/`GS` public enum、descriptor field、workload field 或 test selector | 内部 radix/codelet 名称不能作为 DIT/DIF/GS contract；当前没有这三个维度的独立覆盖。 |

公开 C++ `Transform` 同样保存 extents、batch、strides、direction、placement、length mode、modulus、NTT order、stage matrices、execution mode 和 `fft_type`（[`plan.hpp`](../include/cubutterfly/plan.hpp#L20)），但它不扩大上表的 C-ABI 合法性。

注意：头文件 [L161-L162](../include/cubutterfly/cubutterfly.h#L161) 仍把 exported real-FFT adapter 描述为“public plan contract remains C2C”，而同一头文件的 `SetFftTransformType`、当前 `src/c_api.cpp` compact-real 分支和 `tests/check_xd.py` 已经形成实际 R2C/C2R C-ABI 路径。本契约按当前实现和 consumer harness 记录它，同时把 `test_c_api.c` 没有 R2C/C2R case 作为独立缺口。

## 2. Existing Commands

以下是已有入口的可执行清单。本文审计没有执行任何 GPU 命令；带 **GPU** 的命令只能在目标卡、同源新 build 上作为验收执行。

| 命令 | 覆盖/用途 | 运行条件 |
| --- | --- | --- |
| `cmake --build build-e13-shared -j2` | 构建当前 shared build 的 public tests/benchmarks | CPU/build；本轮未执行 |
| `ctest --test-dir build-e13-shared --output-on-failure` | CTest 总入口；包含 GPU-labelled tests | **GPU**；不要用历史 build 结果代替 |
| `build-e13-shared/cubutterfly_c_api_tests` | `tests/test_c_api.c` 的 C-ABI small/arbitrary/rank2/rank3/embedding/NTT-layout cases | **GPU** |
| `build-e13-shared/cubutterfly_tests` | `tests/test_butterfly.cpp`：FP32/FP64、FP16/BF16 low-level、zeta、structured、layout/placement/embedding | **GPU** |
| `build-e13-shared/cuntt_tests` | `tests/test_ntt.cpp`：word32/word64、forward/inverse、modulus、backend、APPT/order | **GPU** |
| `build-e13-shared/cubutterfly_public_planner_tests` | public planner mapping/serialization consistency，不执行 kernel | 规划器检查；其 binary 当前由 CMake 构建 |
| `build-e13-shared/cubutterfly_xd_fft_layout_tests` | internal xD C2C/R2C/C2R layout/core tests；可加 `--fused`, `--wide`, `--pair`, `--register-axis`, `--register-axis-l8`, `--register-axis-l16`, `--cooperative-pair`, `--cooperative-pair-layout1`, `--batch-tile`, `--axis-block-caps`, `--axis-exchange`, `--axis-resources`, `--local-core` | **GPU**；CMake/CTest 中标为 `gpu;xd` |
| `build-e13-shared/cubutterfly_rank3_permute_tests` | internal rank3 permutation、packed/padded stride、element bytes | **GPU**；不是 operator transform correctness |
| `python tests/check_xd.py --library build-e13-shared/libcuntt.so` | ctypes public C ABI：xD C2C、R2C/C2R、zeta、NTT、structured、stride/IP/embedding、mapping replay；`--case real_rank` 可筛 real | **GPU**；不在 CTest |
| `python scripts/benchmark_all_operators_xd.py --bench build-e13-shared/cubutterfly_plan_bench --library build-e13-shared/libcuntt.so --output /tmp/cubutterfly-xd-coverage --profile coverage --dry-run` | 仅生成 480 行 rank2/3 coverage manifest，不运行 benchmark | dry-run；实际去掉 `--dry-run` 为 **GPU** |
| `pytest -q tests/test_batch_precision_matrix.py tests/test_benchmark_all_operators_xd.py` | CPU manifest/queue 形状、计数、字节上限和 dry-run contract | CPU；不证明 lowering 可用 |

CMake 对 xD tests 的精确 CTest 名称见 [`CMakeLists.txt`](../CMakeLists.txt#L511)：`xd_fft_layout_correctness`, `xd_fft_fused_axis_correctness`, `xd_fft_wide_axis_correctness`, `xd_fft_pair_correctness`, `xd_fft_register_axis_correctness`, `xd_fft_cooperative_pair_correctness`, `xd_fft_register_l8_correctness`, `xd_fft_register_l16_correctness`, `xd_fft_pair_layout1_correctness`, `xd_fft_batch_tile_correctness`, `xd_fft_axis_block_caps_correctness`, `xd_fft_axis_exchange_correctness`, `xd_fft_axis_resources_correctness`, `xd_fft_local_core_correctness`, `rank3_permute_correctness`；同一段还注册 `ntt_stage_probe`, `mapping_record_roundtrip`, `public_planner_consistency`, `cuntt_correctness`, `cubutterfly_correctness`, `cubutterfly_c_api` 等。

## 3. Finite Workload Inventory

### Config files

`config/*workload*.json` 共 116 个 raw rows（未去重），具体范围如下。scope 文案本身已经声明的“follow-up/partial/not full acceptance”必须保留，不能把它们合并成全覆盖矩阵。

| 文件 | rows | 文件声明的范围 | 实际轴/组合 |
| --- | ---: | --- | --- |
| [`factor_followup_workloads.json`](../config/factor_followup_workloads.json#L1) | 2 | FP64 factor follow-up，不是 full FFT | FFT fp64，`logN=18,24`，B=1，OOP，forward |
| [`factor_partial_workloads.json`](../config/factor_partial_workloads.json#L1) | 8 | FP32/64 factor partial-readiness，full FFT separate | FFT fp32/fp64，`logN=18,24`，B=1/16，OOP，forward |
| [`install_contract_workloads.json`](../config/install_contract_workloads.json#L1) | 6 | 六类 operator 的 small integration cells | FFT/FWHT/structured fp32、subset/superset uint32、NTT word32；均 `logN=8,B=2`，OOP；structured 固定 `1,0.25,-0.5,1`，NTT p=2013265921 |
| [`install_search_workloads.json`](../config/install_search_workloads.json#L1) | 14 | library-exported candidate search | 前 6 行省略 `operator`（[`prepare_portable_campaign.py`](../scripts/prepare_portable_campaign.py#L65) 的 semantic default 补 FFT；raw config 自身不具备完整 operator 标签）；其余含 FWHT fp32/fp64、zeta/xor uint32、structured fp32、NTT word32/64。FFT 行为 `logN=16 B=64` OOP、`18/20` B=16/64/4 IP；FWHT `15,B=128`、`12,B=64`；其余大多 `logN=12,B=1024/128`；NTT p=998244353、576460756061519873 |
| [`pipeline_followup_workloads.json`](../config/pipeline_followup_workloads.json#L1) | 11 | multi-group/large FFT follow-up，不是 full matrix | FFT fp32/fp64 `logN=12,18,22,24`，B=1/4/16/64；FWHT/structured/subset/superset/NTT 各一行 `logN=12,B=64` |
| [`research_comprehensive_workloads.json`](../config/research_comprehensive_workloads.json#L1) | 72 | finite cross-operator research acceptance；明确排除 low precision、arbitrary、rank2 | FFT 29、NTT 14、FWHT 12、subset 5、superset 5、structured 6、xor 1；FFT 到 `logN=22`，最大逻辑元素 `2^24`；全部 config rows 都是 rank1 |
| [`staged_model_validation_workloads.json`](../config/staged_model_validation_workloads.json#L1) | 3 | staged model validation，不是 comprehensive | FFT fp32 `logN=12/18`、FWHT fp32 `logN=12`，B=32/16/32 |

对 72-cell 主矩阵，精确字段范围是：

* precision 仅 fp32/fp64（FFT/FWHT/structured）、word32/word64（NTT）、uint32（zeta/xor）；没有 fp16 或 bf16。
* FFT N/B 为 `logN={8,12,16,18,20,22}`、B 从 1 到 16384 的少量点；FP64 主矩阵只到 `logN=20,B=16`。只有一行 FFT fp32 `logN=8,B=32,element_stride=2,batch_stride=519`，只有一行 FFT fp32 `logN=8,B=256,in-place`。
* 方向字段主要是 forward；主矩阵的 inverse 仅 FFT fp32 `logN=12,B=32`、FFT fp64 `logN=16,B=8`、FWHT fp32 `logN=12,B=32`、NTT word64 `logN=16,B=4`，另有一行 NTT word64 natural->bit-reversed。zeta/structured 在 config 中没有 inverse row。
* NTT workload modulus 主要是 p=2013265921（word32）和 p=576460756061519873（word64）；安装搜索另有 p=998244353。config 没有 DIT/DIF/GS 字段、APPT static 行、input bit-reversed 行、rank2/3 行或 R2C/C2R `fft_type` 行。
* raw config 中出现的 batch 集合为 `1,2,4,8,16,32,64,128,256,1024,4096,16384`；没有以 stride/layout/in-place/embedding 与每个 operator 做笛卡尔积。

### Separate generated extension

[`scripts/build_batch_precision_matrix.py`](../scripts/build_batch_precision_matrix.py#L30) 是 CPU-only manifest generator，不是 GPU acceptance。其 suite 计数由 [`test_batch_precision_matrix.py`](../tests/test_batch_precision_matrix.py#L28) 锁定为：`batch_scaling=91`、`precision_scaling=78`、`precision_anchors=120`、`inverse_anchors=24`、`ntt_inverse=6`、`ntt_same_modulus=9`。它确实列出 FP16/BF16 的 native/FP32 accumulation anchors，最大输入字节 2 GiB、batch element cap `2^28`，但脚本明确不检查 device、不开 benchmark；因此只能证明 queue 语义，不证明当前 build 可执行。

该 extension 还明确排除：zeta uint64/float、multi-limb/RNS/negacyclic NTT、generic fp16-fp32（dedicated WMMA 不在 queue）、arbitrary lengths。structured inverse anchors 也故意 deferred。

### Rank-aware xD coverage script

`scripts/benchmark_all_operators_xd.py` 的 `coverage` profile 是四个 shape `(8,8),(32,32),(4,4,4),(8,8,8)`、B=`1,4,16`、forward/inverse、selected/resident-t128；operator/precision pairs 是 FFT fp32/fp64、FWHT fp32/fp64、NTT uint32/uint64、subset/superset uint32、structured fp32/fp64，共 `10*4*3*2*2=480` rows（见 [`test_benchmark_all_operators_xd.py`](../tests/test_benchmark_all_operators_xd.py#L234)）。它不包含 FP16/BF16、R2C/C2R、stride、embedding、in-place 或各 modulus/order 的矩阵；实际执行仍是 GPU。

## 4. Per-family Coverage Contract

“已有证据”只说明入口实际构造了哪些 case；“缺口”说明仍合法但没有当前矩阵/同源执行证据，或说明当前实现拒绝的组合。

| family / public entrypoint | 已有同源入口与准确证据 | 当前有限 coverage | 合法组合的缺口或明确边界 |
| --- | --- | --- | --- |
| **FFT C2C** / `cubutterflySetOperator(FFT)` | `tests/test_c_api.c`: `run_arbitrary_fft` N=5/B=1 (Bluestein, rank1)、`run_2d_arbitrary_fft` 3x5、`run_2d_power_fft` 4x8；`tests/check_xd.py` `_fft_case` 覆盖 C-ABI rank2/3、forward+normalized inverse、forced xD/replay；`tests/test_butterfly.cpp:test_fft` 和 `test_fp64` 覆盖 low-level backend/FP32/FP64。 | Public C-ABI evidence 是 fp32/fp64、小 B、形状 4x8/4x2x8/2x3x5 及 arbitrary；low-level FP32 具体有 `log_n=1..10` 及 larger `11,16`，FP64 有 `1,5,8,10,11`，另有 layout `10,12,14`。config 的 performance matrix 仍几乎全是 rank1 power-of-two。 | public C-ABI 没有 FP16/BF16 C2C execution case；没有对每个 rank2/3 shape 的 FP16/BF16/FP64 batch/layout 矩阵；standard arbitrary rank3 明确拒绝（[`src/c_api.cpp`](../src/c_api.cpp#L2887)），arbitrary FFT OOP-only。DIT/DIF/GS 没有独立入口。 |
| **FFT arbitrary** | `test_c_api.c:run_arbitrary_fft`, `run_2d_arbitrary_fft`; `tests/check_xd.py` `fft_rank3_2x3x5_*` 是 xD generic rank3，但不是 rank3 Bluestein arbitrary。 | rank1 N=5、rank2 3x5 C-ABI correctness；config 没有 arbitrary FFT row。 | legal standard arbitrary FFT 仅 complex FP32/FP64；rank3 Bluestein、in-place、R2C/C2R arbitrary、low precision arbitrary 没有合法 current path/evidence。 |
| **NTT** / `CUBUTTERFLY_OPERATOR_NTT` | C API: `run_2d_ntt` rank2 4x8 UINT64 fwd/inv；`run_arbitrary_ntt` rank1 N=3 UINT64；hierarchical N=4096/N=2^20; `run_ntt_output_order` word32/word64 N=8 B=2 fwd/inv natural and bit-reversed output；C++ `test_ntt.cpp:test_forward_backend`, `test_inverse_backend`, `test_custom_modulus`, `test_roundtrip`, `test_hybrid2d_backend`, compact stage, stage pipeline, APPT bijection。 | word32/64、多个 modulus、N=2^1..2^20、forward/inverse、natural/bit-reversed output、APPT layout metadata 有 correctness evidence；72-cell performance 只含 14 rank1 rows。 | APPT static 执行仍只 rank1 N=2^20 OOP；没有 input bit-reversed legal row；rank1 direct strided/IP/embedding 不是当前 direct contract；rank2/3 arbitrary NTT 的完整同源矩阵没有；任意 modulus/root compatibility 没有穷举。`normalize_inverse` 对 NTT 被强制 canonicalized 为 inverse_n，不能算另一种语义。没有 DIT/DIF/GS selector。 |
| **FWHT** / `CUBUTTERFLY_OPERATOR_FWHT` | `test_c_api.c`: `run_2d_fwht` rank2 4x8 forward、`run_xd_rank3_fwht` 2x2x2 fwd/inv、`run_embedding_fwht` N=6->8、`run_strided_embedding_fwht` output stride2；`tests/check_xd.py:_fwht_case`, `_strided_fwht_case`, `_strided_rank3_fwht_case`, `_inplace_*`, `_embedding_*`; C++ low-level `test_fwht`, `test_low_precision_contract`。 | FP32 public C-ABI rank1/2/3、OOP/IP、positive strides、zero-extension 以及 fwd/inv small cases；FP16/BF16 low-level native/FP32 accumulation 有 N=8/11、B=3 的 evidence；FP64 low-level 有 fwd/inv。 | public C-ABI 没有 FP16/BF16/FP64 execution matrix；config 只有 fp32；没有 rank3 low-precision、rank3 embedding+stride 的笛卡尔覆盖，没有大 N/B public C-ABI row。zero-extension 要 OOP，rank1 common layout 要 input/output exactly equal。 |
| **subset zeta** | `test_c_api.c:run_2d_zeta` 是 rank2 UINT32 4x8 fwd/inv；`tests/check_xd.py:_zeta_case` subset fwd/inv；`test_butterfly.cpp:test_xor_zeta`/`test_extended_zeta_operators` 覆盖 subset/superset/Xor alias 的 N=8/11 与 backends。 | uint32、power-of-two、small rank2 correctness；config subset 是 5 个 rank1 forward uint32 rows（N=8/12/20，B=1/256/4096）。 | 没有 public C-ABI rank1/3、大 N/B、stride/IP/embedding、FP16/BF16/FP64 或 uint64 zeta；config 没有 inverse。zero-extended zeta 如需执行必须单列 OOP physical extent；generated extension 将 zeta uint64/float 明确列为 out-of-scope。 |
| **superset zeta** | `tests/check_xd.py:_zeta_case` 和 `test_butterfly.cpp:test_extended_zeta_operators`；C API main 只调用 `run_2d_zeta` 的 subset operator，未调用 superset C API function。 | uint32 rank2 xD Python C-ABI fwd/inv、low-level superset fwd/inv；config 5 个 rank1 forward rows。 | **C-ABI superset execution 在 `test_c_api.c` 缺失**；没有 rank1/3、large B/N、strided/IP/embedding、inverse performance row。Xor-zeta config row 和 `XorZeta` C++ alias 是 subset compatibility alias，不能算第三个独立 arithmetic kernel。 |
| **structured 2x2** | `test_c_api.c:run_2d_structured` rank2 4x8 FP32 fwd/inv；`tests/check_xd.py:_structured_case` 使用 per-axis matrices；`test_butterfly.cpp:test_structured_operators` 覆盖 FP32/FP64、多 backend、broadcast/per-stage 与 low-level zero-extension。 | FP32/FP64 low-level correctness；public C-ABI rank2 fixed per-axis matrices；config 6 个 rank1 FP32 forward rows且固定 matrix `1,0.25,-0.5,1`。 | public C-ABI 没有 FP16/BF16/FP64 execution；没有 rank1/3 C-ABI、stride/IP、large N/B matrix policy；config 没有 inverse、broadcast-vs-per-stage 或 varying coefficient conditioning。generated batch queue structured inverse deferred。 |
| **R2C/C2R compact real FFT** / `cubutterflySetFftTransformType` | `tests/check_xd.py:_real_fft_case` 通过 ctypes public C ABI 绑定 `cubutterflySetFftTransformType`，同源当前 build 已验证 rank1 `(5,),(6,)`、rank2 `(4,6),(3,5)`、rank3 `(2,3,4),(2,3,5)` 的 FP32/FP64 R2C+C2R，另有 scale and mapping replay；`tests/test_xd_fft_layout.cu:run_real_case` 是 internal `XdRealFftPlan` + cuFFT oracle。 | public ctypes and internal xD both prove packed OOP standard compact real semantics for rank1/2/3 and larger internal mapping cases；inverse normalization/output scale checked。rank1 是 current positive evidence，不是 unsupported boundary。 | `config/*workload*.json` 完全没有 `fft_type`/R2C/C2R row；`test_c_api.c` 也没有 R2C/C2R function。当前 source 明确拒绝 FP16/BF16 real compact、embedding、explicit strides、in-place（[`src/c_api.cpp`](../src/c_api.cpp#L2438)）。没有 real low-precision 或 real performance matrix。 |

## 5. Cross-cutting Dimension Ledger

| dimension requested | 已覆盖到哪里 | 仍缺的精确组合 |
| --- | --- | --- |
| FP16 / BF16 | `tests/test_butterfly.cpp:test_low_precision_contract`：FP16/BF16，Native 与 FP32 accumulation，FP32/structured/FWHT/FFT low-level；CPU generated extension 还列 6 floating contracts。 | 没有 `config/*workload*.json` row；没有 public C-ABI C2C/FWHT/structured/zeta rank2/3 low precision；没有 R2C/C2R low precision；没有同源 GPU batch/large-N acceptance。 |
| FP32 / FP64 | C API and C++ tests cover selected forward/inverse; research 72 matrix has FP32/FP64 for FFT/FWHT/structured. | 主 matrix 不提供 full rank2/3, stride, IP, embedding, real, coefficient, modulus cross-product；FP64 zeta/NTT semantics不适用。 |
| NTT word32/word64 | `test_ntt.cpp`, C API output-order, config p=2013265921/998244353/576460756061519873；Python p=97 rank2 harness。 | 不是任意 modulus/order/rank/stride/layout 全覆盖；input bit-reversed 没有合法 public path，APPT 只有一个 rank1 N=2^20 contract。 |
| direction / normalization | C API and Python correctness execute fwd/inv for FFT/FWHT/NTT/zeta/structured selected cases；config inverse rows少。NTT normalize flag被忽略并固定 inverse_n。 | zeta/superset/structured inverse performance cells缺失；R2C/C2R 是 transform type boundary，不应与 C2C inverse 合并。 |
| DIT / DIF / GS | public API/config/test search无该 enum/field/selector；内部 codelet/radix 只说明实现细节。 | 三者没有可执行的独立 contract，不能宣称覆盖；若以后要比较，必须先新增 selector、reference semantics 和 workload key。 |
| rank 1/2/3 | public C ABI `test_c_api.c` 有 FFT/NTT/zeta/structured/FWHT rank2，FWHT rank3；`check_xd.py` 有 C2C/real rank2/3；internal xD rank2/3。 | 72-cell/config 全部 rank1；各 family 的 rank3、non-power-of-two、in-place、stride、embedding 没有完整矩阵。`rank3_permute` 只有 permutation，不是 operator correctness。 |
| stride/layout | config 仅一行 FFT element stride2 + padded batch stride；C++ layout tests有 element_stride2/padded batch/IP；Python C ABI有 FWHT rank2/3 positive strides；NTT direct rank1要求 packed。NTT natural/APPT/bit-reversed output 有 small probes。 | 没有 FFT/NTT/zeta/structured 的 public C-ABI general strides matrix；没有 R2C/C2R stride；没有 input bit-reversed；APPT execution matrix仅 rank1 N=2^20 OOP。 |
| in-place / out-of-place | config FFT 有 N=8 B=256 IP、install-search 有 FFT N=18/20 IP；C++/Python C ABI 有 C2C FFT、FWHT IP small cases；其他 config大多 OOP。 | 没有 NTT/zeta/structured/R2C/C2R public execution IP matrix；embedding source要求 OOP；real compact source明确拒绝 IP。 |
| zero-extended embedding | C API FWHT N=6->8 rank1 + strided, Python rank2/3 FWHT embedding, C++ FP32 FWHT/structured zero-extended。 | FFT/NTT standard arbitrary 与 embedding 是不同语义；没有 NTT/zeta/structured rank2/3 embedding matrix，没有 embedding IP，R2C/C2R embedding被拒绝。 |
| NTT length/modulus/order | C++ reaches logN 1..20 and several primes; config 72 rows use two main primes; C API N=3/N=4x8/N=2^20 metadata; bit-reversed output and APPT metadata probes。 | 没有所有 legal N x modulus x word_bits x input/output order；standard arbitrary rank3 rejected, direct NTT stride/IP constraints limit rows；DIT/DIF/GS absent。 |
| N/B and large invocation | research config FFT max `N=2^22`, max `N*B=2^24`; config B max 16384 for FWHT/FFT small N. C++ NTT/C API probe reaches N=2^20; generated CPU manifest reaches FFT logN23 and memory cap 2 GiB. | generated 2 GiB rows are not executed; no same-build GPU evidence for all large B/N, no rank2/3 large-area matrix, and no operator-wide Cartesian product. |

## 6. Queue and Historical-Evidence Boundary

当前队列并不等于当前 build 已验收：

* [`results/active_experiment_queue.json`](../results/active_experiment_queue.json#L1) 仍指向 `results/reordered_queue_20260925/priority_repaired`，deadline 是 `2026-09-28 21:00 +08`，相对本文日期已过期。
* [`results/experiment_plan_index.json`](../results/experiment_plan_index.json#L1) 的 current recovery 在外部 worktree `cuButterfly-ntt-hybrid-order`；next integrated acceptance 是外部 `cuButterfly-v0.9.1-rc2`，状态 `active-cli-fix1-weak-fft-then-ntt-fwht`，monitoring deadline `2026-10-07 09:00 +08`，不是本 checkout 的新 build。
* [`results/integrated_acceptance_20260930/planned_queue.json`](../results/integrated_acceptance_20260930/planned_queue.json#L1) 的 `integrate`/`build-precompile`/A100 correctness jobs 是 `planned-not-launched`；它们的 cohorts 是计划人口，不是实测 coverage。
* [`docs/integrated_acceptance_rc2.md`](integrated_acceptance_rc2.md#L1) 明确写着 “planned, awaiting source integration and a new build”，旧 rc.1 早于 FFT/NTT/FWHT fixes；其 “36/36” 和 “standalone comparison complete” 都是命名 build/GPU 的证据，不能提升为当前 build acceptance。
* [`docs/research_acceptance.md`](research_acceptance.md#L42) 的 72-cell 是有限矩阵，且明确说 low precision、arbitrary length、multidimensional transform 在矩阵外；其 5/72 snapshot、旧 A100 report 和旧 module population 只能作历史记录。该文也规定历史 timing 不导入新 run（[L63-L72](research_acceptance.md#L63)）。

计划中的 integrated rc.2 cohorts（[`integrated_acceptance_rc2.md`](integrated_acceptance_rc2.md#L43)）也只表示待执行人口：General 749（core 74 + precision/batch 284 + crypto 208 + composed 138 + extensions 29 + planning 16）、FFT repair 36、FFT weak-region 10、NTT external 346（其中声明 214 comparable、78 baseline-unsupported、54 correctness exclusions）、FWHT full-state 36、FP64 direct/I/O 68，以及 stage controls 56 CT/GS basis points 和 24 workloads x 4 variants。该计划还要求 integrate/build-precompile 之后在两个 A100 target 上 fresh rerun；这些数字不能加到上面的 current coverage 计数。

### Current-build acceptance gate

一个 row 只有同时满足以下条件，才能在新 acceptance summary 中计为 `measured/current`：

1. workload key 完整包含 operator、storage/compute/accumulation、logical/physical shape、rank、batch、direction、normalization、placement、strides/batch strides、length mode、modulus/word bits、NTT input/output order、FFT type；
2. source revision、public library、benchmark binary、JIT/module identity 和 build hash 一致，且不是上述外部/旧 worktree；
3. correctness 在同一 public entrypoint 通过，unsupported baseline/empty timing 单列，不能填 0 或算 win；
4. hardware UUID、memory、driver/protocol 与 row 记录一致；
5. 若是 performance/selector row，必须有当前 source 的 fresh plan selection/replay；历史 mapping 不能单独恢复为当前通过。

因此，本文件列的“已有证据”不能自动转化为本轮 acceptance；它只定义下一次同源执行的有限 contract 和缺口。

## 7. Minimum Next Matrix (Finite, Not Cartesian)

为了补当前最高信号缺口而不把任务扩大成无限 cross-product，最小可执行增量应至少保留这些独立 rows：

| row group | finite rows to add | purpose |
| --- | --- | --- |
| low precision public API | FP16/BF16 × native/FP32 accumulation，FFT/FWHT/structured，rank1 N=8/B=3 和 rank2 4x8/B=2，各 forward/inverse where legal | 把 low-level evidence 接到 public C ABI；R2C/C2R 仍应列为 explicit unsupported |
| real public API | R2C/C2R FP32/FP64，rank1 odd/even、rank2 4x6、rank3 2x3x5，B=2，packed OOP | 将 `check_xd.py` contract 接入同源 acceptance；不添加 stride/IP/embedding，因为 source 当前拒绝 |
| NTT semantic anchors | word32/word64，各一个 N=2^12 和 N=2^16，forward/inverse，p=2013265921、998244353、576460756061519873（word-compatible），natural->natural；另一个 legal bit-reversed output row | 分离 modulus/order/direction；不是所有 primes 的穷举 |
| rank/layout | FFT/FWHT/structured rank2 + rank3 各一条 positive strided OOP、各一条 IP（仅 source legal）；zeta/NTT 单列 unsupported 或 packed-only | 不把 permutation-only test 当 transform coverage，并记录 physical layout |
| large N/B | 复用 batch extension 的边界：FFT fp32 `logN=23`、FP64 `logN=20`、NTT word64 `logN=20`，B=1/16/256；每组只取 current-build correctness anchor | 验证 generated queue 到底能否落地；2 GiB manifest 本身不算测量 |

这个 minimum matrix 仍不包含 DIT/DIF/GS，因为当前 API 没有这三个可选择 contract；若研究目标需要它们，先改 API/plan key 和 reference，再新增 workload。

## 8. Current Same-Build Execution (2026-10-06)

本节记录本轮修复后的同源验收结果。性能结果均使用当前 `build-e13-shared`、A100 80GB PCIe、CUDA-event timing、warmup=10、repeat=50、trials=3；plan 创建、校准和 warmup 不计入 kernel 时间。GPU runner 默认并显式传递 `CUBUTTERFLY_COMPILE_MODE=research`。

* `results/all_operators_xd_20261006_gpu1_scale_v4`：240 个 selected scale 行全部完成，失败 0；覆盖 128x128、16^3、32^3、64^3，rank 2/3，B=1/4/16，正逆方向和六类算子。FFT forward 相对 cuFFT 几何平均 **1.156x**（24 行，24 行胜出，范围 1.058--1.286x）。该结果使用修复后的库 hash `8d8edab3…`，与旧 v3 的启动开销异常分开记录。
* `results/all_operators_xd_20261006_gpu1_coverage_v4`：selected/resident-t128 各 240 行，共 480/480 行完成，失败 0、重复 0、缺失 0。selected 的 FFT forward 相对 cuFFT 几何平均 **1.357x**（24 行，23 行胜出，最小 0.969x）；resident 控制为 0.649x（24 行，6 行胜出）。跨六类算子的 selected 相对 resident 控制中位数为 **1.874x**，218/240 个配对单元 selected 更快。
* `results/all_operators_xd_20261006_gpu1_coverage_v4_correctness.json`：当前库 selected coverage 的 240 个单元真实执行并通过输出校验，失败 0、跳过 0、未完成 0；覆盖 FFT/FWHT/NTT/subset-zeta/superset-zeta/structured-2x2，rank 2/3，FP32/FP64 或 UINT32/UINT64，B=1/4/16，正逆方向。该文件只记录正确性，不把 timing 当作正确性证据。
* `results/all_operators_xd_20261006_gpu1_coverage_v4_resident_correctness.json`：selected 与 resident-t128 的 480 个 coverage 单元全部真实执行并通过输出校验，失败 0、跳过 0、未完成 0。
* `results/all_operators_xd_20261006_gpu1_scale_v4_correctness.json`：同一构建的 240 个尺度单元真实 GPU 输出校验已完成，240/240 通过，失败 0、跳过 0。
* `results/operator_coverage_gap_20261006/gpu1_nonlarge_all.json`：有限缺口清单的非大尺寸 104 行中，100 行 candidate/packed boundary 通过，4 行 FP16/BF16 R2C/C2R 按契约返回 `expected-unsupported`。`gpu1_large_64g.json` 的 9 个大 N×B 锚点中 8 个通过单位脉冲全量校验；FP32 `logN=23,B=256` 因当前进程可用显存不足而记录为 CUDA OOM 资源边界。
* `results/all_operators_xd_20261006_cpu_full.json`：CPU 参考矩阵 360 行中 348 通过、0 失败、12 行明确跳过；跳过项仅为 64^3 NTT 的二次复杂度 CPU 参考，不代表 GPU 不支持。layout/in-place/embedding/real 等价检查 4/4 通过。
* `tests/check_xd.py --case resident` 在同一 A100 80GB 上 17/17 通过；`--case real` 的 R2C/C2R FP32/FP64 rank1/rank2/rank3 及 register/scale cases 15/15 通过，包含 odd/even rank1。当前构建的 CTest 全部 **63/63** 通过，覆盖 NTT stage、C API、xD FFT 各 lowering、stage overlap、structured、设计空间和安装 consumer；mapping 测试由 CTest 固定 research compile mode。

未执行的组合不会被计入 current pass。`results/operator_coverage_gap_20261006/manifest.json` 固化了第 7 节的有限增量清单；其中 R2C/C2R FP16/BF16、显式 stride/in-place/embedding 仍是实现边界，FP16/BF16 public C-ABI、完整 large-N/B performance 和 DIT/DIF/GS 仍保持单独的缺口或 API 边界。rank1 compact real 已有 positive evidence，不再列为 unsupported。
