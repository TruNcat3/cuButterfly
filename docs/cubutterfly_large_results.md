# Large-Length cuButterfly Results on V100

## Hierarchical Backend

The common runtime now supports `logN=6..20` through a hierarchical backend.
Its first kernel fuses `local_stages=5..10` in one shared-memory tile. Remaining
stages use ping-pong global kernels. `local_stages`, radix, and tile threads are
searched independently of the operator.

The broad search fixes approximately `2^22` points per invocation and sweeps
`logN=12,14,16,18,20`, `local_stages=6,8,10`, radix-2/radix-4, and 128/256
threads. The selected points below are medians of five process trials with 50
warmups and 100 timed executions.

| Operator | logN 12 | logN 14 | logN 16 | logN 18 | logN 20 | Selected mapping |
|:--|--:|--:|--:|--:|--:|:--|
| FWHT Gbutterfly/s | 174.42 | 126.56 | 105.06 | 93.08 | 85.94 | `L10`, radix-4, 256 threads |
| XOR-zeta Gbutterfly/s | 180.33 | 129.64 | 105.45 | 94.40 | 86.14 | `L10`, radix-4, 256 threads |
| FFT Gbutterfly/s | 85.77 | 64.26 | 53.47 | 46.47 | 34.96 | `L10`, radix-4, 256 threads |
| cuFFT Gbutterfly/s | 283.14 | 239.19 | 187.09 | 178.11 | 199.99 | vendor plan |
| FFT/cuFFT throughput | 30.3% | 26.9% | 28.6% | 26.1% | 17.5% | resident transform |

The stable `L10` optimum says that the V100 benefits from maximizing state
residency inside the implemented tile. It does not say ten stages are a
universal optimum: ten is the current largest available local core.

## Online-Reorder Backend

The online path writes prefix result `(h,l)` directly at physical address
`(l,h)`. This store is already required at the stage-group boundary, so the
permutation adds no separate full-array pass. The suffix reads each `h` column
contiguously, retains it in shared memory for all remaining stages, and writes
the requested logical output layout. `reorder_columns` controls how many
independent columns share one suffix CTA.

The table below is a same-run comparison: approximately `2^22` points, radix-4,
256 threads, 50 warmups, 100 repetitions, and five process trials. Each backend
uses its best tested split and column count for that operator and length.

| Operator | logN | Hierarchical ms | Online ms | Online speedup | Online mapping |
|:--|--:|--:|--:|--:|:--|
| FWHT | 12 | 0.144374 | 0.114145 | 1.265x | `L10,C128` |
| FWHT | 14 | 0.231956 | 0.237916 | 0.975x | `L10,C16` |
| FWHT | 16 | 0.314020 | 0.310149 | 1.012x | `L8,C1` |
| FWHT | 18 | 0.400804 | 0.313856 | 1.277x | `L9,C1` |
| FWHT | 20 | 0.488264 | 0.319795 | 1.527x | `L10,C1` |
| FFT | 12 | 0.303084 | 0.268708 | 1.128x | `L10,C64` |
| FFT | 14 | 0.462899 | 0.347167 | 1.333x | `L10,C64` |
| FFT | 16 | 0.627569 | 0.420168 | 1.494x | `L10,C8` |
| FFT | 18 | 0.812104 | 0.446792 | 1.818x | `L10,C4` |
| FFT | 20 | 1.199616 | 0.640174 | 1.874x | `L10,C1` |
| XOR-zeta | 12 | 0.139612 | 0.113910 | 1.226x | `L10,C128` |
| XOR-zeta | 14 | 0.226468 | 0.204063 | 1.110x | `L10,C16` |
| XOR-zeta | 16 | 0.313037 | 0.310630 | 1.008x | `L8,C1` |
| XOR-zeta | 18 | 0.399790 | 0.313876 | 1.274x | `L9,C1` |
| XOR-zeta | 20 | 0.487004 | 0.319519 | 1.524x | `L10,C1` |

This confirms the APPT mechanism, but also shows that reordering is not free in
the transaction sense. At `logN=20`, Nsight Systems attributes 382.942 us to
the prefix with permutation store and 297.726 us to the fused suffix. The old
prefix was 307.390 us, while its ten suffix kernels summed to 897.401 us. The
online store therefore makes the prefix about 25% slower but reduces the suffix
from ten kernels to one and cuts total traced time from 1.205 ms to 0.681 ms.

FFT remains behind cuFFT after removing the stage-by-stage boundary:

| logN | Online FFT ms | cuFFT ms | Online/cuFFT throughput |
|--:|--:|--:|--:|
| 12 | 0.268708 | 0.088740 | 33.0% |
| 14 | 0.347167 | 0.122368 | 35.2% |
| 16 | 0.420168 | 0.179272 | 42.7% |
| 18 | 0.446792 | 0.211978 | 47.4% |
| 20 | 0.640174 | 0.209623 | 32.7% |

The remaining gap is no longer attributable to ten global stage boundaries.
The next attribution must separate permutation-store sectors, bit-reversed
prefix loads, twiddle traffic, arithmetic/core efficiency, and occupancy.

## cuFFT Trace Attribution

Nsight Systems traces use FP32 `logN=20`, batch 4, forward, unnormalized,
resident transform semantics. Trace overhead changes CUDA-event time, so the
attribution uses summed GPU kernel durations.

| Implementation | Kernel sequence | GPU kernel sum |
|:--|:--|--:|
| cuButterfly | one 307.390 us prefix plus ten 84.255-104.127 us stages | 1.204791 ms |
| online-reorder | one 382.942 us permuted prefix plus one 297.726 us suffix | 0.680668 ms |
| cuFFT | two `regular_fft<1024,...>` kernels | 0.247358 ms |

Each FP32 complex array is 32 MiB. A full-array load/store through CUDA's global
address space therefore issues 64 MiB of logical data requests. The current
out-of-place hierarchy makes one such pass for the fused prefix and ten for the
remaining stage kernels: approximately 704 MiB of requested value traffic,
before twiddle requests. This is not the same as 704 MiB of HBM2 traffic:
global accesses may be served by L2, and a kernel boundary does not flush L2.
The 32 MiB working set is larger than V100's L2, so substantial HBM2 traffic is
expected, but its amount must be measured rather than inferred. cuFFT realizes
a `1024 x 1024` decomposition in two highly fused kernels; the trace establishes
the launch decomposition and timing advantage, not its exact memory traffic.

NCU counter collection is prepared in `scripts/profile_cubutterfly_fft.sh`.
The current unprivileged shell receives `ERR_NVGPUCTRPERM`; run the script with
profiling permission to confirm actual DRAM bytes, L2 hit rate, and long
scoreboard stalls.

```bash
sudo -E ./scripts/profile_cubutterfly_fft.sh
sudo chown -R "$USER:$USER" results/ncu_fft
```

## External FWHT Baseline

Dao-AILab `fast-hadamard-transform` commit
`e7706faf8d1c3b9f241e36860640ad1dac644ede` is installed in an isolated
PyTorch 2.1.2/CUDA 11.8 environment. Upstream only emits `sm_75+`; the recorded
local patch adds an `sm_70` gencode target without changing kernel source.
Every shape and dtype passes the normalized double-transform identity check.

The FP32 comparison uses the same shapes, batch, forward/unnormalized semantics,
approximately `2^22` points, 50 warmups, 100 repetitions, and five trials.

| logN | original cuButterfly ms | warp-register ms | Dao ms | integrated/Dao throughput |
|--:|--:|--:|--:|--:|
| 8 | 0.051323 | 0.043131 | 0.043172 | 100.1% |
| 10 | 0.056863 | 0.044626 | 0.043960 | 98.5% |
| 12 | 0.140534 | 0.045199 | 0.044073 | 97.5% |
| 14 | 0.227369 | 0.055992 | 0.053617 | 95.8% |
| 15 | 0.270643 | 0.069243 | 0.063949 | 92.4% |

Dao's single-CTA register/shared-memory core handles complete transforms through
`N=32768`. The new `--local-exchange warp-register` unit admits that established
register hierarchy as a processing-unit option while retaining cuButterfly
ownership of mapping, batching, layout, and scheduling. This closes the former
multi-pass gap to within 0%-8.3% on these shapes; the original column remains as
the before-integration control.

## Reproduction

```bash
python3 scripts/sweep_cubutterfly_designs.py \
  --operators fwht fft xor-zeta \
  --backends hierarchical cufft \
  --compute-units radix4 --logNs 12 14 16 18 20 \
  --tile-thread-options 256 --hierarchical-local-stages 10 \
  --target-points 4194304 --warmup 50 --repeat 100 --trials 5 \
  --normalizations none --output results/cubutterfly_large_v100_selected_raw.csv

python3 scripts/sweep_cubutterfly_designs.py \
  --operators fwht fft xor-zeta --backends online-reorder \
  --compute-units radix4 --logNs 12 14 16 18 20 \
  --tile-thread-options 32 64 128 256 \
  --hierarchical-local-stages 6 8 9 10 \
  --reorder-column-options 1 2 4 8 16 32 64 128 256 \
  --target-points 4194304 --warmup 20 --repeat 100 --trials 5 \
  --normalizations none --output results/cubutterfly_online_reorder_v100_raw.csv

./scripts/profile_cubutterfly_fft_nsys.sh

conda create -y -n cubutterfly-baselines \
  python=3.10 pip setuptools wheel packaging ninja
conda activate cubutterfly-baselines
pip install \
  torch==2.1.2 --index-url https://download.pytorch.org/whl/cu118
./scripts/install_external_fht.sh
python3 scripts/benchmark_external_fht.py --sm70-patched \
  --logNs 8 10 12 14 15 --dtypes fp16 bf16 fp32 \
  --target-points 4194304 --warmup 50 --repeat 100 --trials 5 \
  --output results/external_dao_fht_v100_raw.csv
```
