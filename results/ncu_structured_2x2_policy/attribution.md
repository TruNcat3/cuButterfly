# Structured 2x2 V100 Counter Attribution

NCU uses base clocks and replay; its time is mechanism evidence, not the CUDA-event performance authority.

| Case | Kernels | Time us | DRAM MiB | Warp inst M | FP32 inst M | Active warps | Barrier stall | Scoreboard stall | Reg/thread | Shared B |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| structured8_temporal_r4 | 1 | 52.160 | 32.17 | 10.85 | 67.11 | 91.5% | 23.9% | 41.0% | 28 | 1024 |
| fwht8_temporal_r4 | 1 | 49.280 | 32.23 | 8.95 | 33.55 | 90.8% | 18.2% | 48.5% | 16 | 1024 |
| structured12_hierarchical_r8 | 3 | 147.200 | 96.45 | 19.35 | 100.66 | 82.4% | 7.3% | 70.1% | 39 | 4096 |
| structured12_warp_register_broadcast | 1 | 52.992 | 31.88 | 6.87 | 100.66 | 65.5% | 5.8% | 19.2% | 40 | 16384 |
| structured12_warp_register_per_stage | 1 | 56.544 | 32.06 | 6.96 | 100.66 | 56.1% | 5.5% | 25.5% | 43 | 16384 |
| fwht12_hierarchical_r8 | 3 | 142.752 | 96.36 | 16.92 | 50.33 | 91.2% | 5.9% | 71.8% | 23 | 4096 |
| fwht12_warp_register | 1 | 46.112 | 31.68 | 3.36 | 55.84 | 66.0% | 8.9% | 21.6% | 32 | 16384 |
| structured20_online_r8 | 2 | 381.376 | 65.11 | 23.54 | 169.87 | 69.6% | 17.3% | 14.6% | 40 | 4096 |
| fwht20_online_r8 | 2 | 374.752 | 65.24 | 19.63 | 85.98 | 91.4% | 13.6% | 13.5% | 30 | 4096 |

## Attribution

### logN=8: dense arithmetic is visible but not dominant

With the same temporal radix-4 mapping, Structured2x2 takes 1.058x the FWHT time, executes 2.00x the FP32 instructions, and uses 28 versus 16 registers/thread. Active warps stay above 91% for both, so the extra dense arithmetic and barrier pressure produce only a small end-to-end penalty.

### logN=12: the gap is the physical processing unit

The matched hierarchical paths differ by only 3.1% overall. The two global suffix kernels are effectively equal; the dense prefix raises registers/thread from 23 to 39 and reduces active warps from 92.6% to 70.7%. The generated FWHT register codelet is 3.10x faster than hierarchical FWHT and moves 3.04x less DRAM traffic by retaining all stages in one kernel. This isolated the original Structured2x2 opportunity to a generated register-resident matrix unit, not to a different architecture schedule. The generated Structured2x2 register unit takes 52.992 us, a 2.78x speedup over its hierarchical path.

### logN=12 register control: the current residual is arithmetic

Structured2x2 takes 1.149x the FWHT register time while moving 1.006x the DRAM bytes. It executes 2.05x the warp instructions and 1.80x the FP32 thread instructions, raises registers/thread from 32 to 40, and reduces active warps from 66.0% to 65.5%. Shared bank conflicts are negligible in both kernels. The 40-register dense kernel retains essentially the same active-warps level as FWHT (65.5% versus 66.0%). This rules out global layout, shared-bank behavior, and an achieved-occupancy collapse as the primary remaining cause.

### logN=12 coefficient policy

Under identical matrix values, the broadcast specialization takes 52.992 us versus 56.544 us for the table path, a 1.067x speedup. It changes registers/thread from 43 to 40, raises active warps from 56.1% to 65.5%, and lowers long-scoreboard stall from 25.5% to 19.2%. FP32 instruction count is unchanged.

### logN=20: the apparent advantage is not robust

At base clocks the matched online Structured2x2 path is 1.8% slower, executes 1.20x the warp instructions and 1.98x the FP32 instructions, while DRAM traffic and cache hit rates are nearly unchanged. Its 40-register dense kernels reduce active warps to about 70%, but low DRAM utilization and similar scoreboard stalls allow most added arithmetic to overlap. The earlier sequential-scan speedup must not be used as a performance claim.

## Methodological Consequence

The counters separate the replaceable local unit from the space/time schedule. Dense local arithmetic changes register and instruction demand, while the mapping transition across lengths remains temporal -> hierarchical -> online reorder. The generated register result confirms that the transport hierarchy can be reused without changing the architecture schedule. Against the same-transport FWHT register core, Structured2x2 takes 1.149x the base-clock time with 1.006x DRAM traffic, 2.05x warp instructions, 1.80x FP32 thread instructions, and 40 versus 32 registers/thread. The 40-register dense kernel retains essentially the same active-warps level as FWHT (65.5% versus 66.0%). The nearly identical DRAM volume and low shared bank-conflict counts isolate the current residual to dense-pair arithmetic; register allocation remains a design constraint at other lengths. Changing the architecture paradigm is not supported by this evidence.
