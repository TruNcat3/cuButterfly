# v0.6/v0.7 Crossover Attribution

If the v0.7 candidate set strictly contains the v0.6 candidate set, then `min(T_v0.7) <= min(T_v0.6)` at every workload. The measured crossovers therefore diagnose incomplete physical-space inclusion, coarse scheduling quanta, or stale controls; they are not a property of the NTT dependency graph.

## logN=12 uint32 Wave Boundary

| batch | Hybrid2D ms | hybrid Td=6 ms | hybrid Td=12 ms | v0.7/v0.6 | CTA waves | last-wave fill |
|---:|---:|---:|---:|---:|---:|---:|
| 80 | 0.018872 | 0.017906 | 0.017902 | 1.054x | 1 | 50.0% |
| 160 | 0.031320 | 0.027689 | 0.027673 | 1.132x | 1 | 100.0% |
| 256 | 0.048912 | 0.056918 | 0.056949 | 0.859x | 2 | 60.0% |
| 512 | 0.085256 | 0.099375 | 0.098488 | 0.866x | 4 | 20.0% |

Td=6 and Td=12 differ by less than 1.6% at every point, so the generated-table threshold is not the crossover cause. The resident kernel launches one CTA per transform and is limited to two active CTAs/SM by its register footprint. Batch 160 fills one 80-SM resident wave; batch 256 requires a second wave with only 60% of its slots used. Hybrid2D exposes finer intra-transform CTA work and avoids this same quantization.

## Updated logN=20 Control

| workload | latest v0.6-best ms | packet/lane32 best ms | v0.7/v0.6 |
|:--|---:|---:|---:|
| uint32 logN=20 batch=16 | 1.014657 | 1.085352 | 0.935x |

The historical packet result used the older 17:13 v0.6 control. Against the later 9:11 resident/Hybrid2D envelope, packet is no longer the winner at this point. Baseline evolution was a second source of apparent crossover.

## Consequence

The release selector must take the lower envelope of both generations. The research generator must also import the v0.6 Hybrid2D local core and multi-CTA data-space decomposition as legal v0.7 physical mappings. Once that candidate-space inclusion is enforced, a remaining crossover would indicate a measurement or selection bug rather than a legitimate architecture result.
