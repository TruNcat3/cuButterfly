# HybridDataflow Td model evaluation

Training uses the full Td scan at six representative batches. Testing uses unseen
batch multiples of 16 from the dense scan; overlapping batches are excluded.

| model | scenarios | top-1 | top-2 | geomean regret | p95 regret | worst regret |
|---|---:|---:|---:|---:|---:|---:|
| static-guarded | 140 | 0.693 | 0.986 | 1.0052 | 1.0357 | 1.0694 |
| ridge | 140 | 0.279 | 0.607 | 1.0402 | 1.1306 | 1.5197 |
| hist-gradient | 140 | 0.386 | 0.864 | 1.0188 | 1.0945 | 1.1510 |
| random-forest | 140 | 0.493 | 0.929 | 1.0250 | 1.0639 | 1.5197 |

## Features and decision

The feature set combines packet count, critical serial slots, token utilization, cubin
register count, analytically derived resident CTAs, continuous and discrete CTA density,
partial waves, and interactions with word width and graph length. Cubin resources are
extracted by `extract_hybrid_dataflow_resources.py`; no timing label is used as a feature.

The runtime admission target is geomean regret <=1.02, p95 <=1.05, and worst <=1.10 on
unseen batches. The static guarded policy passes all three. The best learned model
(`hist-gradient`) passes only the geomean target and reaches 1.151 worst regret, so it is
not admitted to runtime selection. Learned ranking remains useful as a top-2 search hint;
the measured static table remains authoritative on V100.
