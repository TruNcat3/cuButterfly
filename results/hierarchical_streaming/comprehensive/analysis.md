# Hierarchical Streaming Batch Comparison

CUDA-event times use identical V100 binaries and measurement settings. Rows are medians of independent process trials; speedup above one is better than the v0.6 barrier implementation.

## uint32

| batch | barrier ms | resident 10+10 ms/speedup | generic 7+7+6 ms/speedup | resident 7+7+6 ms/speedup |
|---:|---:|:--|:--|:--|
| 1 | 0.080179 | 0.129372 / 0.620x | 0.371343 / 0.216x | 0.428339 / 0.187x |
| 4 | 0.271708 | 0.347505 / 0.782x | 1.501737 / 0.181x | 2.627093 / 0.103x |
| 8 | 0.501760 | 0.546652 / 0.918x | 3.706859 / 0.135x | 5.996687 / 0.084x |
| 16 | 1.058591 | 1.020949 / 1.037x | 10.324991 / 0.103x | 13.564334 / 0.078x |
| 32 | 2.852577 | 2.398044 / 1.190x | 25.470772 / 0.112x | 29.840281 / 0.096x |

Geomean speedup versus barrier:

- `resident1010`: `0.887x`
- `generic776`: `0.143x`
- `resident776`: `0.104x`

Maximum trial spread: `16.99%`.

## uint64

| batch | barrier ms | resident 10+10 ms/speedup | generic 7+7+6 ms/speedup | resident 7+7+6 ms/speedup |
|---:|---:|:--|:--|:--|
| 1 | 0.142316 | 0.203489 / 0.699x | 0.502252 / 0.283x | 0.644895 / 0.221x |
| 4 | 0.503357 | 0.555151 / 0.907x | 1.477263 / 0.341x | 3.363328 / 0.150x |
| 8 | 0.963727 | 0.972923 / 0.991x | 2.871419 / 0.336x | 7.630172 / 0.126x |
| 16 | 1.895014 | 1.840476 / 1.030x | 6.695895 / 0.283x | 16.561808 / 0.114x |
| 32 | 3.712512 | 3.623834 / 1.024x | 24.247295 / 0.153x | 35.610031 / 0.104x |

Geomean speedup versus barrier:

- `resident1010`: `0.921x`
- `generic776`: `0.269x`
- `resident776`: `0.138x`

Maximum trial spread: `13.02%`.

