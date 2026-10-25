# Explicit stage FFT subgraph experiment

This is a route-B prototype. It is deliberately outside the public selector
and does not change the default FFT candidates.

The implementation in `src/xd_stockham_pair_fft.cu` assigns one A100 CTA to
each contiguous 32 x 32 FP32 C2C plane. It loads the plane into two shared
buffers, performs an in-shared bit reversal, and executes five radix-2 stages.
Each stage writes both butterfly outputs to the other buffer and synchronizes
before the next stage. The same procedure is then applied to the columns. The
whole CTA operation is captured and replayed as one CUDA Graph instance by
`apps/stockham_subgraph_bench.cu`.

The initial direct Stockham output-index formula was rejected after a 32-point
unit check: it produced a relative error of 1.36 for 64 planes. The current
prototype uses explicit bit reversal followed by radix-2 ping-pong stages. It
is therefore stage-aware and layout-controllable, while the name Stockham
refers to the intended generated-stage organization rather than claiming the
rejected index formula.

## A100 GPU2 result

Build and run:

```bash
cmake -S . -B build-e13-cufftdx
cmake --build build-e13-cufftdx --target stockham_subgraph_bench -j 8
CUDA_VISIBLE_DEVICES=2 ./build-e13-cufftdx/stockham_subgraph_bench 64 20 100
```

The run reported:

| path | 64 planes, Graph replay |
| --- | ---: |
| explicit radix-2 prototype | 0.012329 ms |
| existing cuFFTDx pair subgraph | 0.005540 ms |
| cuFFT | 0.006738 ms |

The prototype's maximum absolute error was `3.33e-5` and its maximum relative
error against cuFFT was `3.70e-7`. The pair subgraph error was `2.70e-5`.
Additional runs were correct for one plane (`2.78e-7` relative error) and 256
planes (`3.62e-7`). The prototype was `1.83x` slower than cuFFT at 64 planes,
`1.94x` slower at one plane, and `2.70x` slower at 256 planes. The existing
cuFFTDx pair path was faster than cuFFT on the same 64-plane Graph protocol.

The gap is expected for this first generated-stage implementation: every
radix-2 stage performs full CTA barriers and repeatedly evaluates scalar
twiddles. The experiment establishes correctness and a real Graph-resident
stage boundary; it does not establish a competitive FFT core. The next
optimization should replace scalar twiddle evaluation with a generated table
or recurrence and fuse the row-to-column boundary, while preserving the
stage-level output layout.
