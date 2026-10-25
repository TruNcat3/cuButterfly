# Application-level acceleration boundary

This note separates application results that are positive, near parity, and
negative. It is a result ledger, not a claim that every butterfly-heavy
application is already accelerated.

| Application and contract | cuButterfly result | Classification | Evidence-based cause |
| --- | ---: | --- | --- |
| FNO Darcy, 64², B=4, eager | 0.9554x vs Torch/cuFFT | negative | The isolated FFT is faster, but Python/FFI submission, allocation, and launch boundaries dominate the complete call. |
| FNO Darcy, 128², B=4, eager | 0.9574x vs Torch/cuFFT | negative | Same boundary effect; the butterfly fraction is not large enough to amortize the host-side work. |
| FNO Darcy, 64² / 128², B=4, Graph | 1.0299x / 1.0564x vs Torch/cuFFT | positive | CUDA Graph removes most repeated submission and launch overhead. |
| 2D Poisson, 2048², B=4, eager | 0.536x vs Torch/cuFFT | negative | Complete-call eager overhead and multi-kernel submission dominate. |
| 2D Poisson, 2048², B=4, Graph | 1.159x vs Torch/cuFFT | positive | Graph execution exposes the faster adaptive fused-axis plan. |
| QuaRot FWHT, short N=32/64, B=1/4 | 1.014–1.016x / 1.008–1.013x vs the original path | near parity | The FWHT request is short and decode/non-butterfly work dominates. The result is an optimization opportunity, not evidence that the dataflow is invalid. |
| Image filter, matched padding, B=16 | 1.067x vs CuPy and 1.090x vs Torch/cuFFT on the 1024 row | positive | The butterfly portion is large enough for the lower-level gain to survive the complete call. |
| CKKS Graph, B=1, N=16384/32768/65536 | 1.065x / 1.067x / 1.104x vs corrected HEonGPU | positive | The NTT-heavy graph keeps the adapter and non-NTT work inside the same measured call and still benefits from the selected path. |

The two genuinely negative rows are therefore **execution-mode and integration
effects**, not a failed operator result. The short QuaRot rows are a different
case: their butterfly share is too small for a large end-to-end gain. A missing
application or baseline remains a gap and is not converted into a negative or a
positive result by substituting a smaller size, another precision, or a CPU
implementation.

## Why the corrected baseline is used

The HEonGPU comparison had a reproducibility bug in its nonblocking stream path:
producer operations were submitted on the default stream while consumers used
the requested stream. Repeated runs produced changed outputs and errors as high
as about `1e121`; the default-stream control was exact. The graph path also had
to retain host snapshots for fixed H2D nodes. Both repairs are applied to the
baseline and cuButterfly runs symmetrically. They establish a valid comparison
contract; they are not optimizations performed for the baseline.

## Required decomposition for every future application row

The unified benchmark records the operator-only latency and complete-call
latency separately, together with butterfly fraction, conversion/layout time,
non-butterfly time, launch/Graph mode, and correctness status. This makes the
following diagnosis explicit:

\[
T_{app}=T_{butterfly}+T_{other}+T_{conversion}+T_{submission}.
\]

An operator speedup can disappear when `T_other`, `T_conversion`, or
`T_submission` dominates. The benchmark must report that decomposition instead
of attributing the result to the butterfly kernel alone.
