# Low-precision local templates

Local arithmetic templates have precision-specific capabilities. They share the
same organization framework: partitioning, tile grouping, boundary layouts,
capacity constraints and selection remain composable mapping choices. Replacing
the storage type alone does not establish that a provider implements the desired
arithmetic or packing.

## Current arithmetic and provider coverage

| Storage / arithmetic | Scalar FFT template | cuFFTDx local/factor adapter in this repository | Packing / rounding |
| --- | --- | --- | --- |
| FP16 / native | `LowFftOperator<Fp16,true>` | Supported subject to size, geometry and compiled resources; normalized inverse currently excluded | Scalar rounds primitives to FP16; cuFFTDx half2 packs two independent transforms |
| BF16 / native | `LowFftOperator<Bf16,true>` | Not implemented | Scalar uses explicit BF16 rounded primitives; no FP32-provider substitution |
| FP16 / FP32 arithmetic | `LowFftOperator<Fp16,false>` | Not implemented | Complex product computed in FP32 then narrowed; butterfly outputs retain FP16 storage |
| BF16 / FP32 arithmetic | `LowFftOperator<Bf16,false>` | Not implemented | Complex product computed in FP32 then narrowed; butterfly outputs retain BF16 storage |
| FP32 / native | `FftOperator<Complex32,float>` | Supported subject to provider and resource checks | FP32 values; distinct register/native codelet options |
| FP64 / native | `FftOperator<Complex64,double>` | Supported subject to provider and resource checks | FP64 values; distinct resource demand |

“Supported” identifies a repository adapter, not every configuration being
compiled or resource-feasible. These are repository coverage statements, not
claims that upstream cuFFTDx lacks other capabilities. Native scalar and provider
arithmetic may have different internal rounding; provider identity and measured
output errors must remain visible in comparisons.

The low-storage FP32-arithmetic rows do not mean the entire transform retains
FP32 intermediates. The scalar complex product narrows into its low-precision
value type before the butterfly additions. The separate `fp16-fp32` internal
value mode is also not implicitly an FP32 cuFFTDx provider: an adapter must
explicitly implement the conversion and numerical contract before it is eligible.

## Capability checks and model identity

[`cufftdx_arithmetic_precision_bits`](../src/fft_local_core_contract.hpp) is
consumed by candidate provider checks in [plan_registry.cpp](../src/plan_registry.cpp)
and JIT provider validation in [jit_module.cpp](../src/jit_module.cpp). It returns
no provider arithmetic for BF16, mixed modes, unsupported accumulation, other
operators or the unsupported normalized FP16 inverse. A missing local adapter
does not invalidate the mathematical transform; scalar and other compatible
templates remain candidates.

This check precedes SM/size/EPT capability lookup. Physical half2 packing remains
in `fp16_local_block_shape`; launch geometry, shared memory, registers and actual
provider resource validation remain separate checks.

Precision and accumulation already participate in registry semantic matching and
the [stage cost model](../scripts/stage_cost_model.py). The classification does
not introduce a separate search algorithm or pool BF16 service samples with
FP32/FP64. Adding a template requires explicit arithmetic, support, generation,
resource and numerical-validation contracts through these existing interfaces.

One separate recorded gap remains: scalar `LowFftOperator` stores a
`complex_multiply` choice without consuming Gauss3. Current FourMul measurements
are unaffected; this document does not advertise a working Gauss3 low template.

## Independent tile columns

The generic shared-iterative lowering now consumes `data_tiles_per_cta` for
FFT, FWHT, zeta and structured butterfly requests. Each CTA interleaves several
independent graph tiles across lanes. This changes organization, not arithmetic;
the selected precision-specific trait retains its rounding behavior. It is
separate from `batch_tile_count`, which serializes external batches or sets the
overlap chunk size.

For local extent L, C independent columns and value width W, the two-buffer
shared requirement is `2 * L * C * W`. Columns must be powers of two and fit the
independent tiles of every physical group. The enumerator, plan projection,
module resource query, stage model and CPU compilation preparation consume this
axis. Hardware capacity bounds the combinations; larger column counts are not
assumed faster.

This combination currently uses the on-demand shared-iterative module with
direct boundaries. Resident local partitions, explicit tiled-transpose edges
and precompiled-only execution reject multi-column requests rather than
ignoring them. The separate public NTT mapping has not acquired this particular
axis; its existing core-specific grouping mechanisms remain distinct.

The bounded A100 experiment
records the implementation checks, actual BF16 gains and outstanding regression
status. It does not claim a new BF16 cuFFTDx core has been implemented.
