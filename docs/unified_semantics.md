# Unified public semantics, 0.9.3

The public C, C++ and Python frontends now pass through the same CPU semantic
normalizer before plan creation. Its canonical representation is also used by
the complete-workload cache key and public mapping replay. Descriptor
validation and serialization do not query a GPU or construct a plan.

The preceding reviewed performance campaign was cancelled at the user's
request. `results/reviewed_campaign_20261007_r1/cancellation.json` preserves the
cancellation separately from its frozen manifest. Calibration, execution and
resume entry points reject that campaign. Existing measurements are retained;
no replacement performance population is defined by this repair.

## Public contract

| Dimension | Common representation |
| --- | --- |
| Operator and domain | Operator enum, NTT modulus/word width, structured matrices per axis |
| Numeric representation | Storage and compute types are separate; explicit mixed precision is retained |
| Shape and load | Extents vector and independent batch count |
| Mathematical direction | Forward/inverse and the operator's normalization convention |
| Logical input/output order | Independent natural/bit-reversed order per axis |
| Physical input/output layout | Independent axis strides and batch strides, in elements of each side's type |
| Placement | In-place or out-of-place, independent of logical order |
| Transform interpretation | FFT C2C/R2C/C2R and standard/zero-extended embedding |
| Execution policy | Separate selection section with execution mode, policy and optional mapping |

For bit reversal, logical coordinate `i` is stored at `reverse_bits(i, log2(N))`
along that axis. Bit reversal requires a power-of-two extent on that side.
This is not an instruction to run DIT, DIF or GS. Those remain implementation
choices. In a real FFT the last half-spectrum extent is `N/2+1`; its ordering
and strides refer to complex elements, while the real side counts scalars.

Missing stride arrays independently select packed axis strides. Nonzero batch
strides are preserved even when both axis arrays are omitted. A shape setter
preserves other supplied fields; inconsistent per-axis lengths then fail
validation, rather than silently resetting the request. Known overlapping
layouts are rejected. Large sparse layouts whose non-overlap cannot be proved
within the bounded validator remain expressible and report a capability gap.

Canonicalization resolves defaults and removes inactive distinctions: forward
normalization is false, inverse NTT includes inverse-N, and zeta/structured
inverse semantics do not use the floating normalization flag. Native compute
resolves to the storage precision except for low precision, where native and
FP32 accumulation remain distinct. It never silently replaces an active
requested order, stride or computation precision.

The C API provides `cubutterflyDescriptorValidate`, `DescriptorGetJson`,
`DescriptorSetJson` and `DescriptorGetCapabilities`. JSON import is atomic and
rejects unknown fields, incorrect types and lossy integer conversion. The
calling thread can retrieve detailed failures with
`cubutterflyGetLastErrorString`. `PlanGetDescriptorJson` exposes what was
actually planned. Complete public mapping JSON binds the same semantic
contract and rejects replay with changed semantics.

The descriptor envelope is `cubutterfly-descriptor-v1`, containing a
`cubutterfly-semantic-v1` semantic record and a separate selection record.
EXPLICIT policy requires a nonempty mapping or algorithm; DEFAULT/MEASURE
require no explicit algorithm. Contradictory selections are rejected instead
of silently ignoring the supplied mapping.
Legacy NTT scalar layout setters remain compatibility adapters. Their
rank-one bit-reversed spelling normalizes to the common per-axis order.
APPT static layout retains its explicit legacy contract and kernel limits.

## C++ and Python use

```cpp
#include <cubutterfly/plan.hpp>

cubutterfly::Transform t;
t.op = CUBUTTERFLY_OPERATOR_FWHT;
t.storage = CUBUTTERFLY_DATA_FP32;
t.extents = {4, 8};
t.batch = 2;
t.input_batch_stride = 48;
t.output_batch_stride = 64;
t.input_orders = {CUBUTTERFLY_ORDER_BIT_REVERSED, CUBUTTERFLY_ORDER_NATURAL};
cubutterfly::Descriptor description(t);  // CPU only
description.validate();
auto canonical = description.json();
auto capability = description.capabilities_json();
// With a GPU context: cubutterfly::Plan plan(context, description);
```

Install the thin Python package with `python -m pip install ./python`. It has
no mandatory PyTorch dependency and does not import CUDA at package import.
Use an explicit library path or `CUBUTTERFLY_LIBRARY` to select the build.

```python
from cubutterfly import Library, Transform, Descriptor, Operator, DataType

library = Library("build-e13-cufftdx/libcuntt.so")
transform = Transform(op=Operator.FWHT, storage=DataType.FP32,
                      extents=[4, 8], batch=2, input_batch_stride=48,
                      output_batch_stride=64, input_orders=[1, 0])
with Descriptor(library, transform) as description:
    description.validate()
    print(description.json())
    print(description.capabilities_json())
```

`python -m cubutterfly descriptor.json --library /path/to/libcuntt.so` performs
the same CPU validation/export. Python `Plan` accepts a descriptor or transform,
an optional construction-time stream, external workspace, and device pointers
or objects with `data_ptr()`. Its execution is asynchronous; allocation and
stream lifetimes remain the caller's responsibility. It does not time or hide
compilation. `tests/check_xd.py` now constructs descriptions through this public
package rather than maintaining another setter translation.

## Execution and evidence boundary

A common gather/natural-child/scatter lowering implements per-axis order and
independent layout boundaries for existing arithmetic kernels. It also supports
in-place requests using workspace, including real FFT sides with different
element widths. The native rank-one NTT bit-reversed output path remains
eligible without forcing an extra permutation. The generic per-axis composite
schedule now accepts ranks through eight. Descriptors can express higher ranks;
the capability query reports the current execution limit separately.

The capability status `eligible` means that static structural/lowering checks
passed. It does not certify arithmetic roots, target resources, executable
specializations or performance. Version 0.9.3 adds UINT64 zeta, explicit
FP32-storage/FP64-compute adaptation, same-precision low real FFT adaptation,
rank-eight compact real FFT and arbitrary-length rank-N axis composition.
Low real FFT currently requires power-of-two axes of length at least two.
Other absent arithmetic cores and requests beyond the rank-eight execution
limit still report `missing-lowering`. They are not relabeled mathematical
impossibilities. See [the bounded support milestone](bounded_support_20261008.md).

The common boundary is a correctness implementation, not evidence of a fused
boundary or optimal performance. Its mapping records both conversions and the
child. Complete-call execution includes those conversions; child-only tuning
times do not include them and are explicitly identified as such. The existing
rank-one primitive registry is still a projection of core workloads, not a
substitute for the full multidimensional semantic contract or fresh calibration.

Numeric mappings use `kind=numeric-adapter` and record their child, workspace,
and conversion-inclusive execution scope. FP32 storage with FP64 compute casts
into FP64 working buffers, performs the requested transform, then casts back.
FP16/BF16 real FFT uses its matching complex storage and compute core, expanding
or extracting the compact Hermitian spectrum explicitly. It is a full-spectrum
adapted implementation, not a claim of native compact low-precision performance.

The repair is checked through CPU descriptor/ABI tests, raw ordered-boundary
tests and public-interface value/replay tests. These are implementation
regressions, not a new comprehensive performance campaign. The local evidence
record is `results/semantic_repair_20261007/README.md`.
