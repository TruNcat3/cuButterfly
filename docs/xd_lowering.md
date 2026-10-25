# xD lowering

cuButterfly exposes one multidimensional planning interface for rank-one
through rank-three descriptors. The interface separates the mathematical
contract from the physical lowering. It does not claim that every xD
configuration is compiled, measured, or globally optimized.

The bounded all-operator promotion run on 2026-10-06 is recorded in
[all-operator xD coverage](../results/all_operators_xd_20261006_gpu2_v2/coverage_report.md).
It exercises every public semantic operator through both the selector and an
explicit resident control, and pairs this with the resident correctness
suite. The legacy xor-zeta spelling is treated as the subset-zeta
compatibility alias documented by the public API.

## Execution modes

The C API selects the mode with `cubutterflySetExecutionMode`. The C++ API
uses `Transform::execution_mode` and `PlanOptions`.

- `CUBUTTERFLY_EXECUTION_AUTO` keeps the rank-one path and invokes the bounded
  xD selector for eligible rank-two and rank-three workloads.
- `CUBUTTERFLY_EXECUTION_XD` requests the xD selector for rank at least two.
- `CUBUTTERFLY_EXECUTION_COMPOSITE` forces the legacy per-axis composition.
  Its rank-three path uses the generic rank-N layout boundary.

For `AUTO` and `XD`, the selector constructs the executable candidates allowed
by the descriptor, device, and memory budget. It compares complete plans when
measurement is possible, including their boundary work. This is a finite
candidate selection procedure, not exhaustive xD-space calibration and not a
global-optimality guarantee.

The [2026-10-06 length-64 follow-up](../results/paper_profile_rc2/end_to_end/xd_local64_v18/README.md)
extends the executable local-core inventory to 64-point axes. The selector can
now combine the N64 ThreadShared recurrence core on strided axes with the
Block core on a contiguous axis and with batch tiling. On A100 80GB, the
64^3 cases selected this mixed lowering for B=1, 4 and 16 in FP32 and for
B=4 and 16 in FP64; the measured B=4 FP32 plan is 22.886 microseconds
versus 25.958 microseconds for cuFFT.
This is a bounded coverage repair, not a global-optimality claim.

The [xD v5 follow-up](../results/paper_profile_rc2/end_to_end/xd_layout_v5/README.md)
implements retained axis layouts and coalesced rank-three permutations. Its
measured larger 3D cases improve 1.49–2.97x over v3, while native-cuFFT parity
remains open. The report separates the final measurements from intermediate
development runs and records the remaining layout/core costs.

The [axis-core and graph-selection follow-up](../results/paper_profile_rc2/end_to_end/xd_axis_v6/README.md)
removes standalone permutations from supported tiled C2C plans. Sixteen
matched forward cases improve 1.231x geometrically over v5; their performance
relative to native cuFFT improves from 0.631x to 0.775x. Common 2D/3D parity
remains open, especially small resident and compact-real cases. The report
includes exact-map profiling, correctness and the bounded compiled inventory.

The [xD v9 cooperative-core record](../results/paper_profile_rc2/end_to_end/xd_register_v9/README.md)
adds a bounded register-axis and cooperative-pair inventory. It records the
compiled candidates, mapping contract, resource gating and correctness
evidence; it is not a claim that the finite inventory is a complete cross-axis
pipeline or a globally optimal xD search.

## Candidate families

The selector may compare the following lowerings when their contracts hold.

| Candidate | Contract and implementation |
| --- | --- |
| `composite` | Legacy per-axis composition. It remains the correctness fallback for supported generic layouts. |
| `flattened` | A strict tensor-product flattening for FWHT, subset zeta, superset zeta, and structured 2x2 butterflies. FFT and NTT are not flattened because their cross-axis twiddle or modular dependencies are not equivalent to concatenating one rank-one transform. |
| `resident-t128`, `resident-t256` | Resident shared-memory subgraphs for all six operators on small packed power-of-two workloads after the shared-capacity check. Intermediate data stays in the resident groups; there is no global intermediate buffer in this lowering. |
| `strided-axis-cufft` | A sequence of imported rank-one cuFFT cores using strided batches and the corresponding axis boundaries. It is not a vendor whole-xD transform. |
| `packed-axis-cufft` | A non-contiguous axis is permuted to axis-last and processed by one imported batched rank-one cuFFT core. Its physical layout is retained for the next axis; natural layout is restored at the output boundary. It is also not a vendor whole-xD transform. |
| `axis-tiled-cufftdx` | Optional local FFT cores perform coalesced axis access, shared-memory exchange, FFT and writeback inside one kernel. Supported axes preserve the physical tensor layout; other axes use the packed cuFFT fallback. |
| `axis-tiled-cufftdx-ept16` | The same axis lowering with a wider register allocation and a precision-dependent block batch. It competes with EPT 8 using complete-plan timing. |
| `pair-tiled-cufftdx`, `pair-tiled-cufftdx-ept16` | Two consecutive, physically trailing complex axes execute in one CTA over a complete dependency-closed plane. Thread FFT codelets exchange intermediate values through padded shared memory, avoiding one global intermediate and launch. EPT selects the core for any remaining unfused axes. |
| `axis-tiled-register` | A register-only four-lane cooperative core for non-contiguous axes of length 64 and 128. Contiguous or unsupported axes retain the existing core. The mapping records the actual core for each axis. |
| `axis-tiled-local`, `axis-tiled-local-cap16` | A 64/128/256-point local FFT built from Thread FFT8/16 codelets and SoA XOR shared exchange. N64 uses an 8x8 factorization; N128/N256 use the existing 16x(N/16) factorization. Each CTA processes 8 or 16 independent lines. FP64 length 256 uses 8 lines under either cap. Unsupported axes retain the existing Block/vendor axis fallback; admission requires an actual local axis. |
| `axis-tiled-local-recurrence` | The same local factors and shared exchange, with one phase seed per thread and successive complex multiplies replacing repeated phase-table loads. The search may choose this policy independently for each supported complex axis. |
| `pair-tiled-cooperative` | A four-lane cooperative pair core for two physically trailing axes, with the existing cuFFTDx axis core for any remaining axis. The pair supports the finite 32/64/128 shape inventory only when the compiled SM, registers and actual opt-in dynamic shared-memory budget permit it. The 32-point path uses four lanes over local Thread FFT8 codelets to reduce FP64 live ranges. |
| `pair-tiled-cooperative-register` | The cooperative pair core combined with `axis-tiled-register` for the remaining axis. It is a separate candidate from `pair-tiled-cooperative`; selecting the pair does not imply that every other axis uses the register core. |

The resident candidates are available only when the operator, storage,
direction, shape, stage resources, and shared-memory capacity satisfy their
contract. The flattened candidate is available only for the four exact
tensor-product operators listed above. A candidate name in a mapping records
the selected implementation, not a claim that the family is universally
faster.

The v16 local core changes the implementation of an individual complex axis,
not the multidimensional factorization. It reads the natural axis layout,
computes local factors `[16, N/16]`, exchanges the intermediate values on chip,
and writes the natural layout. It adds no global intermediate between those
local factors. Forward/inverse direction and final scaling share this path.
Compact-real transforms may use it on their complex axes; the real boundary
retains its existing implementation.

Per-logical-axis `axis_resources` can combine Block, cooperative-register and
`thread-shared` cores. Empty overrides preserve uniform-family behavior.
The bounded coordinate sweep tries 36 Block configurations, three register
lane widths and two local line caps under each of two twiddle policies per axis, skipping identical physical
mappings and unavailable requests. Every proposal is measured as a complete
plan, with the configurable promotion threshold (1% by default). Actual local axes record
`thread-shared-axis`, `local_factors`, and `io_exchange=soa-xor-local`; requested
EPT and exchange fields do not override that core's internal factorization.
The optional `local_twiddle_policy` resource field selects constant roots (`0`,
the default) or recurrence (`1`). Nonzero values require a `thread-shared`
core. Default fields are omitted from serialized mappings so old physical
replays remain exact. Actual recurrence axes report
`twiddle_policy=recurrence-root-64`, `recurrence-root-128` or
`recurrence-root-256`; this policy
propagates through batch tails and compact-real complex axes. The inventory
retains the original line caps and precision-dependent shared-memory limits.
This finite executable inventory is not exhaustive search or a new cost-model
calibration. Cache namespaces are versioned independently of old results.

The xD FFT axis candidates preserve the logical axis order. The strided
variant may launch a rank-one core for each required prefix because strided
batches cannot span the gaps between prefixes. The packed variant prebuilds
physical layouts along the chosen axis sequence and reuses one temporary
packed buffer. The last core writes into the buffer required by the final
layout boundary, so ping-pong parity does not require an extra device copy.
A typical rank-three C2C path has three permutations rather than four, with
the exact sequence recorded in `input_axis_order`, `permutation`, and
`output_permutation`. C2R restores the compact natural layout before its real
axis; its scratch copy preserves the user's spectrum. Out-of-place C2C uses
the first transform or permutation to preserve input in both directions.
`prefixlaunch > 256` can
omit the strided candidate from the default tuning budget; this is a budget
omission, not a theoretical restriction on the design space.

For xD FFT, the selector evaluates all permutations of the complex axes
(two orders for rank two, six for rank three), starting with the historical
descending order. Compact real FFT keeps the R2C/C2R axis fixed and only
reorders the preceding complex axes. The selected `axis_order` is recorded in
the physical mapping and included in the versioned cache key, so an old
mapping is not silently replayed after this search space changes. Retained
layout mappings use `layout_policy=retain-axis-layout-v1`; cache namespaces
are `xd-v8-core-fusion` and `xd-real-v8-core-fusion` for the earlier v8
inventory. The cooperative-core inventory uses `xd-v9-cooperative-cores` and
`xd-real-v9-cooperative-cores`; the namespace change prevents old mappings from
silently replaying across the new core choices. The withdrawn whole-vendor v4
experiment is not a production lowering.

The dedicated xD cache keys also include the raw `normalize_inverse` flag.
Even though it does not affect a forward transform's values, the explicit
mapping contract preserves it, so forward flag variants must not collide.

`CUBUTTERFLY_ENABLE_XD_CUFFTDX` enables the axis-tiled core independently of
the legacy rank-one cuFFTDx inventory. Its default follows
`CUBUTTERFLY_ENABLE_CUFFTDX`; headers come from `CUBUTTERFLY_MATHDX_ROOT`.
The current executable inventory covers FP32/FP64 power-of-two complex axes
of length 8 through 1024, in either direction. It compares requested EPT 8
and 16, clamped to the axis length. The block batch is
`min(32, min(256 * EPT, value_budget) / axis_length)`, where the value budget
is 4096 for FP32 and 2048 for FP64. These are implementation choices, not limits on
the mathematical design space. Availability checks the compiled SM against
the executing device. Unsupported axes and builds keep the existing vendor
axis lowerings. The selector measures complete plans before promoting this
candidate; enabling the feature does not guarantee it wins.

The legacy Thread-pair inventory covers axes 8, 16, 32 and 64 in FP32, and 8,
16 and 32 in FP64. Its FP64 limit comes from the available MathDx Thread
codelets, and it uses a fixed 48 KiB shared-memory guard. This is separate from
the cooperative pair inventory below. Each CTA loads a natural-layout plane,
evaluates rows in registers, exchanges through shared memory with pitch
`columns + 1`, evaluates columns, then stores in natural layout.
The two axes must be consecutive in the execution order and physically last
then penultimate. A rank-three plan can fuse this pair before or after its
remaining axis if that physical layout remains eligible. This is a two-axis
FFT, not a flattened rank-one FFT. Pair fusion currently applies to packed
out-of-place C2C plans, not compact real boundaries.

The v9 cooperative pair inventory is a separate finite set of lengths 32, 64
and 128, with four threads cooperating on every one-dimensional FFT. Its row and
column passes exchange a complete natural-layout plane through shared memory;
the shared budget is queried from the executing device rather than hard-coded
to 48 KiB. FP64 pair CTAs use 256 threads and fold a 128-line pass into two
rounds to stay within the SM80 register budget; FP32 may use the full 512-thread
geometry. On an A100, FP64 128x128 is unavailable because its padded plane
exceeds the device opt-in dynamic shared-memory budget. Rectangular FP64
64x128 and 128x64 remain resource-gated candidates. SM matching, occupancy,
register pressure and actual dynamic shared-memory capacity are checked before
the candidate is admitted.

The mapping records `fused_axes`, `axis_ept`, `pair_groups`, each group's
core, suffix, pair shape and block batch. Cooperative mappings additionally
record `axis_core`, `pair_core`, `cooperative_lanes`, and the actual
register-cooperative axis or pair core in each physical axis record. Pair
codelet resource shape is not controlled by `axis_ept`.
Supported axes do not emit standalone permutations. Final C2C normalization
and R2C output scaling are folded into a final tiled complex axis where
available; C2R still preserves its compact input and executes its real boundary.

## Compact real FFT

The real transform API is separate from the complex xD candidate families and
supports ranks one through eight for both FP32 and FP64 as of 0.9.3:

- R2C input uses scalar FP32 or FP64 values with the full logical shape.
- R2C output and C2R input use a compact half-spectrum whose last extent is
  `N_last / 2 + 1`, for both odd and even `N_last`.
- C2R output uses the full logical shape.
- The public contract is standard packed, out-of-place storage. The input and
  output element types are intentionally heterogeneous.
- `normalize_inverse` controls inverse normalization, and
  `fft_output_scale` applies the requested additional output scale.
- Compute precision must be native or match the scalar storage precision.

Compact real plans may reuse the v9 single-axis register core for eligible
complex axes, but they do not fuse a cooperative pair at the compact R2C/C2R
boundary. The pair candidates are therefore complex-only at that boundary.

For NTT, the existing modular inverse contract always applies `inverse_n`.
The floating-transform normalization flag does not select an unnormalized NTT;
this is canonicalized before candidate selection so every lowering agrees.

The implementation lowers each axis through rank-one cores and does not
silently delegate the entire multidimensional transform to a vendor whole-xD
FFT. Explicit non-packed strides and in-place real R2C/C2R plans report
`NOT_SUPPORTED`. For complex C2C and non-FFT operators, strided, in-place, and
embedding requests may use the generic composite fallback. Rank-three
arbitrary C2C can use the packed-axis vendor-core candidate; arbitrary rank-
three NTT is not covered by the current xD selector.

## Measurement, replay, and limits

Every instantiated compact-real lowering serializes the physical boundary
cost that it actually created.  The `physical.cost` record contains prefix,
permutation, scale, real-boundary, and C2R D2D-copy launch counts, the total
launch count, workspace bytes, and compact-boundary bytes.  Batch-tiled plans
record both the body cost and the tail cost.  The local cost fitter consumes
this record before using its legacy inference for old rows.  These fields are
execution metadata; they do not imply that the target-specific coefficients
have already been calibrated.

Research compilation is completed before timed execution. When the complete
candidate set fits the tuning budget, each candidate captures twenty complete
plan executions in a CUDA graph on a temporary nonblocking stream. Two graph
warmups precede five timed graph launches per trial; capture, instantiation and
warmup are excluded. The median of three trials determines each candidate's score.
If any candidate cannot be captured, the entire set uses the same event/host-
launch fallback protocol, with the reason recorded. The original caller stream
is restored, including when it is the legacy default stream.
A new candidate must improve the current best by more than one percent to
displace the retained baseline. Equal or noisy results keep the baseline. The
margin is configurable as a decimal fraction through
`CUBUTTERFLY_XD_PROMOTION_MARGIN`; it defaults to `0.01`, while `0.03` restores
the conservative behavior for a shared or noisy GPU during calibration.
The selected mapping records the effective margin in its selection reason.
This policy is a selection guard, not a performance guarantee.

The effective promotion factor is part of the xD selection-cache key, so
changing the margin automatically invalidates mappings produced by the other
policy.

`CUBUTTERFLY_XD_TUNE_MAX_BYTES` bounds the temporary allocations used by
complete-plan tuning. When unset, the bound is adaptive: it has a `512 MiB`
floor, a `2 GiB` cap, and uses at most one quarter of currently free VRAM;
the complete-plan fit check applies an additional free-memory guard. This
prevents a large-batch plan from silently skipping all candidate measurements
while keeping tuning bounded on a shared GPU. Set it explicitly to control the
bound, or to `0` to skip empirical promotion while retaining an executable
fallback. Compile and search costs are outside steady-state kernel timing.

The former compact rank-2 FP32 shortcut is now a parameterized policy rather
than an unconditional source-level preference. `CUBUTTERFLY_RANK2_FFT_POLICY`
defaults to `auto`, which consults the hardware registry and cost model;
`compact` enables the bounded compact-axis candidates, `measure` enables them
and triggers their runtime measurement even when the algorithm policy remains
`DEFAULT`, and `off` disables the shortcut. The shape bound is
controlled by `CUBUTTERFLY_RANK2_FFT_MAX_LENGTH` (a positive power of two,
default 128), and the candidate list by
`CUBUTTERFLY_RANK2_FFT_CANDIDATES`, a comma-separated subset of
`temporal-radix2`, `temporal-radix4`, and `cufft`. Compact candidates are
validated in the configured order; an unsupported candidate is recorded and
the next candidate is tried. These values are included in the axis-cache key,
so changing a policy cannot replay an incompatible result.
Local axis resources are admitted through the compiled capability query rather
than a hard-coded length list; adding a new local template therefore extends
the same selector and calibration path automatically.

The mapping JSON records the full semantic contract, target name and VRAM
capacity, SM information, runtime fingerprint, and selected physical lowering.
The selection reason reports candidate measurements and omissions; the
selection source distinguishes measurement, cache, and explicit replay.
Replays are rejected when the
contract or target fingerprint does not match. Diagnostic explicit mappings
such as `xd:resident-t128`, `xd:resident-t256`,
`xd:strided-axis-cufft`, `xd:packed-axis-cufft`, `xd:axis-tiled-cufftdx`,
`xd:axis-tiled-cufftdx-ept16`, `xd:pair-tiled-cufftdx`,
`xd:pair-tiled-cufftdx-ept16`, `xd:axis-tiled-register`,
`xd:pair-tiled-cooperative`, and `xd:pair-tiled-cooperative-register` can force a C2C or
non-FFT candidate for controlled tests. They do not bypass its correctness or
resource checks.

The static cost-model path uses the same physical mapping fields as the
runtime selector. It includes per-axis `prefix_loops`, materialized
permutation boundaries, workspace bytes, requested scale kernels, and the
compact-real vendor boundary. Rank-one R2C/C2R mappings therefore retain one
real-boundary launch even when `axes` is empty; rank-one inverse C2R also records
the legacy D2D copy. These are ranking features for cache misses and budgeted
selection. Complete-plan timing remains authoritative, and the feature weights
still require calibration on the target GPU.

For cooperative candidates, preparation sets the opt-in dynamic-shared-memory
attribute and verifies occupancy before execution or complete-plan timing.
These resource queries and any codelet initialization are outside steady-state
kernel timing.

There is currently no cross-axis streaming-overlap schedule or exhaustive
core/tile configuration enumeration in the xD selector. The existing rank-one planners
may still reuse their own internal search. These boundaries are deliberate and
should remain visible in reports rather than being described as a completed
global xD optimizer or a complete subblock cross-axis pipeline.

## Verification entry points

The public contract and explicit mappings are exercised by:

- `tests/check_xd.py`
- `scripts/benchmark_xd.py`
- `scripts/benchmark_xd_real.py`
- `cubutterfly_xd_fft_layout_tests` (also `--fused` when enabled)
- `cubutterfly_rank3_permute_tests`

The tests cover rank-two and rank-three complex paths, resident candidates,
compact real transforms, mapping replay, and the supported fallback contracts.

The lowering choices are compared with cuFFT, VkFFT, TurboFFT, tcFFT, and
cuFFTDx in [xD related work](xd_lowering_related_work.md). The current
implementation includes tiled rank-2/rank-3 boundaries, retained layouts across
axes, and an optional axis core that exchanges data inside shared memory. The
larger multi-tile stream and swizzled layout remain explicit future candidates.
