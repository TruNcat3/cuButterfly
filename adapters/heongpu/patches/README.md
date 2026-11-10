# CKKS v2 Stream Ordering Repair

This patch targets HEonGPU commit
`1928a14bb002c5d7f1cb5534eca9fa19b555b4f2`. It is a source-level diagnosis
artifact. The runner applies it to a generated source/header copy using
`--stream-ordering-fix`, identically for native and replacement providers.
The pinned checkout and original archive are preserved. The executable checks
that the requested patch mode matches its compiled mode, and execution
metadata records the staged source and patch hashes. The control observed that two fresh-input native
nonblocking-stream calls differ in all 327680 output words and have enormous
decoded errors, while the default-stream control produces identical output
words and maximum absolute error `9.88e-6`. Input words and metadata stayed
unchanged in both controls. The patched N16384 nonblocking run then produced
identical ciphertexts across both backends and both repetitions. A full Graph
run with the separate host-lifetime repair also passed original, changed and
restored-input validation. See the current evidence in
`results/rns_tile_io_20261021/README.md`.
The raw controls are under
`/mnt/v80-nas1/cubutterfly-ckks-n16384-20261021/probe/native-{nonblocking,default}.log`.

## Coverage

The patch propagates the existing execution stream through these runtime paths.

| Path | Changed producers | Consumer |
| --- | --- | --- |
| `scale_up_ckks` to `multiply_const_plain_ckks_v2` | Real and imaginary RNS constant uploads | Gaussian multiplication kernel |
| `add_constant_plain_ckks_v2` | Real and imaginary RNS constant uploads | Gaussian addition kernel |
| `multiply_matrix_v2` | PQ modulus and P-mod-Q uploads | Broadcast and matrix kernels |
| `evaluate_poly_from_polynomial_basis` | Coefficient allocation, complex constant encoding, FFT, conversion, NTT | Coefficient multiplication kernel |

The first three paths are reached by regular bootstrap v2. The polynomial basis
path is reached through `eval_mod`, recursive polynomial evaluation and its leaf
basis evaluation. The old three-argument constant encoder is retained as a
default-stream wrapper for preparation callers; runtime polynomial evaluation
uses the explicit four-argument overload.

The source audit also inspected v2 matrix temporary arrays, baby/giant-step index
uploads, host-key uploads and NTT configurations. These already use the local
execution stream. Constructor-time prime-location tables and bootstrap encoding
matrix preparation remain on the default stream. Their completion must precede
application execution; the current runner uses a device synchronization after
input preparation. This patch does not replace that preparation dependency.

## Graph Lifetime and Scope

This is a stream propagation patch, not complete CUDA Graph support. Runtime
RNS constants, PQ metadata, index tables and coefficient encoding still use
temporary host vectors. Captured H2D nodes may retain their addresses after the
vectors are destroyed. `GraphMemory` now retains pinned snapshots of fixed
host metadata and rejects changes between preparation and capture. It retains
the real H2D graph nodes for both providers. This separate repair passed a
dedicated GPU lifetime test and the N16384 full-call Graph validation. The
constant encoder also retains host work and temporary allocations in a runtime
path. Moving it onto the correct stream does not remove those costs.

Ciphertext copies use their source buffer's stream. Their readiness must be
checked at the public operation boundary. This patch does not audit every
operator in HEonGPU, RNG setup or memory-pool behavior, and does not establish
that reduced-N v2 parameters produce useful bootstrap precision.

## Applying and Validating

From a disposable copy of the pinned HEonGPU source tree:

```bash
patch --dry-run --batch --fuzz=0 -p1 -i /workspace/cuButterfly/adapters/heongpu/patches/ckks_v2_stream_ordering.patch
patch --batch --fuzz=0 -p1 -i /workspace/cuButterfly/adapters/heongpu/patches/ckks_v2_stream_ordering.patch
```

The patch uses standard unified diff with three context lines and `a/src` and
`b/src` paths. GNU patch with `--fuzz=0` was applied to a fresh temporary copy
of the two pinned files. Both patched files were byte-identical to the reviewed
candidate implementation. This check does not compile or execute the candidate.

The new member overload needs the matching staged `operator.cuh`; an unchanged
header cannot compile the patched implementation. The existing three-argument
symbol remains available. Recompiling the complete operator overlay against the
staged header can avoid rebuilding the upstream static archive because that
overlay already defines these methods and intercepts the operator object. All
translation units including the modified class declaration should use that same
header. The integration has been compiled at N16384, N32768 and N65536.

After integration, compare native and replacement with fresh inputs on default
and nonblocking streams. Record input contents and metadata before and after
each preparation call, decrypt native output before considering speedup, then
validate actual separate inverse requests. The nine N16384 request shapes have
passed original, changed and restored-input CPU exact tests for both providers.
Each larger workload still needs its own replacement and Graph evidence;
N16384 validation does not establish their results.
