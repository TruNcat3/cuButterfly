# TFHE complete-call workload

The pinned HEonGPU revision exposes eight public TFHE operations: `NAND`,
`AND`, `NOR`, `OR`, `XNOR`, `XOR`, `NOT`, and `MUX`. It does not expose
standalone TFHE `mod_up` or `mod_down`; modulus switching is private to the
bootstrapping implementation. CKKS `mod_up_from_q0` and modulus-drop paths
must therefore be reported as CKKS component experiments.

The runner measures the complete public call after setup and encryption. Its
timed region includes host dispatch, output allocation, pre-computation,
bootstrapping, sample extraction, key switching, adapter/layout work, and CUDA
synchronization. Key generation, encryption, decryption, and host transfers are
reported separately. Every output is decrypted and compared bit-for-bit with a
CPU Boolean reference.

Build and run the native reference on an A100 with the pinned HEonGPU archive:

```bash
CUDA_VISIBLE_DEVICES=1 python paper/experiments/run_heongpu_tfhe_e2e.py \
  --output results/e2e_current_20261018/tfhe_native_full.csv \
  --shapes 1,8,64,512 \
  --gates NAND,AND,NOR,OR,XNOR,XOR,NOT,MUX \
  --warmup 5 --repetitions 20
```

The complete native reference is saved at
`results/e2e_current_20261018/tfhe_native_full.csv`. It contains 44 gate and
circuit rows, all with `correct=1`. The runner also measures depth 2, 4, and 8
NAND/XOR chains at every requested batch shape. A cuButterfly replacement must
use the same ciphertexts, keys, streams, and shapes. Its full-call timing is
reported even when the replacement has extra launches or conversion kernels;
those costs are part of the result.

The source-level boundary is private HEonGPU `SmallForwardNTT` and
`SmallInverseNTT` inside the bootstrap kernels. Replacing them requires a
device callback or templated bootstrap variant with the same shared-memory
layout, roots, modular arithmetic, and stream contract. Until that replacement
passes the exact-bit check, the native file is a baseline and not a
cuButterfly speedup claim.

The prepared `cubutterfly-shared` backend replaces those device calls through
`adapters/heongpu/tfhe_bootstrap_overlay.cu`. The vendor checkout and archive
remain unchanged; the build uses a private archive copy with the bootstrap
object replaced by the overlay. Its shared-device template retains linear
1024-word shared storage, the supplied bit-reversed root tables, word64
GPU-NTT CT/GS arithmetic, inverse scaling, and the caller's stream. The
organization uses a CTA barrier after every stage. This absorbs the existing
arithmetic provider into a local template; it has not entered public RNS
planning or installation search. The existing RNS kernels cannot be called
directly: their entry points are global kernels and their public size range
starts at 4096.

Prepare both the complete-call runner and the exact device test without a GPU:

```bash
python paper/experiments/run_heongpu_tfhe_e2e.py \
  --backend cubutterfly-shared --execution graph --device-test --build-only \
  --output /tmp/tfhe-not-run.csv --jobs 4
```

After assigning an exclusive GPU, first run the same command without
`--build-only` (and with `--skip-build`) for six forward/inverse comparisons
against both native helpers and an independent CPU negacyclic DFT. The test
covers random values, zero/modulus boundaries, an impulse, and a nondefault
stream. The device test passed six exact cases on the A100 80GB GPU UUID
`GPU-0eae5f07-33de-321b-70a9-ffeae49309e6`; evidence is under
`results/e2e_alignment_20261019/tfhe/`. Its eager smoke file uses the earlier
dynamic-branch integration control (`native-helper`), not the unmodified
external-library baseline. All 16 smoke rows passed, including automatically
included depth-2/4/8 circuits. Those two-repetition results are not the final
performance matrix.

The current Graph implementation compiles unchanged native kernels and a
separate namespace with the template provider fixed at compile time. It
captures each complete public gate/circuit on one nondefault stream. Each
backend owns an independent reserving GraphMemory arena and output set.
Before instantiation, only the treatment's four bootstrap kernel identities
are replaced. Complete argument types must match at compilation; argument
bytes and launch dimensions are checked after replacement. Exact function
addresses identify nodes, unknown kernels are rejected, and expected counts
are checked per bootstrap: unique forward/inverse once each, regular
forward/inverse 511 times each. No device flag or per-stage branch remains.

Graph validation changes inputs through three additional bit patterns,
restores the original encrypted inputs, checks every graph output against the
CPU Boolean result, and checks that launching another graph preserves earlier
outputs. CUDA events time complete graph replay after preparation. Preparation,
capture, validation and arena reservation are outside steady-state timing.
Rows report `backend`, `execution`, graph-node counts and the validation mode.
All measured latencies, including slower treatment rows, are retained.

For a bounded first Graph check after GPU assignment:

```bash
CUDA_VISIBLE_DEVICES=GPU-0eae5f07-33de-321b-70a9-ffeae49309e6 \
python paper/experiments/run_heongpu_tfhe_e2e.py \
  --backend cubutterfly-shared --execution graph --skip-build \
  --shapes 1,8 --gates NAND --circuit-depths '' \
  --warmup 1 --repetitions 2 --graph-replays 1 \
  --output results/e2e_alignment_20261019/tfhe/paired_graph_smoke.csv
```

Omit `--circuit-depths ''` to include the circuits. The standalone `native`
backend supports `--execution graph` as the original archive control using the
same capture-memory shim; its default `eager` mode remains available. The
paired Graph control is named `native-original-kernel`, while standalone
Graph rows are `native-original-library`. Graph execution is compiled but
has not yet been run or validated; it does not establish a Graph speedup.

Standalone TFHE key switching in this revision uses torus32 decomposition,
table lookup and accumulation; it contains no NTT. CKKS key switching does
contain NTT and now has a [complete-call comparison](README_ckks.md), including
both key-switch methods. Those CKKS results are separate from this native-only
TFHE reference.
