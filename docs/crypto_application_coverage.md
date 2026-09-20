# Cryptographic butterfly application coverage

> Note: Paths marked `local artifact` are local experiment records and are not included in this source release.

Storage width is not the numeric contract. A cryptographic NTT is exact
arithmetic over a specified field or ring, with a specified evaluation domain,
ordering and representation. A 64-bit container does not imply support for all
64-bit primes. A 256-bit field is not a batch of four independent 64-bit fields.

## Workload axes

Record these dimensions independently:

- Transform length `N` and independent polynomial batch `B`.
- Prime `p`, its bit length, arithmetic representation and machine word width.
- Cyclic, negacyclic or coset evaluation, forward/inverse direction, roots,
  normalization and physical input/output order.
- RNS primes `(p_0, ..., p_{L-1})` and channel count `L`. The combined modulus
  is their product; its bit length is distinct from the sum of container widths.
- For future cores: number of limbs per field element and extension degree.
  Neither can be represented by silently increasing `B` or `L`.

The finite application study (local artifact: `../results/crypto_application_20260920/README.md`; not included in this source release)
extends the existing batch/precision experiment. Its manifests name the
population before measurement, retain missing implementations, and report
throughput per polynomial and per residue transform separately. Non-power-of-two
batches are intentional tail cases; transform legality still follows the root
requirements. Input values include modular boundaries as well as random values.

## Application evidence and semantics

| Application family | Relevant contract | Coverage interpretation |
| --- | --- | --- |
| Lattice signatures | Dilithium uses `N=256`, `p=8380417` (23 bits) | A natural-order, canonical-residue negacyclic transform exercises its ring; it is not byte-for-byte compatibility with the reference Montgomery/bit-reversed ABI. |
| STARK-oriented fields | BabyBear `2013265921` and KoalaBear `2130706433`, both 31 bits | Cyclic and multiplicative-coset transforms, with independent `N` and `B`. |
| Homomorphic encryption | Multiple NTT-friendly primes, often of different bit lengths | Negacyclic RNS transform composition; channel count is independent of batch. CRT reconstruction, rescaling and key operations are outside transform timing. |
| Kyber | `N=256`, `p=3329` (12 bits) | Its incomplete NTT leaves quadratic blocks. A complete cyclic transform with this modulus is not a substitute. |
| Goldilocks | `2^64-2^32+1` | Outside the current `p<2^63` arithmetic contract. |
| Pairing-curve scalar fields | BN254 scalar field (254 bits), BLS12-381 scalar field (255 bits) | Requires multi-limb field arithmetic. BLS12-381's 381-bit base field is a different field. |
| Extension fields | For example, four BabyBear components per extension-field element | Requires extension-field multiplication and explicit element layout. |

Primary sources: [Dilithium parameters](https://github.com/pq-crystals/dilithium/blob/master/ref/params.h),
[Kyber NTT](https://github.com/pq-crystals/kyber/blob/master/ref/ntt.c),
[BabyBear](https://github.com/Plonky3/Plonky3/blob/main/baby-bear/src/baby_bear.rs),
[KoalaBear](https://github.com/Plonky3/Plonky3/blob/main/koala-bear/src/koala_bear.rs),
[SEAL CKKS example](https://github.com/microsoft/SEAL/blob/main/native/examples/5_ckks_basics.cpp),
[Goldilocks](https://github.com/Plonky3/Plonky3/blob/main/goldilocks/src/goldilocks.rs),
[BN254 scalar field](https://github.com/arkworks-rs/curves/blob/master/bn254/src/fields/fr.rs),
[BLS12-381 scalar field](https://github.com/arkworks-rs/curves/blob/master/bls12_381/src/fields/fr.rs).

Width-controlled primes in the study are deterministic synthetic NTT-friendly
primes. They test arithmetic/resource scaling; they are not cryptographic
security parameter recommendations or substitutes for named protocol constants.

## Composition over the existing public plans

For the existing cyclic transform with primitive root `omega`, a coset transform
evaluates at `g * omega^k`: multiply coefficient `a_j` by `g^j` on the GPU,
then execute the public NTT plan. Its inverse applies the normalized inverse
NTT and multiplies coefficient `j` by `g^(-j)`.
The finite matrix chooses nontrivial cosets (`g^N != 1`); general nonzero `g`
also permits a subgroup permutation, including the `g=1` cyclic special case.

A negacyclic transform is the special case `g=psi`, where `psi^2=omega` and
`psi^N=-1`. This requires `2N | (p-1)`. The selected `psi` must square to the
actual root used by the underlying plan; merely finding some primitive `2N`-th
root and passing a round-trip test would not establish ordering compatibility.

The isolated composition benchmark uses these GPU pre/post operations and one
public plan per RNS modulus. It initially executes channels sequentially on one
stream. It exposes one mapping per channel and can replay exact-contract cyclic
search results. This separates reusable butterfly scheduling from application
semantics without claiming a joint optimum for cross-channel scheduling.

Kernel timing includes every GPU twist, NTT and RNS channel in the composed
operation. Allocation, plan construction, JIT compilation, table preparation,
host transfers and exact CPU checking are outside it. These timing boundaries
must match before comparison with another library. The current composed study
has no matching external adapter, so it cannot produce an external-library win.

## Support and evidence boundaries

The existing library implements single-word cyclic NTT with `p<2^63`;
word32 additionally requires `p<2^31`. Barrett has stricter bounds (30/62 bits)
and a restricted backend. The new benchmark adds explicit application
composition; it does not silently extend the installed planner's semantic ABI.

Goldilocks, multi-limb fields, extension fields and Kyber's incomplete NTT remain
implementation gaps, not missing calibration samples. Supporting them requires
the corresponding arithmetic/transform lowering, independent exact-reference
checks, and measurements of the changed compute and resource costs. The shared
butterfly dependency structure motivates reuse of hierarchical dataflow; it
does not establish that an unimplemented core is fast or universally optimal.

The queued population and CPU preparation are not GPU correctness or performance
evidence. Reports retain partial and unsupported states until their respective
checks complete. Existing running study inputs and installed binaries remain
frozen so old and new results have unambiguous provenance.
