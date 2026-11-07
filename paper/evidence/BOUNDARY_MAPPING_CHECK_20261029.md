# Boundary mapping check

_Bounded CPU verification of the APPT/Hermes bank and offset mapping, 2026-10-29._

---

## Scope

This check evaluates the mapping used by [`appt_layout.hpp`](../../include/cuntt/appt_layout.hpp:10) and APPT26 Algorithm 1 ([`samplepaper.tex`](../../../APPT26/samplepaper.tex:331)). Let

\[
M=2p=2^k,\quad N_{part}=2^s\ge M,\quad N=2^n\ge N_{part},
\]

with `M` dividing `N`. The checked formula is

\[
b(i)=(i\mathbin{\oplus}(i\mathbin{\gg}s))\mathbin{\&}(M-1),\qquad o(i)=i\mathbin{\gg}k.
\]

The physical array in the source is bank-major:
`physical = b(i) * (N/M) + o(i)`. This is a storage layout, with `o` a bank-local offset; it is not by itself a GPU shared-memory bank address.

## Checks

[`check_boundary_bank_mapping.py`](../tools/check_boundary_bank_mapping.py) enumerates `N` values for every `1 <= k <= 6`, `k <= s <= 10`, and `s <= n <= 14`, plus `(n,s,k)=(16,8,5)` (`N=65536`, `Npart=256`, `M=32`, the APPT26 maximum-N example).

For every case it checks:

1. `b(i)` and `o(i)` agree with the C++ and paper formulas.
2. Bank-major flattening is a bijection; the inverse recovers every logical index.
3. Each aligned group `i=aM+u`, `0 <= u < M`, has all `M` banks exactly once.
4. Each cross-stride group `i=c+vNpart` has distinct banks for every valid `v`. It is a complete `M`-request group when `N >= M*Npart`; otherwise the shorter valid prefix is checked.

## Result

```text
status: pass
parameter cases: 381
logical values checked: 1,527,934
aligned contiguous groups: 293,414
cross-stride groups: 72,484
complete cross-stride cases: 242
partial cross-stride cases: 139
```

The complete cross-stride property follows directly for `i=c+vNpart`: because `c<Npart`, `floor(i/Npart)=v`, so `b(i)=(c xor v) mod M`; for `v=0..M-1` this is a permutation of the `M` low-bit values. The aligned contiguous property follows because `floor((aM+u)/Npart)` is constant over the group when `Npart >= M`, while `u mod M` ranges over all bank values.

## Boundary of the claim

The proof concerns the stated scalar bank-index groups and the bank-major array representation. It does not establish that every GPU vector access, warp instruction, or arbitrary predecessor graph is conflict-free. Word width, bank-port behavior, broadcasts, transaction grouping, and the actual instruction schedule still require separate analysis.

For the Hermes parameters `(N,Npart,M)=(65536,256,32)`, directly interpreting the bank-major flattened address as an interleaved GPU address gives `physical mod M = 0` for all logical values in the first aligned group: all 32 requests appear in one interleaved bank. This is a representation mismatch, not a failure of the APPT bank mapping. The physical array must be consumed with its bank-major interpretation (or an explicit address translation).

The check therefore supports the narrow statement “the APPT formula is a bijective bank/offset arrangement with distinct banks for the two specified scalar access families.” It does not support a universal simultaneous-transfer injectivity guarantee or automatic packet refinement for outer-memory transactions.

## Reproduction

```bash
python paper/tools/check_boundary_bank_mapping.py
```
