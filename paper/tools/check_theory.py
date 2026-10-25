#!/usr/bin/env python3
"""Small CPU witnesses for paper equations, not CUDA implementation tests."""
import argparse
import cmath
import json
import math
from pathlib import Path


def reverse_bits(i, log_n):
    return int(f"{i:0{log_n}b}"[::-1], 2)


def mod_inverse(value, modulus):
    """Return a modular inverse on Python versions before pow(..., -1, m)."""
    value %= modulus
    old_r, r = value, modulus
    old_s, s = 1, 0
    while r:
        quotient = old_r // r
        old_r, r = r, old_r - quotient * r
        old_s, s = s, old_s - quotient * s
    if old_r != 1:
        raise ValueError("value is not invertible modulo modulus")
    return old_s % modulus


def permute(values):
    log_n = len(values).bit_length() - 1
    return [values[reverse_bits(i, log_n)] for i in range(len(values))]


def transform(values, inverse=False, gs=False, modulus=None):
    n = len(values)
    log_n = n.bit_length() - 1
    if modulus:
        root = pow(5, (modulus - 1) // n, modulus)
        if inverse:
            root = mod_inverse(root, modulus)
        reduce = lambda x: x % modulus
    else:
        root = cmath.exp((1 if inverse else -1) * 2j * math.pi / n)
        reduce = lambda x: x
    data = list(values) if gs else permute(values)
    lengths = [1 << s for s in range(1, log_n + 1)]
    if gs:
        lengths.reverse()
    for length in lengths:
        half = length // 2
        for start in range(0, n, length):
            for j in range(half):
                w = pow(root, j * (n // length), modulus) if modulus else root ** (j * (n // length))
                a, b = data[start+j], data[start+j+half]
                if gs:
                    data[start+j], data[start+j+half] = reduce(a+b), reduce((a-b)*w)
                else:
                    data[start+j], data[start+j+half] = reduce(a+w*b), reduce(a-w*b)
    if gs:
        data = permute(data)
    if inverse:
        scale = mod_inverse(n, modulus) if modulus else 1 / n
        data = [reduce(v * scale) for v in data]
    return data


def direct(values, modulus=None):
    n = len(values)
    if modulus:
        root = pow(5, (modulus - 1) // n, modulus)
        return [sum(x * pow(root, j*k, modulus) for j, x in enumerate(values)) % modulus for k in range(n)]
    return [sum(x * cmath.exp(-2j*math.pi*j*k/n) for j, x in enumerate(values)) for k in range(n)]


def zeta(values, superset=False, inverse=False):
    data = list(values)
    for bit in range(len(data).bit_length() - 1):
        for i in range(len(data)):
            if bool(i & (1 << bit)) != superset:
                data[i] = (data[i] + (-1 if inverse else 1) * data[i ^ (1 << bit)]) % (1 << 32)
    return data


def run():
    counts = dict(bit_reversal_edges=0, floating_transforms=0, exact_ntt_transforms=0,
                  zeta_roundtrips=0, bank_layouts=0, local_inverse_identities=0)
    for log_n in range(1, 9):
        n = 1 << log_n
        for s in range(log_n):
            for i in range(n):
                assert reverse_bits(i ^ (1 << s), log_n) == reverse_bits(i, log_n) ^ (1 << (log_n-1-s))
                counts['bit_reversal_edges'] += 1
    max_error = 0.0
    for n in (2, 4, 8, 16, 32, 64):
        values = [complex((i*7) % 13 - 6, (i*3) % 5 - 2) for i in range(n)]
        ref = direct(values)
        for gs in (False, True):
            actual = transform(values, gs=gs)
            error = max(abs(a-b) for a, b in zip(actual, ref))
            max_error = max(max_error, error)
            assert error < 1e-9
            roundtrip = transform(actual, inverse=True, gs=gs)
            assert max(abs(a-b) for a, b in zip(roundtrip, values)) < 1e-9
            counts['floating_transforms'] += 2
        integers = [(i*7 + 0xfffffff9) % (1 << 32) for i in range(n)]
        for superset in (False, True):
            assert zeta(zeta(integers, superset), superset, inverse=True) == integers
            counts['zeta_roundtrips'] += 1
    # 5 is a primitive root modulo 97; all tested lengths divide 96.
    for n in (2, 4, 8, 16, 32):
        values = [(i*7 + 11) % 97 for i in range(n)]
        ref = direct(values, 97)
        for gs in (False, True):
            actual = transform(values, gs=gs, modulus=97)
            assert actual == ref
            assert transform(actual, inverse=True, gs=gs, modulus=97) == values
            counts['exact_ntt_transforms'] += 2
    for w in range(1, 97):
        # (1/2) D(w^-1) C(w) = identity in F_97.
        c = ((1, w), (1, -w))
        invw, inv2 = mod_inverse(w, 97), mod_inverse(2, 97)
        d = ((1, 1), (invw, -invw))
        product = [[sum(d[i][k]*c[k][j] for k in (0, 1))*inv2 % 97 for j in (0, 1)] for i in (0, 1)]
        assert product == [[1, 0], [0, 1]]
        counts['local_inverse_identities'] += 1
    for banks in (2, 4, 8, 16, 32):
        addresses = [banks*r + (c ^ r) for r in range(banks) for c in range(banks)]
        assert sorted(addresses) == list(range(banks*banks))
        for r in range(banks):
            assert len({(banks*r + (c ^ r)) % banks for c in range(banks)}) == banks
        for c in range(banks):
            assert len({(banks*r + (c ^ r)) % banks for r in range(banks)}) == banks
        # Unswizzled columns are a counterexample to conflict-free columns.
        assert len({(banks*r + 1) % banks for r in range(banks)}) == 1
        counts['bank_layouts'] += 1
    sample = [complex(i*i, i % 3) for i in range(8)]
    natural = transform(sample, gs=True)
    assert max(abs(a-b) for a, b in zip(natural, permute(natural))) > 1e-3
    return dict(status='passed', scope='CPU mathematical examples; not CUDA or performance validation',
                counts=counts, max_fft_absolute_error=max_error,
                negative_controls=['unpermuted GS output differs', 'unswizzled columns collide'],
                bank_assumptions='P lanes, P single-word banks, one word per lane, full row or column',
                unproved_by_enumeration=['general graph partition correctness', 'all floating rounding contracts',
                                        'real multiword GPU transactions', 'GPU overlap or global optimality'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = run()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
