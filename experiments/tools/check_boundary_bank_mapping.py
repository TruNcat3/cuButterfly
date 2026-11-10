#!/usr/bin/env python3
"""Bounded CPU checks of the APPT bank/offset formula; no GPU dependencies."""

import json


def check_case(n_log, part_log, bank_log):
    n, part, banks = 1 << n_log, 1 << part_log, 1 << bank_log
    assert n >= part >= banks >= 2 and n % banks == 0
    span = n // banks
    seen = bytearray(n)
    for i in range(n):
        bank = (i ^ (i >> part_log)) & (banks - 1)
        offset = i >> bank_log
        assert bank == (i ^ (i // part)) % banks
        assert offset == i // banks
        physical = bank * span + offset
        assert 0 <= physical < n and not seen[physical]
        seen[physical] = 1
        base = offset << bank_log
        low = bank ^ ((base >> part_log) & (banks - 1))
        assert base + low == i
    assert all(seen)

    contiguous_groups = n // banks
    for a in range(contiguous_groups):
        group = [((a * banks + u) ^ ((a * banks + u) >> part_log))
                 & (banks - 1) for u in range(banks)]
        assert len(set(group)) == banks

    valid_v = min(banks, n // part)
    for c in range(part):
        group = [((c + v * part) ^ ((c + v * part) >> part_log))
                 & (banks - 1) for v in range(valid_v)]
        assert len(set(group)) == valid_v
    return {
        "N": n, "Npart": part, "M": banks,
        "bijection_values": n,
        "aligned_contiguous_groups": contiguous_groups,
        "cross_stride_groups": part,
        "cross_stride_group_width": valid_v,
        "cross_stride_complete": valid_v == banks,
        "status": "pass",
    }


def main():
    parameters = {
        (n_log, part_log, bank_log)
        for bank_log in range(1, 7)
        for part_log in range(bank_log, 11)
        for n_log in range(part_log, 15)
    }
    parameters.add((16, 8, 5))  # Hermes's stated maximum-N configuration.
    cases = [check_case(*case) for case in sorted(parameters)]
    n, part, banks = 65536, 256, 32
    flattened = [((i ^ (i // part)) % banks) * (n // banks) + i // banks
                 for i in range(banks)]
    interleaved_banks = [address % banks for address in flattened]
    assert set(interleaved_banks) == {0}
    result = {
        "status": "pass",
        "domain": "1<=k<=6; k<=s<=10; s<=log2(N)<=14; plus (16,8,5)",
        "case_count": len(cases),
        "totals": {
            key: sum(case[key] for case in cases)
            for key in ("bijection_values", "aligned_contiguous_groups", "cross_stride_groups")
        },
        "complete_cross_stride_cases": sum(case["cross_stride_complete"] for case in cases),
        "partial_cross_stride_cases": sum(not case["cross_stride_complete"] for case in cases),
        "flatten_counterexample": {
            "N": n, "Npart": part, "M": banks,
            "logical_indices": [0, banks - 1],
            "meaning": "inclusive logical range; one word per value in hypothetical M-bank interleaving",
            "distinct_appt_banks": banks,
            "distinct_flattened_modulo_banks": len(set(interleaved_banks)),
            "flattened_modulo_bank": 0,
        },
        "cases": cases,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
