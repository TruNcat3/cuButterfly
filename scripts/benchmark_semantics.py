"""Lower registry semantic fields to the two benchmark CLI protocols.

The hardware registry uses a common semantic schema for Butterfly and NTT
records.  The benchmark binaries do not share that schema: Butterfly has no
modulus option and accepts repeated ``--stage-matrix`` options, while NTT has
``--modulus``/``--word-bits`` and no stage-matrix option.  This module keeps
that small protocol adaptation independent of the rest of the project so it
can also be loaded by frozen compatibility launchers.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping


def _is_neutral_modulus(value: Any) -> bool:
    """Return whether *value* is the registry's neutral modulus spelling."""
    if isinstance(value, bool):
        return False
    if isinstance(value, str):
        return value.strip() == "0"
    try:
        return value == 0
    except (TypeError, ValueError):
        return False


def _is_empty_stage_matrices(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (list, tuple)):
        return not value
    return False


def _decode_stage_matrices(value: Any) -> list[str]:
    """Decode registry ``matrix:matrixxmatrix`` text for repeated CLI flags."""
    if isinstance(value, str):
        encoded = value.strip().split("x")
    elif isinstance(value, (list, tuple)):
        encoded = list(value)
    else:
        raise ValueError(
            "stage_matrices must be a colon-separated registry string"
        )

    matrices: list[str] = []
    for index, item in enumerate(encoded):
        if isinstance(item, (list, tuple)):
            coefficients = [str(coefficient).strip() for coefficient in item]
        elif isinstance(item, str):
            text = item.strip()
            if "," in text:
                raise ValueError(
                    "stage_matrices must use colon-separated 2x2 matrices"
                )
            coefficients = [coefficient.strip() for coefficient in text.split(":")]
        else:
            raise ValueError(
                f"stage_matrices matrix {index} is not a supported 2x2 encoding"
            )
        if len(coefficients) != 4 or any(not coefficient for coefficient in coefficients):
            raise ValueError(
                f"stage_matrices matrix {index} must contain four coefficients"
            )
        matrices.append(",".join(coefficients))
    return matrices


def _has_stage_matrix(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple)):
        return bool(value)
    return True


def normalize_benchmark_semantics(point: Mapping[str, Any]) -> dict[str, Any]:
    """Return a copied point lowered to a benchmark-compatible schema.

    Non-NTT points may omit only the neutral registry ``modulus=0``.  A
    non-neutral modulus is rejected because ``cubutterfly_bench`` has no
    modulus protocol.  Registry stage matrices are lowered to one comma
    separated value or a list of such values; ``command_for`` emits the list
    as repeated ``--stage-matrix`` options, matching the C++ CLI.  NTT keeps
    its modulus and word precision unchanged, while a non-empty stage matrix
    is rejected because ``cuntt_bench`` has no corresponding option.
    """
    if not isinstance(point, Mapping):
        raise TypeError("benchmark point must be a mapping")

    result = deepcopy(dict(point))
    ntt = result.get("operator") == "ntt"

    if "modulus" in result and not ntt:
        if _is_neutral_modulus(result["modulus"]):
            del result["modulus"]
        else:
            raise ValueError(
                "non-NTT benchmark cannot represent nonzero modulus"
            )

    if ntt and "stage_matrix" in result and _has_stage_matrix(result["stage_matrix"]):
        raise ValueError("NTT benchmark does not support stage_matrix")

    if "stage_matrices" not in result:
        return result

    raw_matrices = result.pop("stage_matrices")
    if _is_empty_stage_matrices(raw_matrices):
        return result

    if ntt:
        raise ValueError("NTT benchmark does not support non-empty stage_matrices")
    if "stage_matrix" in result:
        raise ValueError(
            "stage_matrices cannot be combined with an existing stage_matrix"
        )

    matrices = _decode_stage_matrices(raw_matrices)
    if not matrices:
        return result
    result["stage_matrix"] = matrices[0] if len(matrices) == 1 else matrices
    return result
