import copy
import pathlib
import sys

import pytest


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from benchmark_semantics import normalize_benchmark_semantics
from calibration_space import command_for


@pytest.mark.parametrize(
    "operator", ["fft", "fwht", "subset-zeta", "superset-zeta", "xor-zeta"]
)
@pytest.mark.parametrize("modulus", [0, "0"])
def test_non_ntt_neutral_modulus_is_omitted(operator, modulus):
    point = {"operator": operator, "modulus": modulus, "precision": "fp32"}
    normalized = normalize_benchmark_semantics(point)
    assert "modulus" not in normalized
    assert "--modulus" not in command_for("bench", point)


def test_non_ntt_nonzero_modulus_is_rejected():
    with pytest.raises(ValueError, match="nonzero modulus"):
        normalize_benchmark_semantics({"operator": "fft", "modulus": 17})


def test_ntt_keeps_modulus_and_word_precision_and_omits_empty_matrices():
    point = {
        "operator": "ntt",
        "precision": "word64",
        "modulus": 998244353,
        "stage_matrices": "",
    }
    normalized = normalize_benchmark_semantics(point)
    assert normalized == {
        "operator": "ntt",
        "precision": "word64",
        "modulus": 998244353,
    }
    command = command_for("cuntt_bench", point)
    assert command[command.index("--modulus") + 1] == "998244353"
    assert command[command.index("--word-bits") + 1] == "64"
    assert "--stage-matrices" not in command


def test_structured_registry_matrix_lowers_to_cli_matrix():
    point = {
        "operator": "structured-2x2",
        "stage_matrices": "1:0.25:-0.5:1",
    }
    normalized = normalize_benchmark_semantics(point)
    assert normalized["stage_matrix"] == "1,0.25,-0.5,1"
    command = command_for("cubutterfly_bench", point)
    assert command[-2:] == ["--stage-matrix", "1,0.25,-0.5,1"]


def test_structured_registry_per_stage_matrices_repeat_cli_option():
    point = {
        "operator": "structured-2x2",
        "stage_matrices": "1:0:0:1x2:3:4:5",
    }
    normalized = normalize_benchmark_semantics(point)
    assert normalized["stage_matrix"] == ["1,0,0,1", "2,3,4,5"]
    command = command_for("cubutterfly_bench", point)
    assert command[-4:] == [
        "--stage-matrix", "1,0,0,1", "--stage-matrix", "2,3,4,5"
    ]


def test_stage_matrix_conflict_is_rejected():
    point = {
        "operator": "structured-2x2",
        "stage_matrices": "1:0:0:1",
        "stage_matrix": "2,0,0,2",
    }
    with pytest.raises(ValueError, match="combined"):
        normalize_benchmark_semantics(point)


def test_original_point_and_semantic_axes_are_unchanged():
    point = {
        "operator": "structured-2x2",
        "precision": "fp32",
        "direction": "inverse",
        "element_stride": 2,
        "batch_stride": 19,
        "mapping_json": {"backend": "temporal-tile", "stages": [1, 2]},
        "stage_matrices": "1:0.25:-0.5:1",
        "modulus": "0",
    }
    original = copy.deepcopy(point)
    normalized = normalize_benchmark_semantics(point)
    assert point == original
    for key in ("direction", "element_stride", "batch_stride", "precision", "mapping_json"):
        assert normalized[key] == point[key]


def test_unknown_option_is_kept_for_the_original_cli_error_path():
    command = command_for("cubutterfly_bench", {
        "operator": "fft",
        "unknown_option": "value",
    })
    assert command[-2:] == ["--unknown-option", "value"]


def test_ntt_nonempty_stage_matrices_are_not_silently_dropped():
    with pytest.raises(ValueError, match="NTT benchmark does not support"):
        normalize_benchmark_semantics({
            "operator": "ntt",
            "precision": "word32",
            "modulus": 17,
            "stage_matrices": "1:0:0:1",
        })
