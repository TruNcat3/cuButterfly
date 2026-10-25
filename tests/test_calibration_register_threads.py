import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from calibration_space import command_for, predicted_sample, stratified_candidates
from calibrate_local_hardware import _stage_template_key


def test_register_threads_survives_template_identity_and_resolves_tile_threads():
    base = {"operator": "fwht", "precision": "fp32", "logN": 8, "batch": 4,
            "backend": "temporal-tile", "fft_core": "scalar", "local_stages": 0,
            "local_exchange": "warp-register", "tile_threads": 128,
            "register_threads": 128}
    alternate = {**base, "tile_threads": 256, "register_threads": 256}
    assert _stage_template_key(base) != _stage_template_key(alternate)
    assert predicted_sample(base)["tile_threads"] == 128
    assert predicted_sample(alternate)["tile_threads"] == 256


def test_legacy_zero_register_threads_keeps_default_shape_and_mapping_cli_is_single_argument():
    point = {"operator": "fwht", "precision": "fp32", "logN": 8, "batch": 4,
             "backend": "temporal-tile", "local_exchange": "warp-register",
             "tile_threads": 128, "register_threads": 0}
    sample = predicted_sample(point)
    assert sample["register_threads"] == 0
    assert sample["tile_threads"] == 128
    mapping = {**point, "mapping_json": json.dumps(point)}
    command = command_for("bench", mapping)
    assert command.count("--mapping-json") == 1
    assert "--register-threads" not in command
    assert "--tile-threads" not in command


def test_stratified_candidates_do_not_collapse_real_register_shapes():
    points = []
    for threads in (128, 256):
        point = {"operator": "fwht", "precision": "fp32", "logN": 8,
                 "backend": "temporal-tile", "fft_core": "scalar",
                 "local_exchange": "warp-register", "tile_threads": threads,
                 "register_threads": threads, "mapping_json": "{}"}
        points.append(point)
    selected = stratified_candidates(points, 0)
    assert {point["register_threads"] for point in selected} == {128, 256}
