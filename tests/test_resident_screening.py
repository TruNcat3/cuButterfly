"""Finite screening must expose new local choices before launch variants."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from calibration_space import stratified_candidates, candidate_id


def test_shared_resident_units_reach_small_budget():
    points = []
    for width in (0, 2, 4):
        for threads in (32, 64, 128, 256):
            mapping = dict(backend="shared-iterative", stage_partition=[8,4], tile_threads=threads,
                local_stage_partitions=[] if not width else [[width]*(8//width), [width]*(4//width)])
            points.append(dict(operator="fwht", backend="shared-iterative", fft_core="scalar",
                logN=12, batch=128, mapping_json=json.dumps(mapping)))
    selected = stratified_candidates(points, 3)
    widths = {max((v for p in json.loads(row["mapping_json"])["local_stage_partitions"] for v in p), default=0)
              for row in selected}
    assert widths == {0,2,4}
    assert len({candidate_id(row) for row in stratified_candidates(points,0)}) == len(points)


def test_rectangular_factors_and_exchange_chunks_both_reach_budget():
    points = []
    for parts in ([], [[6,5],[]]):
        for chunks in ([], [0,8]):
            for lanes in (1,2,4):
                mapping = dict(prefix_codelet="native", prefix_shared_layout="linear",
                    prefix_codelet_lanes=lanes, local_stage_partitions=parts, exchange_chunks=chunks)
                points.append(dict(operator="fft", backend="online-reorder", fft_core="register-tile",
                    mapping_json=json.dumps(mapping)))
    selected = [json.loads(row["mapping_json"]) for row in stratified_candidates(points,3)]
    assert selected[0]["local_stage_partitions"] == []
    assert any(row["local_stage_partitions"] for row in selected)
    assert any(row["exchange_chunks"] for row in selected)


def test_explicit_default_axes_do_not_consume_strategy_coverage():
    from calibration_space import _strategy_identity
    assert _strategy_identity({}, []) == _strategy_identity(
        dict(local_stage_partitions=[[],[]], exchange_chunks=[0,0]), [])
