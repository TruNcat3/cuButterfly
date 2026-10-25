import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))

from calibrate_local_hardware import _bound_stage_points, _stage_template_key
from calibration_space import mapping_point


def point(operator="fwht", **mapping):
    return mapping_point(dict(operator=operator, precision="word32" if operator == "ntt" else "fp32",
                              logN=10, batch=5),
                         dict(backend="shared-iterative", stage_partition=[2, 3, 5],
                              tile_threads=96, **mapping))


def test_bulk_width_and_fusion_have_distinct_physical_template_keys():
    plain = point(batch_tile_count=1)
    bulk = point(batch_tile_count=4)
    fused = point(batch_tile_count=1, boundaries=[dict(residency="fused"), dict(residency="full-scratch")])
    assert len({_stage_template_key(p) for p in (plain, bulk, fused)}) == 3
    _, scope = _bound_stage_points([plain, bulk, fused], 2)
    assert scope["template_count"] == 3


def test_ntt_fused_partition_is_physical_and_matches_direct_groups():
    fused = point("ntt", boundary_mappings=[dict(storage="resident-fused"), dict(storage="full-scratch")])
    direct = dict(fused, mapping_json={"backend": "shared-iterative", "stage_partition": [5, 5],
                                     "tile_threads": 96})
    direct.pop("stage_partition")
    assert _stage_template_key(fused) == _stage_template_key(direct)
    assert _stage_template_key(fused) != _stage_template_key(point("ntt"))


def test_serialized_organization_wins_over_stale_top_level_aliases():
    base = point(batch_tile_count=4, local_stage_partitions=[[2], [3], [5]])
    stale = dict(base, batch_tile_count=1, tile_threads=256, stage_overlap=1, shared_layout="writer-aligned")
    base["mapping_json"] = {"backend": "shared-iterative", "stage_partition": [2, 3, 5],
                            "local_stage_partitions": [[2], [3], [5]], "tile_threads": 96,
                            "batch_tile_count": 4, "stage_overlap": 0, "shared_layout": "linear"}
    stale["mapping_json"] = base["mapping_json"]
    assert _stage_template_key(base) == _stage_template_key(stale)


def test_total_batch_is_load_but_local_partition_is_template():
    base = point(batch_tile_count=4)
    assert _stage_template_key(base) == _stage_template_key(dict(base, batch=9))
    assert _stage_template_key(base) != _stage_template_key(point(batch_tile_count=4, local_stage_partitions=[[2], [3], [5]]))
