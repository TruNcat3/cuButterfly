"""Public shared organization mappings must preserve runtime compile identity."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from research_compile_requests import _shared_partition, compile_request


@pytest.mark.parametrize("operator,precision", [
    ("fft", "fp32"), ("ntt", "word64"), ("fwht", "fp32"),
    ("subset-zeta", "uint32"), ("superset-zeta", "uint64"),
    ("structured-2x2", "fp64"),
])
@pytest.mark.parametrize("overlap,tile,compiled_tile", [(False,2,2),(False,4,4),(True,7,1)])
def test_fusion_and_batch_tiles_reach_shared_compile_request(operator, precision, overlap, tile, compiled_tile):
    point = dict(operator=operator, precision=precision, logN=8, batch=9,
                 backend="shared-iterative", stage_partition=[2,2,2,2],
                 stage_overlap=overlap, batch_tile_count=tile,
                 tile_threads=512, threads_per_block=512,
                 shared_layout="writer-aligned", dataflow_layout="hermes-xor",
                 local_stage_partitions=[[2,2],[2,2]])
    if operator == "ntt":
        point["boundary_mappings"] = [dict(storage=storage, buffers=1) for storage in
                                      ("resident-fused","full-scratch","resident-fused")]
    else:
        point["boundaries"] = [dict(residency=residency) for residency in
                               ("fused","global-scratch","fused")]
    request = compile_request(point, "research")
    assert request["stage_partition"] == [4,4]
    assert request["local_stage_partitions"] == [[2,2],[2,2]]
    assert request["threads"] == 512
    assert request["batch_tile"] == compiled_tile


@pytest.mark.parametrize("boundary", [dict(storage="ring"), dict(storage="resident-fused",buffers=2)])
def test_ntt_shared_projection_rejects_unconsumed_boundary_attributes(boundary):
    with pytest.raises(ValueError,match="shared NTT boundaries"):
        _shared_partition(dict(stage_partition=[4,4],boundary_mappings=[boundary]),"ntt",8)
