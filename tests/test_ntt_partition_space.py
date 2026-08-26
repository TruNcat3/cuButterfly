import importlib.util
from pathlib import Path
import sys


SCRIPT = Path(__file__).parents[1] / "scripts" / "generate_ntt_partition_space.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("partition_space", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_segment_count_is_an_explicit_dimension():
    document = MODULE.generate_space(
        12, 2, 32, 2, 6, 2, 10, (4, 6), 2, 1.0,
        ("radix2", "radix4", "radix8"), False)
    assert document["dimensions"]["segment_count"] == [2, 6]
    assert set(document["partition_count_by_segments"]) == {
        "2", "3", "4", "5", "6"}
    assert {point["segment_count"] for point in document["shortlist"]} == {
        2, 3, 4, 5, 6}
    assert all(sum(point["stage_partition"]) == 12
               for point in document["shortlist"])
    assert document["schema"] == "cuntt-ntt-partition-space-v2"
    assert all(point["minimum_execution_group_count"] <=
               point["segment_count"] for point in document["shortlist"])


def test_shortlist_is_bounded_per_segment_count():
    document = MODULE.generate_space(
        20, 1, 32, 2, 8, 2, 10, (6, 8, 10), 3, 0.5,
        ("radix4", "radix8"), False)
    counts = {}
    for point in document["shortlist"]:
        counts[point["segment_count"]] = counts.get(point["segment_count"], 0) + 1
        assert len(point["physical_templates"]) == 3
        assert point["execution_templates"]
    assert set(counts) == set(range(2, 9))
    assert all(1 <= count <= 3 for count in counts.values())
    assert all(count == 3 for segments, count in counts.items() if segments > 2)
    assert document["partition_count"] > len(document["shortlist"])
    assert {(point["segment_count"],
             point["execution_template"]["execution_group_count"])
            for point in document["execution_shortlist"]} == {
                (m, g) for m in range(2, 9) for g in range(2, m + 1)}


def test_resident_lowering_separates_logical_m_from_physical_g():
    templates = MODULE.execution_templates((5, 5, 5, 5))
    lowered = [point for point in templates
               if point["execution_stage_partition"] == [10, 10]]
    assert len(lowered) == 1
    assert lowered[0]["boundary_storage"] == [
        "resident-fused", "full-scratch", "resident-fused"]
    assert lowered[0]["execution_group_count"] == 2
    assert lowered[0]["materialized_boundary_count"] == 1


def test_output_side_large_subgraph_breaks_permutation_tie():
    document = MODULE.generate_space(
        20, 1, 32, 3, 3, 6, 8, (6, 8), 20, 1.0,
        ("radix4",), False)
    ordered = [point["stage_partition"] for point in document["shortlist"]]
    assert ordered.index([6, 6, 8]) < ordered.index([8, 6, 6])
