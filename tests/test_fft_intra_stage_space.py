"""Joint geometry/buffer CPU preflight never globally removes prefetch."""
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import fft_intra_stage_space as space

HARDWARE = dict(sm=80, max_threads_per_block=1024, max_shared_per_block_optin_bytes=166912)


class Database:
    def check_both(self, *args):
        return "supported"


@pytest.fixture(autouse=True)
def cpu_database(monkeypatch):
    monkeypatch.setattr(space, "load_database", lambda _: Database())


def case(log_n=20):
    return dict(logN=log_n, shape=[1 << log_n], batch=16, precision="fp64",
                inverse=False, normalize_inverse=False)


def test_joint_space_retains_legal_double_buffer_shapes_and_records_oversize():
    result = space.build_space(case(), HARDWARE, "unused")
    accepted = {entry["provider"]: entry for entry in result["accepted"]}
    rejected = {entry["provider"]: entry for entry in result["rejected"]}
    assert len(accepted) + len(rejected) == 24
    large = rejected["factor_f10x10_c4_e8_t4_p2"]
    assert {reason["code"] for reason in large["reasons"]} == {"shared-lower-bound-exceeds-hardware"}
    assert large["resource_lower_bounds"]["max_dynamic_shared_lower_bound_bytes"] == 262144
    small = accepted["factor_f10x10_c2_e8_t4_p2"]
    assert small["resource_lower_bounds"]["max_dynamic_shared_lower_bound_bytes"] == 131072
    family = [entry for entry in accepted.values() if entry["family"] == small["family"]]
    assert {entry["mapping"]["prefetch_depth"] for entry in family} == {0, 1, 2}
    assert not small["resource_lower_bounds"]["compiler_resources_known"]
    t1d0 = accepted["factor_f10x10_c2_e8_t1_p0"]
    t1d1 = accepted["factor_f10x10_c2_e8_t1_p1"]
    assert t1d1["resource_lower_bounds"]["max_dynamic_shared_lower_bound_bytes"] == \
        t1d0["resource_lower_bounds"]["max_dynamic_shared_lower_bound_bytes"]


def test_log16_partitions_and_source_semantics_are_preserved():
    original = case(16)
    before = dict(original)
    result = space.build_space(original, HARDWARE, "unused")
    assert original == before
    rows = result["accepted"] + result["rejected"]
    assert len(rows) == 48
    assert len({row["provider"] for row in rows}) == 32
    assert {tuple(row["mapping"]["factor_partition"]) for row in rows} == {(6, 10), (8, 8)}
    for row in rows:
        point = space.calibration_space.mapping_point(original, row["mapping"])
        assert point["batch"] == 16 and point["shape"] == [65536] and point["precision"] == "fp64"


def test_invalid_hardware_is_rejected_before_renderer(monkeypatch):
    monkeypatch.setattr(space.compile_module, "render_factor_fft", lambda *args: pytest.fail("renderer must not run"))
    with pytest.raises(ValueError, match="max_threads_per_block"):
        space.build_space(case(), {**HARDWARE, "max_threads_per_block": 0}, "unused")


def test_thread_constraints_filter_each_combination():
    result = space.build_space(case(), {**HARDWARE, "max_threads_per_block": 128}, "unused")
    assert result["accepted"]
    assert all(max(group["threads"] for group in entry["resource_lower_bounds"]["stages"]) <= 128
               for entry in result["accepted"])
    assert any(reason["code"] == "threads-exceed-hardware"
               for entry in result["rejected"] for reason in entry["reasons"])
