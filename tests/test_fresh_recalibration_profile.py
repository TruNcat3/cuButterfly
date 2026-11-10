"""Fresh rate probes must not inherit old measurements or hardware identity."""
import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from finalize_recalibration import select_hardware_profile


def test_fresh_profile_replaces_all_old_measurements():
    current = dict(device="GPU", compute_capability="8.0", global_memory_bytes="1024")
    old = dict(current, capabilities={"rate": 1}, stage_service={"old": True}, old_timing=10)
    fresh = dict(device="GPU", compute_capability="8.0", capabilities={"rate": 2})
    original = copy.deepcopy(fresh)
    result = select_hardware_profile(old, current, fresh)
    assert result == dict(fresh, global_memory_bytes=1024)
    assert fresh == original
    assert "old_timing" not in result and "stage_service" not in result


@pytest.mark.parametrize("changed", [dict(device="other"), dict(compute_capability="7.0"),
                                    dict(global_memory_bytes=2048), dict(stage_service={"old": True}),
                                    dict(capabilities={})])
def test_rejects_wrong_target_or_inherited_services(changed):
    current = dict(device="GPU", compute_capability="8.0", global_memory_bytes=1024)
    fresh = dict(current, capabilities={"rate": 2}, **{})
    fresh.update(changed)
    with pytest.raises(ValueError):
        select_hardware_profile(current, current, fresh)
