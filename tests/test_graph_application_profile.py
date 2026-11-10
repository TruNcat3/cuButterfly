import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location('graph_application_profile',
    Path(__file__).resolve().parents[1] / 'experiments/tools/profile_graph_application.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_union_does_not_double_count_overlapping_kernels():
    assert MODULE.interval_union([(5, 9), (1, 6), (2, 4), (11, 13)]) == 10
    assert MODULE.interval_union([]) == 0


def test_negative_duration_is_rejected():
    with pytest.raises(ValueError):
        MODULE.interval_union([(2, 1)])
