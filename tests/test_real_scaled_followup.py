"""Dependency completion cannot be inferred from an idle or reused PID."""
import copy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import run_real_scaled_followup as followup


def test_waits_for_completed_state_and_original_process_exit():
    state = {'pid': 123, 'identity': {'gpu_uuid': 'gpu'}, 'status': 'running'}
    identity = followup.runner.digest(state['identity'])
    assert followup.dependency_state(state, identity, 123, 'start', 'start').startswith('waiting')
    with pytest.raises(RuntimeError, match='without completed'):
        followup.dependency_state(state, identity, 123, 'start', None)
    state['status'] = followup.FINISHED
    assert followup.dependency_state(state, identity, 123, 'start', 'start').startswith('waiting')
    assert followup.dependency_state(state, identity, 123, 'start', None) == 'ready'


@pytest.mark.parametrize('change,live,exception', [
    ({'pid': 124}, 'start', ValueError),
    ({'identity': {'gpu_uuid': 'other'}}, 'start', ValueError),
    ({}, 'reused', ValueError),
    ({'status': 'stopped-investigation-required'}, 'start', RuntimeError),
])
def test_changed_or_failed_dependency_stops(change, live, exception):
    original = {'pid': 123, 'identity': {'gpu_uuid': 'gpu'}, 'status': 'running'}
    state = copy.deepcopy(original)
    state.update(change)
    with pytest.raises(exception):
        followup.dependency_state(state, followup.runner.digest(original['identity']), 123, 'start', live)
