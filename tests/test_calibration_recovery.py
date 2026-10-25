"""Recovery must reject changed seed inputs before consuming GPU time."""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import calibrate_local_hardware as calibration


@pytest.mark.parametrize('pin_source', [False, True])
def test_seed_recovery_preflight_precedes_gpu(tmp_path, monkeypatch, pin_source):
    source = tmp_path / 'original' / 'registry.json'
    destination = tmp_path / 'recovery' / 'registry.json'
    output = tmp_path / 'recovery'
    output.mkdir()
    snapshot = {'schema': 'cubutterfly-mapping-seeds-v1',
                'inputs': {'files': [], 'registry_path': str(source)},
                'seeds': [], 'reuse': 'mapping-only; target legality, correctness and timing required'}
    snapshot_path = output / 'mapping_seed_snapshot.json'
    snapshot_path.write_text(json.dumps(snapshot))
    original = snapshot_path.read_bytes()
    monkeypatch.setenv('CUBUTTERFLY_REGISTRY', str(destination))
    command = ['calibrate_local_hardware.py', '--build-dir', str(tmp_path / 'build'),
               '--profile-dir', str(tmp_path / 'hardware'), '--output-dir', str(output),
               '--resume-search']
    if pin_source:
        command += ['--seed-registry', str(source)]
    monkeypatch.setattr(sys, 'argv', command)
    gpu_calls = []
    class BeforeGPU(Exception):
        pass
    def stop_before_gpu():
        gpu_calls.append(True)
        raise BeforeGPU
    monkeypatch.setattr(calibration, 'require_exclusive_gpu', stop_before_gpu)
    if pin_source:
        with pytest.raises(BeforeGPU):
            calibration.main()
        assert gpu_calls == [True]
    else:
        with pytest.raises(ValueError, match='mapping seed inputs changed'):
            calibration.main()
        assert not gpu_calls
    assert snapshot_path.read_bytes() == original
    assert not destination.exists()
    assert not (output / 'stage_calibration.json').exists()
