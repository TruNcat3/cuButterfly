"""CPU-only checks of finite budgets, side selection, and full-output oracle."""
import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('nonfft_endpoint', ROOT / 'scripts/run_nonfft_endpoint_control.py')
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


def test_finite_pool_preserves_endpoint_and_fixed_leaf():
    config = bench.read(bench.DEFAULT_CONFIG)
    study = bench.make_study(config)
    assert [len(c['variants']) for c in study['cases']] == [6, 6, 3, 3]
    for cell, case in zip(config['cells'], study['cases']):
        assert any(v['mapping'] == cell['mapping'] for v in case['variants'])
        for variant in case['variants']:
            if cell['workload']['operator'] == 'ntt':
                assert all(m['core'] == cell['leaf'] and m['threads_per_block'] == 256
                           for m in variant['mapping']['subgraph_mappings'])
            else:
                assert variant['mapping']['compute_unit'] == cell['leaf']
                assert variant['mapping']['stage_partition'] == [8, 8]
    assert {r['logical_data_path_bytes'] for r in bench.resources(config)} == {16777216, 536870912}


def test_selection_respects_side_and_rejects_failed_fast_candidate():
    config = bench.read(bench.DEFAULT_CONFIG)
    study = bench.make_study(config)
    screen = copy.deepcopy(study)
    for case in screen['cases']:
        for variant in case['variants']:
            variant['preflight'] = {'status': 'accepted', 'resolved_mapping': variant['mapping']}
            organization = int(variant['id'].split('-o')[1])
            variant['trials'] = [{'performance_accepted': organization != 4, 'kernel_ms': 1 / organization}]
    winners = bench.select_winners(screen, config)
    for cell in winners['cells']:
        assert cell['choices']['conventional']['id'].endswith('-o1')
        assert cell['choices']['extended']['id'].endswith('-o2')
        assert not cell['same_winner']
    # When extended offers no faster point, identical winners are represented once.
    for case in screen['cases']:
        for variant in case['variants']:
            variant['trials'][0]['kernel_ms'] = 1 if variant['id'].endswith('-o1') else 2
    winners = bench.select_winners(screen, config)
    assert all(c['same_winner'] for c in winners['cells'])
    assert all(len({v['id'] for v in c['choices'].values()}) == 1 for c in winners['cells'])


def test_fp32_oracle_matches_dense_and_retains_dtype():
    values = np.arange(48, dtype=np.float32).reshape(3, 16) / np.float32(7)
    matrix = np.array([[(-1) ** (i & j).bit_count() for j in range(16)] for i in range(16)], dtype=np.float32)
    result = bench.fp32_reference(values)
    assert result.dtype == np.float32
    np.testing.assert_allclose(result, values @ matrix, rtol=2e-6, atol=2e-5)
    assert not np.shares_memory(result, values)
