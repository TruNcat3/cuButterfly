"""CPU-only contract and output ownership checks for the QuaRot diagnostic."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('quarot_contract_bench', ROOT / 'scripts/benchmark_quarot_contracts.py')
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


def test_eight_contracts_preserve_full_normalization(tmp_path):
    path = ROOT / 'configs/experiments/quarot_contracts_20261107.json'
    cases = bench.load_cases(path)
    assert len(cases) == 8
    assert {(c['N'], c['full_n']) for c in cases} == {(32, 32), (64, 11008)}
    cases[1]['full_n'] = 64
    invalid = tmp_path / 'invalid.json'
    invalid.write_text(json.dumps({'cases': cases}))
    with pytest.raises(ValueError, match='full_n'):
        bench.load_cases(invalid)


def test_cpu_oracle_matches_dense_hadamard():
    import torch
    n = 32
    source = torch.linspace(-1, 1, 2 * n).half().reshape(2, n)
    matrix = torch.tensor([[(-1.0) ** bin(i & j).count('1') for j in range(n)] for i in range(n)])
    scale = float(1.0 / torch.tensor(32).sqrt())
    expected = (source.float() @ matrix * scale).half()
    actual = bench.oracle(torch, source, n, scale)
    assert torch.equal(actual, expected)


def test_existing_output_is_never_overwritten(tmp_path, monkeypatch):
    library = tmp_path / 'lib.so'
    library.write_bytes(b'not loaded in this test')
    output = tmp_path / 'report.json'
    output.write_text('existing report')
    monkeypatch.setattr(sys, 'argv', ['benchmark', '--library', str(library), '--output', str(output)])
    monkeypatch.setattr(bench, 'run', lambda *_: pytest.fail('GPU work must not start'))
    with pytest.raises(FileExistsError):
        bench.main()
    assert output.read_text() == 'existing report'


def test_profile_replay_preserves_inference_tensor_lifetime(monkeypatch):
    from contextlib import nullcontext
    from types import SimpleNamespace
    import torch

    with torch.inference_mode():
        output = torch.zeros(1)
    # Exercise actual inference-tensor mutation while replacing GPU-only plumbing.
    graph = SimpleNamespace(replay=lambda: output.add_(1))
    stream = SimpleNamespace(synchronize=lambda: None)
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)
    monkeypatch.setattr(torch.cuda, 'stream', lambda _: nullcontext())
    monkeypatch.setattr(torch.profiler, 'profile',
                        lambda **_: nullcontext(SimpleNamespace(events=lambda: [])))
    result = bench.profile_graph(torch, graph, stream)
    assert result['status'] == 'ok'
    assert output.item() == 1


def test_graph_timing_normalizes_complete_calls():
    assert bench.ms_per_call(12.0, repeats=100, graph_calls=20) == pytest.approx(0.006)
    assert bench.ms_per_call(12.0, repeats=100, graph_calls=1) == pytest.approx(0.12)
