import importlib.util
import hashlib
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


PATH = Path(__file__).resolve().parents[1] / "experiments/workloads/e13/run_quarot_random_end_to_end.py"
SPEC = importlib.util.spec_from_file_location("quarot_graph_runner", PATH)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def test_timed_replay_resets_cache_copies_inputs_and_returns_owned_outputs():
    sequence = runner._CapturedSequence.__new__(runner._CapturedSequence)
    sequence.cache = SimpleNamespace(length=99, _needs_init=[False, False])
    sequence.input_ids = torch.zeros(2, 3, dtype=torch.long)
    sequence.next_token = torch.zeros(2, 1, dtype=torch.long)
    sequence.source_ids = torch.arange(6).reshape(2, 3)
    sequence.source_next = torch.tensor([[7], [8]])
    sequence.logits = [torch.zeros(2, 3)]
    calls = []

    def replay():
        assert sequence.cache.length == 0
        assert sequence.cache._needs_init == [True, True]
        assert torch.equal(sequence.input_ids, sequence.source_ids)
        assert torch.equal(sequence.next_token, sequence.source_next)
        sequence.cache.length = 5
        sequence.logits[0].fill_(len(calls) + 1)
        calls.append(True)

    sequence.graph = SimpleNamespace(replay=replay)
    first = sequence.timed_replay()
    sequence.source_ids.fill_(9)
    second = sequence.timed_replay()
    assert len(calls) == 2
    assert torch.equal(first[0], torch.ones(2, 3))
    assert torch.equal(second[0], torch.full((2, 3), 2.0))
    assert first[0].data_ptr() != sequence.logits[0].data_ptr()


def test_capture_preserves_first_failure_even_when_capture_end_fails(monkeypatch):
    from contextlib import nullcontext
    sequence = runner._CapturedSequence.__new__(runner._CapturedSequence)
    sequence.input_ids = torch.zeros(1, 2, dtype=torch.long)
    sequence.next_token = torch.zeros(1, 1, dtype=torch.long)
    sequence.prefill, sequence.decode = 2, 1
    active = []

    def begin():
        active.append(True)

    def end():
        raise RuntimeError("secondary capture end failure")

    def model(*args, **kwargs):
        if active:
            raise RuntimeError("first unsupported scalar operation")
        cache = SimpleNamespace(length=0, _needs_init=[True],
                                get_cache_specs_for_flash_infer=lambda mask: {})
        return SimpleNamespace(past_key_values=cache)

    sequence.model = model
    sequence.graph = SimpleNamespace(capture_begin=begin, capture_end=end)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: None)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: None)
    monkeypatch.setattr(torch.cuda, "Stream", lambda **kwargs: SimpleNamespace(wait_stream=lambda stream: None))
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: nullcontext())
    with pytest.raises(runner.GraphCaptureUnsupported, match="first unsupported scalar operation") as error:
        sequence.capture()
    assert "secondary capture end failure" not in str(error.value)


def test_capture_position_validation_uses_device_predicate_without_scalar_sync(monkeypatch):
    checks = runner.checked_attention_positions.__globals__
    captured = []
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda tensor: True))
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    monkeypatch.setattr(torch, "_assert_async", lambda predicate, message: captured.append(predicate))
    monkeypatch.setattr(torch, "equal", lambda *args: pytest.fail("host scalar check during capture"))
    positions = torch.tensor([[8]])
    assert checks["checked_positions"](positions, 1, 9) is positions
    checks["checked_positions"](torch.tensor([[0]]), 1, 9)
    assert captured[0].item() is True
    assert captured[1].item() is False


def test_cache_metadata_is_prepared_once_and_preserves_storage_and_step_lengths():
    cache = SimpleNamespace(length=0, _needs_init=[False, False])
    pages = torch.zeros(1)
    calls = []

    def specs(mask):
        assert mask is None
        calls.append(cache.length)
        return {"pages": pages, "offset": torch.tensor(cache.length)}

    cache.get_cache_specs_for_flash_infer = specs
    metadata = runner._prepare_graph_cache(cache, 3, 2)
    assert calls == [3, 4, 5]
    assert cache.length == 0 and cache._needs_init == [True, True]
    for length in (3, 4, 5):
        cache.length = length
        actual = cache.get_cache_specs_for_flash_infer(None)
        assert actual is metadata[length]
        assert actual["pages"] is pages
        assert actual["offset"].item() == length
    assert calls == [3, 4, 5]
    with pytest.raises(RuntimeError, match="attention mask"):
        cache.get_cache_specs_for_flash_infer(torch.ones(1))
    cache.length = 6
    with pytest.raises(RuntimeError, match="outside"):
        cache.get_cache_specs_for_flash_infer(None)


def test_graph_eager_consistency_remains_exact_without_application_threshold(monkeypatch):
    from contextlib import nullcontext
    args = SimpleNamespace(prefill=2, decode=1, atol=None, rtol=None)

    class InexactSequence:
        def __init__(self, *args):
            pass

        def capture(self):
            pass

        def replay(self, *args):
            return [torch.ones(1)]

    monkeypatch.setattr(runner, "_CapturedSequence", InexactSequence)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda device: None)
    reference = {"prefill": torch.zeros(1)}
    with pytest.raises(runner.GraphNumericalValidationError) as error:
        runner._measure_graph_sequence(None, nullcontext(), torch.zeros(1, 2, dtype=torch.long),
                                       torch.zeros(1, 1, dtype=torch.long), args, "cpu",
                                       reference, reference)
    assert error.value.checks["original"]["prefill"]["atol"] == 0
    assert error.value.checks["original"]["prefill"]["pass"] is False


def test_graph_extension_requires_matching_current_stream_receipt():
    with tempfile.TemporaryDirectory(prefix="quarot-graph-test-") as directory:
        tmp_path = Path(directory)
        extension = tmp_path / "_CUDA_graph.so"
        extension.write_bytes(b"test extension")
        with pytest.raises(RuntimeError, match="build_receipt"):
            runner._validated_graph_extension(extension)
        receipt = tmp_path / "build_receipt.json"
        receipt.write_text(json.dumps({"protocol": "quarot_current_stream_only_v1",
                                       "extension_sha256": hashlib.sha256(extension.read_bytes()).hexdigest()}))
        selected, identity = runner._validated_graph_extension(extension)
        assert selected == extension
        assert identity["protocol"] == "quarot_current_stream_only_v1"
        extension.write_bytes(b"changed extension")
        with pytest.raises(RuntimeError, match="hash mismatch"):
            runner._validated_graph_extension(extension)


def test_runner_defaults_and_historical_replay_are_cpu_parseable():
    args = runner.parse_args(['--output', 'report.json'])
    assert args.compute_contract == 'public_fp32_final'
    assert args.execution_mode == 'graph'
    for contract in ('native', 'fp32_composed'):
        args = runner.parse_args(['--output', 'old.json', '--compute-contract', contract,
                                  '--execution-mode', 'eager'])
        assert args.compute_contract == contract and args.execution_mode == 'eager'
