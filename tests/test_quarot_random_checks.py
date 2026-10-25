import importlib.util
from pathlib import Path

import pytest
import torch


PATH = Path(__file__).resolve().parents[1] / "paper/experiments/e13/quarot_random_checks.py"
SPEC = importlib.util.spec_from_file_location("quarot_random_checks", PATH)
checks = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(checks)


def test_seeded_random_int4_is_non_degenerate_and_has_positive_scales():
    def make(seed):
        return checks.random_quantized_weight(16, 32, torch.Generator().manual_seed(seed))
    packed, scales = make(42)
    same_packed, same_scales = make(42)
    assert torch.equal(packed, same_packed)
    assert torch.equal(scales, same_scales)
    assert not torch.equal(packed, make(43)[0])
    low, high = (packed & 15).to(torch.int8), (packed >> 4).to(torch.int8)
    low[low >= 8] -= 16
    high[high >= 8] -= 16
    unpacked = torch.stack((low, high), dim=-1).flatten(1)
    reconstructed = unpacked.float() * scales
    assert bool((scales > 0).all() and torch.isfinite(scales).all())
    assert bool((unpacked >= -7).all() and (unpacked <= 7).all())
    assert bool((unpacked > 0).any() and (unpacked < 0).any())
    assert bool((reconstructed.abs().sum(dim=1) > 0).all())


def test_position_prefill_decode_matches_full_sequence():
    full = checks.checked_positions(torch.arange(10).unsqueeze(0), 10, 10)
    prefill = checks.checked_positions(torch.arange(8).unsqueeze(0), 8, 8)
    decode = [checks.checked_positions(torch.tensor([[position]]), 1, position + 1) for position in (8, 9)]
    assert torch.equal(full, torch.cat([prefill, *decode], dim=1))
    with pytest.raises(RuntimeError, match="disagree"):
        checks.checked_positions(torch.tensor([[0]]), 1, 9)
    with pytest.raises(RuntimeError, match="explicit"):
        checks.checked_positions(None, 1, 9)


def test_shared_cache_length_update_preserves_positions_in_later_layers():
    prefill = torch.arange(8).unsqueeze(0)
    assert torch.equal(checks.checked_attention_positions(prefill, 8, 0, 0), prefill)
    assert torch.equal(checks.checked_attention_positions(prefill, 8, 8, 1), prefill)
    decode = torch.tensor([[8]])
    assert torch.equal(checks.checked_attention_positions(decode, 1, 8, 0), decode)
    assert torch.equal(checks.checked_attention_positions(decode, 1, 9, 31), decode)
    with pytest.raises(RuntimeError):
        checks.checked_attention_positions(torch.tensor([[0]]), 1, 9, 31)


def test_cpu_rope_decode_matches_full_sequence_positions():
    pytest.importorskip("transformers")
    from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
    rotary = LlamaRotaryEmbedding(8)
    values = torch.ones(1, 10, 2, 8)
    full_cos, full_sin = rotary(values, torch.arange(10).unsqueeze(0))
    position = checks.checked_attention_positions(torch.tensor([[8]]), 1, 8, 0)
    decode_cos, decode_sin = rotary(values[:, 8:9], position)
    assert torch.equal(full_cos[:, 8:9], decode_cos)
    assert torch.equal(full_sin[:, 8:9], decode_sin)


def test_numeric_check_never_passes_without_explicit_tolerance():
    reference = torch.tensor([1.0, 2.0])
    assert not checks.compare_outputs(reference, reference)["pass"]
    assert checks.compare_outputs(reference, reference, 0, 0)["pass"]
    assert not checks.compare_outputs(reference, reference + 0.5, 0.01, 0.01)["pass"]
    assert not checks.compare_outputs(reference, torch.tensor([float("nan"), 2.0]), 1, 1)["finite"]


def test_numeric_check_counts_nonfinite_values_for_both_variants():
    result = checks.compare_outputs(torch.tensor([float("nan"), float("inf"), 1.0]),
                                    torch.tensor([0.0, float("-inf"), 1.0]), 1, 1)
    assert result["reference_nonfinite_count"] == 2
    assert result["actual_nonfinite_count"] == 1
    assert not result["finite"] and not result["pass"]


def test_relative_l2_and_exact_interpolated_quantiles():
    result = checks.compare_outputs(torch.ones(4), torch.arange(1.0, 5.0))
    assert result["abs_error_quantiles"] == pytest.approx({"p50": 1.5, "p90": 2.7, "p99": 2.97})
    assert result["relative_l2_error"] == pytest.approx((14 / 4) ** 0.5)
    assert checks.compare_outputs(torch.zeros(4), torch.zeros(4))["relative_l2_error"] == 0
    assert checks.compare_outputs(torch.zeros(4), torch.ones(4))["relative_l2_error"] is None


def test_exact_quantiles_above_torch_quantile_size_limit():
    # This is the first failing torch.quantile size, without allocating a model.
    count = 2 ** 24 + 1
    reference = torch.ones(count)
    actual = torch.zeros(count)
    actual[count // 2:] = 3
    result = checks.compare_outputs(reference, actual)
    assert result["abs_error_quantiles"] == {"p50": 2.0, "p90": 2.0, "p99": 2.0}
    assert result["abs_error_quantiles_method"] == "numpy.quantile/full_array/linear"
    assert result["relative_l2_error"] == pytest.approx(
        ((count // 2 + 4 * (count - count // 2)) / count) ** 0.5)
    assert result["reference_nonfinite_count"] == result["actual_nonfinite_count"] == 0


def test_state_hash_covers_packed_weights_and_scales():
    model = torch.nn.Module()
    model.register_buffer("weight", torch.tensor([[0x17]], dtype=torch.uint8))
    model.register_buffer("weight_scales", torch.tensor([[0.01]], dtype=torch.float16))
    original = checks.state_hash(model)
    assert checks.state_summary(model)["positive_finite_scale_count"] == 1
    assert checks.state_summary(model)["buffer_count"] == 2
    assert original == checks.state_hash(model)
    model.weight_scales.mul_(2)
    assert original != checks.state_hash(model)


def test_cpu_hadamard_oracle_matches_dense_matrix_and_inverse():
    from scipy.linalg import hadamard
    matrix = torch.from_numpy(hadamard(8)).double()
    values = torch.arange(16).reshape(2, 8).half() / 8
    oracle = checks.cpu_hadamard_reference(values, None, 1)
    assert torch.allclose(oracle, values.double() @ matrix / 8**0.5, atol=1e-14, rtol=0)
    assert torch.allclose(checks.cpu_hadamard_reference(oracle, None, 1), values.double(), atol=1e-14, rtol=0)
    factor = torch.tensor([[1, 1], [1, -1]])
    assert torch.allclose(checks.cpu_hadamard_reference(values, factor, 2), oracle, atol=1e-14, rtol=0)


def test_float64_oracle_error_is_not_rounded_to_float32():
    reference = torch.tensor([1.0 + 2**-40], dtype=torch.float64)
    error = checks.compare_outputs(reference, torch.tensor([1.0]))
    assert error["max_abs_error"] == 2**-40
