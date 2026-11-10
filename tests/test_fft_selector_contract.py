"""CPU-only contracts for the runtime FFT selector evidence boundary."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _function(source: str, signature: str) -> str:
    start = source.index(signature)
    brace = source.index("{", start)
    depth = 1
    end = brace + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end]


def test_auto_selector_prefers_exact_registry_evidence_before_portable_fallback():
    source = (ROOT / "src/mapping_selector.cpp").read_text()
    function = _function(source, "ButterflySelectionResult select_butterfly_mapping")
    assert function.index("registry_butterfly_mapping(config)") < function.index("select_portable_butterfly(config)")


def test_registry_fft_match_is_shape_and_semantics_bound():
    source = (ROOT / "src/plan_registry.cpp").read_text()
    function = _function(source, "bool matches_semantics(const Json& key, const ButterflyConfig& c)")
    for field in (
        '"operator"', '"precision"', '"placement"', '"logN"', '"batch"',
        '"direction"', '"accumulation"', '"element_stride"', '"batch_stride"',
        '"normalization"',
    ):
        assert field in function


def test_unmeasured_fallback_is_explicitly_labeled():
    source = (ROOT / "src/plan_registry.cpp").read_text()
    function = _function(source, "Result select_static_inventory")
    assert '"unmeasured-feasible"' in function
    assert "no local timing estimate" in function
    assert 'rank_samples(request)' in function
    assert 'rank_samples(compiled)' in function
