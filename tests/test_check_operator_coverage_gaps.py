import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


matrix = _load(
    "operator_coverage_gap_matrix_for_checker_tests",
    ROOT / "scripts" / "build_operator_coverage_gap_matrix.py",
)
checker = _load(
    "operator_coverage_gap_checker",
    ROOT / "scripts" / "check_operator_coverage_gaps.py",
)


def _manifest_path(tmpdir) -> Path:
    path = Path(str(tmpdir)) / "manifest.json"
    path.write_text(
        json.dumps(matrix.build_manifest(), sort_keys=True), encoding="utf-8"
    )
    return path


def test_manifest_load_audits_real_rank1_auto_candidates(tmpdir):
    document = checker._load_json(_manifest_path(tmpdir))
    real = [row for row in document["rows"] if row["section"] == "real_public_api"]

    assert len(real) == 16
    assert all(row["expected_execution"] == "candidate" for row in real)
    assert all(row["execution_mode"] == "auto" for row in real)
    assert document["counts"]["by_expected_execution"] == {
        "candidate": 106,
        "packed-only": 3,
        "unsupported": 4,
    }


def test_cpu_dry_run_has_explicit_large_filter_and_no_gpu(monkeypatch, tmpdir, capsys):
    manifest = _manifest_path(tmpdir)

    def fail_if_gpu_called(*_args, **_kwargs):
        raise AssertionError("CPU dry-run must not enter the GPU runner")

    monkeypatch.setattr(checker, "_run_gpu", fail_if_gpu_called)
    output = Path(str(tmpdir)) / "nested" / "dry-run.json"
    assert checker.main(["--manifest", str(manifest), "--output", str(output)]) == 0
    report = json.loads(output.read_text(encoding="utf-8"))

    assert report["mode"] == "cpu-dry-run"
    assert report["execution"] == {
        "gpu_execution": False,
        "benchmark_invocations": 0,
        "library_loads": 0,
        "device_queries": 0,
    }
    assert report["summary"] == {
        "selected": 104,
        "filtered": 9,
        "statuses": {"dry-run": 104},
    }
    assert {row["reason"] for row in report["selection"]["filtered_rows"]} == {
        "large-filter"
    }
    capsys.readouterr()


def test_candidate_and_large_filters_are_independent(tmpdir):
    manifest = checker._load_json(_manifest_path(tmpdir))

    selected, filtered = checker._select_rows(
        manifest, None, include_large=False, only_candidates=True
    )
    assert len(selected) == 97
    assert len(filtered) == 16
    assert all(row["expected_execution"] == "candidate" for row in selected)
    assert sum(item["reason"] == "large-filter" for item in filtered) == 9

    selected, filtered = checker._select_rows(
        manifest, ["large_n_batch"], include_large=True, only_candidates=True
    )
    assert len(selected) == 9
    assert len(filtered) == 104
    assert {item["reason"] for item in filtered} == {"section-filter"}


def test_row_filter_allows_isolated_large_anchor(tmpdir):
    manifest = checker._load_json(_manifest_path(tmpdir))
    selected, filtered = checker._select_rows(
        manifest, ["large_n_batch"], include_large=True,
        only_candidates=True, row_ids=["large-fft-fp32-log23-b256"]
    )
    assert [row["id"] for row in selected] == ["large-fft-fp32-log23-b256"]
    assert len(filtered) == 112
    assert {item["reason"] for item in filtered} == {"row-filter"}


def test_ntt_cpu_reference_round_trip_and_bit_reverse():
    forward = {
        "N": 8,
        "batch": 2,
        "modulus": "17",
        "direction": "forward",
        "output_order": "natural",
    }
    values = [(index * 3 + 1) % 17 for index in range(16)]
    transformed = checker._ntt_reference(forward, values)
    inverse = dict(forward, direction="inverse")
    assert checker._ntt_reference(inverse, transformed) == values

    bit_reversed = dict(forward, batch=1, output_order="bit-reversed")
    actual = checker._ntt_reference(bit_reversed, values[:8])
    natural = checker._ntt_radix2(values[:8], 17, False)
    assert actual == [natural[checker._reverse_bits(index, 3)] for index in range(8)]


def test_zeta_cpu_reference_round_trip_for_both_directions():
    values = [(index * 17 + 5) & 0xFFFFFFFF for index in range(16)]
    for operator in ("subset-zeta", "superset-zeta"):
        forward = {"shape": [8], "batch": 2, "direction": "forward",
                   "operator": operator}
        inverse = dict(forward, direction="inverse")
        transformed = checker._zeta_reference(forward, values)
        assert checker._zeta_reference(inverse, transformed) == values


def test_c_api_error_classification_does_not_hide_backend_errors():
    class FakeXd:
        STATUS_NOT_SUPPORTED = 3
        STATUS_ALLOC_FAILED = 4
        STATUS_CUDA_ERROR = 5
        STATUS_BACKEND_ERROR = 6
        STATUS_INTERNAL_ERROR = 7

    class Error:
        def __init__(self, status):
            self.status = status

    xd = FakeXd()
    assert checker._classify_c_api_error(xd, Error(3), expected_unsupported=True) == "expected-unsupported"
    assert checker._classify_c_api_error(xd, Error(3), expected_unsupported=False) == "defect-unsupported"
    assert checker._classify_c_api_error(xd, Error(6), expected_unsupported=False) == "environment-error"
    assert checker._classify_c_api_error(xd, Error(4), expected_unsupported=False) == "memory-limit"
    assert checker._classify_c_api_error(xd, Error(1), expected_unsupported=False) == "failed"


def test_load_rejects_old_real_rank1_unsupported_contract(tmpdir):
    document = matrix.build_manifest()
    row = next(row for row in document["rows"] if row["section"] == "real_public_api")
    row["expected_execution"] = "unsupported"
    row["coverage_state"] = "unsupported"
    document["counts"]["by_expected_execution"] = {
        "candidate": 105,
        "packed-only": 3,
        "unsupported": 5,
    }
    document["counts"]["by_coverage_state"] = {
        "boundary": 3,
        "missing-public-evidence": 81,
        "missing-semantic-anchor": 24,
        "unsupported": 5,
    }
    path = Path(str(tmpdir)) / "invalid.json"
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match="real public API row"):
        checker._load_json(path)


def test_large_validation_ranges_cover_every_element_with_bounded_blocks():
    ranges = list(checker._validation_block_ranges(10, block_elements=3))

    assert ranges == [(0, 3), (3, 6), (6, 9), (9, 10)]
    assert sum(stop - start for start, stop in ranges) == 10
    assert max(stop - start for start, stop in ranges) == 3


def test_large_constant_validator_checks_all_fft_blocks_without_full_temp():
    class FakeScalar:
        def __init__(self, value):
            self.value = value

        def item(self):
            return self.value

    class FakeMask:
        def __init__(self, value):
            self.value = value

        def any(self):
            return self

        def item(self):
            return self.value

    class FakeDifference:
        def __init__(self, values):
            self.values = values

        def max(self):
            return FakeScalar(max(abs(value) for value in self.values))

    class FakeTensor:
        def __init__(self, values, block_sizes):
            self.values = list(values)
            self.block_sizes = block_sizes

        def numel(self):
            return len(self.values)

        def view(self, *shape):
            assert shape == (-1,)
            return self

        def __getitem__(self, selection):
            assert isinstance(selection, slice)
            values = self.values[selection]
            self.block_sizes.append(len(values))
            return FakeTensor(values, self.block_sizes)

        def __sub__(self, value):
            return FakeDifference([item - value for item in self.values])

    class FakeTorch:
        @staticmethod
        def ne(block, expected):
            return FakeMask(any(item != expected for item in block.values))

        @staticmethod
        def abs(difference):
            return difference

    fft_blocks = []
    fft_stats = checker._validate_constant_output(
        FakeTorch, FakeTensor([1.0] * 10, fft_blocks), 1.0,
        tolerance=0.0, block_elements=3, label="large FFT impulse reference")
    assert fft_blocks == [3, 3, 3, 1]
    assert fft_stats == {
        "validation_block_elements": 3,
        "validation_blocks": 4,
        "max_abs": 0.0,
    }

    ntt_blocks = []
    ntt_stats = checker._validate_constant_output(
        FakeTorch, FakeTensor([1] * 10, ntt_blocks), 1,
        block_elements=3, label="large NTT impulse reference")
    assert ntt_blocks == [3, 3, 3, 1]
    assert ntt_stats == {
        "validation_block_elements": 3,
        "validation_blocks": 4,
        "exact": True,
    }

    mismatch_blocks = []
    with pytest.raises(AssertionError, match="large FFT impulse reference mismatch"):
        checker._validate_constant_output(
            FakeTorch, FakeTensor([1.0] * 9 + [1.01], mismatch_blocks), 1.0,
            tolerance=0.0, block_elements=3, label="large FFT impulse reference")
    assert mismatch_blocks == [3, 3, 3, 1]
    with pytest.raises(AssertionError, match="non-finite"):
        checker._validate_constant_output(
            FakeTorch, FakeTensor([float("nan")], []), 1.0,
            tolerance=1e-4, block_elements=1)
