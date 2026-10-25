import json
import pathlib
import sys

import pytest


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import benchmark_xd_expanded as benchmark  # noqa: E402
import summarize_xd_expanded as summary  # noqa: E402


PROTOCOL = "20-call CUDA graph; 7 replays; planning, warmup, checking excluded"


def _manifest(binary="binary-a", gpu="GPU-a", protocol=PROTOCOL):
    return {
        "schema": 1,
        "population": "xd-v11-fixed64",
        "binary_sha256": binary,
        "gpu_uuid": gpu,
        "timing_protocol": protocol,
        "cufft_version": 123,
        "vkfft_library_sha256": "vk-a",
        "baselines": ["cufft-native", "vkfft"],
        "cases": benchmark.cases(),
    }


def _report(case, *, binary="binary-a", gpu="GPU-a", protocol=PROTOCOL,
            rows=None, status="passed"):
    return {
        "shape": case["shape"],
        "batch": case["batch"],
        "precision": case["precision"],
        "inverse": case["inverse"],
        "library_sha256": binary,
        "gpu": {"uuid": gpu},
        "cufft_version": 123,
        "protocol": protocol,
        "rows": rows or [],
        "status": status,
    }


def _row(requested, milliseconds, *, algorithm=None, status="passed"):
    result = {"requested": requested, "status": status}
    if algorithm is not None:
        result["algorithm"] = algorithm
    if milliseconds is not None:
        result["graph"] = {
            "median_ms": milliseconds,
            "graph_capture_calls": 20,
            "graph_replays": 7,
        }
    if requested == "vkfft":
        result["baseline_metadata"] = {"library_sha256": "vk-a"}
    return result


def test_paired_metrics_use_speedup_and_strict_win_thresholds():
    case = benchmark.cases()[0]
    record = {
        "case": case,
        "file_state": "loaded",
        "report": _report(case, rows=[
            _row("selected", 1.0, algorithm="axis-config"),
            _row("other-config", 2.0),
            _row("cufft-native", 2.0),
            _row("vkfft", 3.0),
        ]),
    }
    metrics = summary.paired_metric([record], "selected", "cufft-native")
    assert metrics["gmean"] == pytest.approx(2.0)
    assert metrics["median"] == pytest.approx(2.0)
    assert metrics["wins_gt_1"] == 1
    assert metrics["wins_gt_1_03"] == 1
    assert metrics["min"] == pytest.approx(2.0)
    assert metrics["max"] == pytest.approx(2.0)
    assert metrics["n"] == 1
    assert metrics["missing"] == 0
    assert metrics["failed"] == 0
    fastest = summary._fastest([record])
    assert fastest["n"] == 1
    assert fastest["wins"] == {"selected": 1, "cufft-native": 0, "vkfft": 0}


def test_gpu_uuid_prefix_is_normalized_without_merging_devices():
    case = benchmark.cases()[0]
    identity = summary._validate_identity(_manifest(), [(case['id'], _report(case, gpu='a'))])
    assert identity['gpu_uuid'] == 'a'
    with pytest.raises(summary.SummaryError, match='gpu_uuid mismatch'):
        summary._validate_identity(_manifest(), [(case['id'], _report(case, gpu='b'))])


def test_missing_baseline_is_unpaired_not_an_infinite_win():
    case = benchmark.cases()[0]
    record = {
        "case": case,
        "file_state": "loaded",
        "report": _report(case, rows=[
            _row("selected", 1.0, algorithm="axis-config"),
            _row("cufft-native", 2.0),
        ]),
    }
    direct = summary.paired_metric([record], "selected", "vkfft")
    fastest = summary._fastest([record])
    minimum = summary.paired_metric([record], "selected", "min_baseline")
    assert direct["n"] == 0 and direct["missing"] == 1 and direct["wins_gt_1"] == 0
    assert minimum["n"] == 0 and minimum["missing"] == 1
    assert fastest["n"] == 0 and fastest["excluded_missing_baseline"] == 1


def test_fastest_compares_all_three_complete_timings_and_skips_missing():
    case = benchmark.cases()[0]

    def record(selected, cufft, vkfft=None):
        rows = [_row("selected", selected), _row("cufft-native", cufft)]
        if vkfft is not None:
            rows.append(_row("vkfft", vkfft))
        return {
            "case": case,
            "file_state": "loaded",
            "report": _report(case, rows=rows),
        }

    complete = [
        record(1.0, 2.0, 3.0),
        record(3.0, 1.0, 2.0),
        record(2.0, 3.0, 1.0),
        record(1.0, 0.5),
    ]
    fastest = summary._fastest(complete)
    assert fastest["n"] == 3
    assert fastest["wins"] == {"selected": 1, "cufft-native": 1, "vkfft": 1}
    assert fastest["excluded_missing_baseline"] == 1
    assert fastest["basis"].startswith("lowest median_ms among selected")


def test_report_dir_counts_missing_cases_and_keeps_partial_metrics(tmpdir):
    tmp_path = pathlib.Path(str(tmpdir))
    manifest = _manifest()
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    case = manifest["cases"][0]
    report = _report(case, rows=[
        _row("selected", 1.0, algorithm="axis-config"),
        _row("cufft-native", 2.0),
    ])
    (tmp_path / f"{case['id']}.json").write_text(json.dumps(report), encoding="utf-8")

    result = summary.summarize_report_dir(tmp_path)
    group = result["groups"]["all64"][0]
    config = group["configurations"]["selected"]
    assert result["status"] == "partial"
    assert result["coverage"]["loaded_cases"] == 1
    assert result["coverage"]["missing_files"] == 63
    assert config["cufft-native"]["n"] == 1
    assert config["cufft-native"]["missing"] == 63
    assert config["vkfft"]["n"] == 0
    assert config["vkfft"]["missing"] == 64
    assert config["min_baseline"]["n"] == 0
    assert group["fastest"]["n"] == 0
    assert group["fastest"]["excluded_missing_baseline"] == 64


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("library_sha256", "binary-b", "binary_sha256 mismatch"),
        ("gpu", {"uuid": "GPU-b"}, "gpu_uuid mismatch"),
        ("protocol", "different protocol", "timing protocol mismatch"),
    ],
)
def test_identity_mismatch_refuses_aggregation(tmpdir, field, value, message):
    tmp_path = pathlib.Path(str(tmpdir))
    manifest = _manifest()
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    case = manifest["cases"][0]
    report = _report(case, rows=[
        _row("selected", 1.0, algorithm="axis-config"),
        _row("cufft-native", 2.0),
        _row("vkfft", 2.0),
    ])
    report[field] = value
    (tmp_path / f"{case['id']}.json").write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(summary.SummaryError, match=message):
        summary.summarize_report_dir(tmp_path)


def test_vkfft_identity_and_graph_protocol_mismatch_refuse_aggregation(tmpdir):
    tmp_path = pathlib.Path(str(tmpdir))
    manifest = _manifest()
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    case = manifest["cases"][0]
    rows = [
        _row("selected", 1.0),
        _row("cufft-native", 2.0),
        _row("vkfft", 2.0),
    ]
    rows[-1]["baseline_metadata"]["library_sha256"] = "vk-b"
    report = _report(case, rows=rows)
    (tmp_path / f"{case['id']}.json").write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(summary.SummaryError, match="vkfft baseline library_sha256 mismatch"):
        summary.summarize_report_dir(tmp_path)

    rows[-1]["baseline_metadata"]["library_sha256"] = "vk-a"
    rows[0]["graph"]["graph_replays"] = 8
    (tmp_path / f"{case['id']}.json").write_text(json.dumps(_report(case, rows=rows)), encoding="utf-8")
    with pytest.raises(summary.SummaryError, match="graph_replays mismatch"):
        summary.summarize_report_dir(tmp_path)
