import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "paper/tools/profile_application_boundaries_20261012.py"
SPEC = importlib.util.spec_from_file_location("application_boundaries", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_component_schema_has_fixed_categories_and_excludes_setup():
    report = MODULE.component_breakdown(
        application="fno", backend="cubutterfly",
        headline_scope="eager_interval", profiler=True, isolated=True)
    assert report["schema"] == "cubutterfly-application-breakdown-v1"
    assert [row["category"] for row in report["components"]] == list(MODULE.COMPONENT_CATEGORIES)
    assert "compile_plan" in report["headline"]["excludes"]
    assert set(report["headline"]["includes"]) == set(MODULE.COMPONENT_CATEGORIES[1:])
    assert report["profiler"]["nested_totals_are_not_additive"]


def test_profiler_normalization_preserves_events_and_marks_overlap():
    profile = {"events": [
        {"name": "application/fno_cubutterfly", "device_total_us": 100.0},
        {"name": "cubutterfly/rfftn", "device_total_us": 25.0},
        {"name": "aten::copy_", "device_total_us": 3.0},
    ]}
    report = MODULE.normalize_profiler_report(
        application="fno", backend="cubutterfly", profile=profile)
    assert [event["component"] for event in report["events"]] == [
        "application_rest", "operator_gpu", "application_rest"]
    assert all(event["timing_relation"] == "nested_or_independent" for event in report["events"])
    assert report["isolated_samples"] is False


def test_unknown_or_missing_samples_are_explicit():
    report = MODULE.component_breakdown(
        application="fft_filter", backend="torch",
        headline_scope="unknown", notes=("filter stage source is external",))
    assert all(row["status"] == "unmeasured" for row in report["components"])
    assert all(row["median_us"] is None for row in report["components"])
    assert "filter stage source is external" in report["notes"]
