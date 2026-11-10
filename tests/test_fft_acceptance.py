import pathlib
import sys
import json
import pytest
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/"scripts"))
from run_fft_acceptance import workload_matrix, summarize, configured_workloads


def test_configured_matrix_preserves_migration_semantics(tmp_path):
    path = tmp_path / "workloads.json"
    fft = dict(operator="fft", precision="fp64", logN=18, batch=7, placement="in-place",
               direction="inverse", normalization="inverse", element_stride=2, batch_stride=524291)
    path.write_text(json.dumps(dict(schema="cubutterfly-install-search-v1", workloads=[
        fft, fft, dict(operator="ntt", precision="word32", logN=12, batch=1)])))
    assert configured_workloads(path) == [dict(fft, requested_batches=[7])]
    fft["backend"] = "cufft"
    path.write_text(json.dumps(dict(schema="cubutterfly-install-search-v1", workloads=[fft])))
    with pytest.raises(ValueError, match="non-semantic"):
        configured_workloads(path)


def test_explicit_subset_cannot_be_labelled_full_matrix():
    sample = dict(kernel_ms="1", backend="online-reorder", mapping_json="{}", selection_confidence="calibrated-registry")
    cell = dict(workload=dict(precision="fp32"), measurements=[dict(name=n, sample=sample) for n in ("cuFFT", "cuButterfly")])
    protocol = dict(trials=1, precisions=["fp32"], log_n=list(range(3,25)), batches=[1,4,16,64], explicit_workloads=True)
    summary = summarize(dict(protocol=protocol, cells=[cell]))
    assert summary["precisions"]["fp32"]["selected_scope_target_met"]
    assert not summary["full_size_batch_matrix"]
    assert not summary["precisions"]["fp32"]["target_met"]


def test_memory_clamp_preserves_requested_cells_without_duplicate_weight():
    cells=workload_matrix(["fp32","fp64"],[24],[1,4,16,64],40*1024**3)
    double=[c for c in cells if c["precision"]=="fp64"]
    assert [c["batch"] for c in double]==[1,4,16,32]
    cells=workload_matrix(["fp64"],[24],[16,32,64],20*1024**3)
    assert len(cells)==1 and cells[0]["requested_batches"]==[16,32,64]


def test_partial_or_uncalibrated_results_never_satisfy_acceptance():
    sample=dict(kernel_ms="1",backend="shared-iterative",mapping_json="{}",selection_confidence="unmeasured-feasible")
    cell=dict(workload=dict(precision="fp32"), measurements=[dict(name=name,sample=sample) for name in ("cuFFT","cuButterfly")])
    document=dict(protocol=dict(trials=1,precisions=["fp32"]),cells=[cell])
    assert not summarize(document)["precisions"]["fp32"]["target_met"]
    sample["selection_confidence"]="confirmed"
    assert summarize(document)["precisions"]["fp32"]["selected_scope_target_met"]
    assert not summarize(document)["precisions"]["fp32"]["target_met"]
    document["protocol"]["trials"]=3
    assert summarize(document)["precisions"]["fp32"]["completed"]==0
