import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_research_acceptance as acceptance


def test_semantic_cells_preserve_matrix_modulus_and_stride():
    base = dict(operator="structured-2x2", precision="fp32", logN=8, batch=2,
                stage_matrix="1,0.25,-0.5,1", placement="out-of-place")
    assert acceptance.cell_key(base) != acceptance.cell_key(dict(base,stage_matrix="1,0.5,-0.5,1"))
    assert acceptance.cell_key(base) != acceptance.cell_key(dict(base,element_stride=2))
    ntt = dict(operator="ntt",precision="word64",logN=12,batch=1,modulus=576460756061519873)
    assert acceptance.cell_key(ntt) != acceptance.cell_key(dict(ntt,modulus=576460756061519874))


def test_ranking_measures_selection_loss_without_fitting_plan_totals(monkeypatch):
    seen = []
    monkeypatch.setattr(acceptance.stage_cost_model, "fit", lambda rows,profile: seen.append(rows) or {})
    monkeypatch.setattr(acceptance.stage_cost_model, "predict", lambda sample,model,explain:
                        dict(kernel_ms=sample["prediction"],whole_plan_fallback_used=False))
    records = [dict(name=name,status="measured",correct=True,median_kernel_ms=observed,
                    samples=[dict(backend="shared-iterative",prediction=predicted,
                                  mapping_json=json.dumps(dict(backend="shared-iterative",tile_threads=threads)))])
               for name,observed,predicted,threads in [("a",10,12,128),("b",11,9,256),("c",30,35,512)]]
    result = acceptance.ranking(records,{})
    assert seen == [[]]
    assert result["top1"] is False and result["top2"] is True
    assert result["latency_regret"] == pytest.approx(1.1)
    assert result["pairwise_correct"] == 2
    assert result["pairwise_count"] == 3
    assert result["all_services_covered"]


def test_confirmed_services_normalizes_csv_probe_numbers_and_calibrates(monkeypatch, tmpdir):
    tmp_path = Path(str(tmpdir))
    mapping = json.dumps({"backend": "shared-iterative", "tile_threads": 128})
    common = dict(operator="ntt", precision="word64", logN="16", batch="2",
                  placement="out-of-place", direction="forward", normalization="none",
                  accumulation="native", input_order="natural", output_order="natural",
                  mapping_json=mapping, backend="shared-iterative",
                  runtime_fingerprint="frozen")
    records = [
        dict(name="word64", status="measured", correct=True,
             samples=[dict(common, element_stride="3", batch_stride="131073",
                           modulus="576460756061519873")]),
        dict(name="defaults", status="measured", correct=True,
             samples=[dict(common, operator="fft", precision="fp32",
                           mapping_json=json.dumps({"backend": "shared-iterative",
                                                    "tile_threads": 256}))]),
    ]
    described = []
    calibration_points = []
    calibration_protocols = []

    def fake_run(command, check=True):
        point = json.loads(command[command.index("--point-json") + 1])
        described.append(point)
        sample = dict(point, runtime_fingerprint="frozen")
        return acceptance.SimpleNamespace(stdout=json.dumps({
            "status": "resolved", "sample": sample,
            "groups": [{"independent": True}],
        }), stderr="")

    def fake_calibrate(binary, checkpoint, profile, points, runner, exclusive, **kwargs):
        calibration_points.extend(points)
        calibration_protocols.append(kwargs)
        return {"status": "complete"}

    monkeypatch.setattr(acceptance, "run", fake_run)
    monkeypatch.setattr(acceptance, "require_exclusive_gpu", lambda: None)
    monkeypatch.setattr(acceptance.stage_cost_model, "fit", lambda rows, profile: {})
    monkeypatch.setattr(acceptance.stage_cost_model, "predict",
                        lambda sample, model, explain: {"whole_plan_fallback_used": True})
    monkeypatch.setattr(acceptance.stage_service_calibration, "calibrate", fake_calibrate)

    cell_dir = tmp_path / "cell"
    cell_dir.mkdir()
    source = tmp_path / "source_stage.json"
    source.write_text(json.dumps({"protocol": {"warmup": 10000, "repeat": 77, "trials": 3}}))
    args = acceptance.SimpleNamespace(build_dir=tmp_path, output_dir=tmp_path, stage_checkpoint=source)
    upgraded, gaps = acceptance.confirmed_services(records, args, {}, cell_dir)

    assert gaps == []
    assert calibration_protocols == [{"mode": "full", "warmup": 10000, "repeat": 77, "trials": 3}]
    assert len(described) == 2
    word64 = next(point for point in described if point["operator"] == "ntt")
    assert word64["logN"] == 16 and isinstance(word64["logN"], int)
    assert word64["batch"] == 2 and isinstance(word64["batch"], int)
    assert word64["element_stride"] == 3 and isinstance(word64["element_stride"], int)
    assert word64["batch_stride"] == 131073 and isinstance(word64["batch_stride"], int)
    assert word64["modulus"] == 576460756061519873
    defaults = next(point for point in described if point["operator"] == "fft")
    assert "element_stride" not in defaults
    assert "batch_stride" not in defaults
    assert "modulus" not in defaults
    assert len(calibration_points) == 2
    word64_request = next(point for point in calibration_points if point["operator"] == "ntt")
    assert word64_request["modulus"] == 576460756061519873
    defaults_request = next(point for point in calibration_points if point["operator"] == "fft")
    assert "element_stride" not in defaults_request
    assert "batch_stride" not in defaults_request
    assert "modulus" not in defaults_request
    assert all(sample["descriptor_source"] == "actual-stage-probe"
               for row in upgraded for sample in row["samples"])
    assert json.loads((cell_dir / "service_requests.json").read_text())["gaps"] == []


def test_summary_does_not_claim_full_migration_or_count_incomplete_pairs():
    doc = dict(status="running",limitations=[],cells=[
        dict(status="complete",measurements=[dict(name=name,sample={"kernel_ms":t})
             for name,t in [("cuButterfly",2),("cuFFT",4)]]),
        dict(status="comparing",measurements=[dict(name="cuButterfly",sample={"kernel_ms":1})])])
    result = acceptance.summarize(doc)
    assert result["completed_cells"] == 1
    assert result["expected_cells"] == 2
    assert result["baselines"]["cuFFT"]["geometric_mean_speedup"] == 2
    assert result["full_migration_qualified"] is False


def test_public_sample_rejects_wrong_semantics_and_failed_correctness():
    workload=dict(operator="fft",precision="fp32",logN=8,batch=4,placement="out-of-place",normalization="none")
    device=dict(device="A100",compute_capability="8.0")
    sample={**workload,**device,"kernel_ms":"0.1","correct":"1"}
    acceptance.verify_sample(workload,sample,device)
    with pytest.raises(ValueError,match="semantics differ: batch"):
        acceptance.verify_sample(workload,dict(sample,batch=3),device)
    with pytest.raises(ValueError,match="correctness"):
        acceptance.verify_sample(workload,dict(sample,correct="0"),device)
    with pytest.raises(ValueError,match="hardware"):
        acceptance.verify_sample(workload,dict(sample,device="V100"),device)


def test_matrix_is_unique_semantic_only_and_finite():
    document = acceptance.read(Path(__file__).resolve().parents[1] / "config/research_comprehensive_workloads.json")
    cells = document["workloads"]
    assert len(cells) == len({acceptance.cell_key(w) for w in cells}) == 72
    assert {w["operator"] for w in cells} == {"fft", "ntt", "fwht", "subset-zeta", "superset-zeta", "xor-zeta", "structured-2x2"}
    for workload in cells:
        assert not {"backend", "mapping_json", "compute_unit", "local_stages", "fft_core"}.intersection(workload)
        assert workload["batch"] * (1 << workload["logN"]) <= 1 << 24
        if workload["operator"] == "ntt":
            assert (int(workload["modulus"]) - 1) % (1 << workload["logN"]) == 0


def test_comparison_resume_rejects_changed_external_build():
    document = {}
    acceptance.bind_comparison_identity(document, {"vkfft": "original"})
    acceptance.bind_comparison_identity(document, {"vkfft": "original"})
    with pytest.raises(ValueError, match="baseline binaries"):
        acceptance.bind_comparison_identity(document, {"vkfft": "rebuilt"})


def test_capability_gaps_are_not_measured_or_confused_with_runtime_failure():
    row = dict(name="candidate", status="unavailable", correct=False, samples=[],
               errors=["error: bit-reversed output is currently supported only by compact-stage"])
    evidence = acceptance.capability_rejections([row])
    assert evidence["attempted_candidates"] == 1
    assert acceptance.capability_rejections([]) is None
    for changed in [dict(errors=["CUDA error: an illegal memory access was encountered"]),
                    dict(errors=["correctness failed"]), dict(samples=[dict(correct="0")]),
                    dict(status="measured", correct=True)]:
        assert acceptance.capability_rejections([{**row, **changed}]) is None
    document = dict(status="complete-with-unavailable-cells", limitations=[], cells=[
        dict(id="missing", workload=dict(operator="ntt"), status="unavailable", unavailability=evidence),
        dict(id="measured", workload=dict(operator="fwht"), status="complete", measurements=[]),
    ])
    summary = acceptance.summarize(document)
    assert summary["completed_cells"] == 1 and summary["expected_cells"] == 2
    assert summary["all_workloads_measured"] is False
    assert summary["unavailable_cells"][0]["id"] == "missing"


def test_keyboard_interrupt_preserves_resumable_status(monkeypatch, tmp_path):
    args = acceptance.SimpleNamespace(mode="search", output_dir=tmp_path)
    document = dict(status="prepared",registry_path=str(tmp_path / "registry.json"),limitations=[],
                    cells=[dict(status="pending", measurements=[])])
    monkeypatch.setattr(acceptance,"parse_args",lambda argv:args)
    monkeypatch.setattr(acceptance,"prepare",lambda args:document)
    monkeypatch.setenv("CUBUTTERFLY_REGISTRY", "previous")
    def interrupt():
        raise KeyboardInterrupt()
    monkeypatch.setattr(acceptance,"require_exclusive_gpu",interrupt)
    with pytest.raises(KeyboardInterrupt):
        acceptance.main([])
    saved = acceptance.read(tmp_path / "acceptance.json")
    assert saved["status"] == "interrupted" and saved["last_error"] == "KeyboardInterrupt"
    assert saved["cells"] == document["cells"]
    assert saved["code_revisions"][0]["stage_service_model.py"]


@pytest.mark.parametrize("continue_unavailable", [False, True])
def test_capability_gap_continuation_preserves_gap_and_visits_later_cells(monkeypatch, tmp_path, continue_unavailable):
    args = acceptance.SimpleNamespace(
        mode="search", output_dir=tmp_path, build_dir=tmp_path, continue_unavailable=continue_unavailable,
        search_budget=16, finalists=3, trials=3, warmup=100, repeat=100, verify_batches=0,
        seed_budget=4, mapping_seeds=[])
    document = dict(status="prepared", registry_path=str(tmp_path / "registry.json"), scope="test",
                    limitations=[], cells=[dict(id=str(i), workload=dict(operator="ntt", logN=n),
                                               status="pending", measurements=[]) for i, n in enumerate((16, 20))])
    acceptance.write(tmp_path / "calibration_device.json", dict(stage_service={}))
    monkeypatch.setattr(acceptance, "parse_args", lambda argv: args)
    monkeypatch.setattr(acceptance, "prepare", lambda args: document)
    monkeypatch.setattr(acceptance, "require_exclusive_gpu", lambda: None)
    calls = []
    def calibrate(config, output, cufftdx):
        calls.append(output.parent.name)
        row = (dict(name="unsupported", status="unavailable", correct=False, samples=[],
                    errors=["error: bit-reversed output is currently supported only by compact-stage"])
               if output.parent.name == "0" else dict(name="valid", status="measured", correct=True))
        return dict(candidates=[row])
    monkeypatch.setattr(acceptance.calibration, "run_operator_calibration", calibrate)
    monkeypatch.setattr(acceptance, "confirmed_services", lambda rows, *args: (rows, []))
    monkeypatch.setattr(acceptance, "ranking", lambda *args: {})
    monkeypatch.setattr(acceptance, "select_points", lambda value: [r for r in value["candidates"] if r["correct"]])
    monkeypatch.setattr(acceptance, "promote", lambda *args: None)
    monkeypatch.setattr(acceptance, "verify", lambda *args: dict(passed=True))
    if not continue_unavailable:
        with pytest.raises(ValueError, match="expected one confirmed"):
            acceptance.main([])
        assert calls == ["0"] and document["status"] == "interrupted"
    else:
        assert acceptance.main([]) == 0
        assert calls == ["0", "1"]
        assert document["status"] == "complete-with-unavailable-cells"
        assert [c["status"] for c in document["cells"]] == ["unavailable", "search-complete"]


def test_interrupted_pair_resumes_only_missing_trials(monkeypatch, tmp_path):
    workload = dict(operator="fft",precision="fp32",logN=8,batch=1,placement="out-of-place")
    device = dict(device="A100", compute_capability="8.0")
    sample = dict(workload, **device, kernel_ms="0.1", correct="1", mapping_json='{"backend":"shared-iterative"}',
                  selected_implementation="chosen", runtime_fingerprint="frozen")
    cell = dict(workload=workload, status="comparing", selected=dict(name="chosen", samples=[sample]),
                measurements=[dict(name="cuButterfly",trial=0,sample=sample)])
    args = acceptance.SimpleNamespace(build_dir=tmp_path,vkfft_binary=None,fht_python=None,gpuntt_binary=None,
                                     trials=2,warmup=100,repeat=100,verify_batches=0)
    import research_baselines
    monkeypatch.setattr(research_baselines, "available_baselines", lambda *a: dict(available=[dict(name="cuFFT")],unavailable=[]))
    calls = []
    def baseline(*args):
        calls.append("cuFFT")
        return dict(sample=sample,correct=True)
    monkeypatch.setattr(research_baselines, "run_baseline", baseline)
    monkeypatch.setattr(acceptance, "require_exclusive_gpu", lambda: None)
    monkeypatch.setattr(acceptance, "digest", lambda p: "binary")
    def run(command):
        calls.append("cuButterfly")
        import csv, io
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=sample)
        writer.writeheader()
        writer.writerow(sample)
        return acceptance.SimpleNamespace(stdout=output.getvalue())
    monkeypatch.setattr(acceptance, "run", run)
    acceptance.compare_cell(cell,args,dict(identity=dict(device=device)),lambda: None)
    assert calls == ["cuFFT", "cuFFT", "cuButterfly"]
    assert len(cell["measurements"]) == 4 and cell["status"] == "complete"
