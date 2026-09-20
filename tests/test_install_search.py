import copy
import pathlib
import sys
import json
import csv
import io
import pytest
from types import SimpleNamespace

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
from calibration_space import online_candidates, stratified_candidates, candidate_id, command_for, register_tile_candidates
from local_selector_data import semantic_key, write_header, mapping_statements
from fit_local_cost_model import feature_vector, FEATURE_NAMES
from calibration_space import butterfly_candidates


def test_incumbent_confirmation_resumes_individual_trials(tmp_path,monkeypatch):
    import calibrate_local_hardware as calibration
    for name in ("cubutterfly_bench","cuntt_bench"): (tmp_path/name).write_bytes(b"trial-checkpoint")
    monkeypatch.setattr(calibration,"incumbent_commands",lambda args:iter([("incumbent",["bench"])]))
    monkeypatch.setattr(calibration,"run_compiled_search",lambda *args:[])
    measured=[]; exclusive_calls=[]
    def run(command,check=True):
        measured.append(command)
        return SimpleNamespace(returncode=0,stderr="",stdout=
            "operator,precision,logN,batch,backend,correct,kernel_ms,warmup,repeat,mapping_json,runtime_fingerprint\n"
            "fft,fp32,12,2,shared-iterative,1,0.01,3,5,{},test\n")
    def exclusive():
        exclusive_calls.append(1)
        if len(exclusive_calls)==3: raise RuntimeError("GPU became busy")
    monkeypatch.setattr(calibration,"run",run)
    monkeypatch.setattr(calibration,"require_exclusive_gpu",exclusive)
    args=SimpleNamespace(build_dir=tmp_path,resume_search=False,search_only=True,
                         operator_trials=3,operator_warmup=3,operator_repeat=5)
    output=tmp_path/"operator_calibration.json"
    with pytest.raises(RuntimeError,match="GPU became busy"):
        calibration.run_operator_calibration(args,output,True)
    partial=json.loads(output.read_text())[0]
    assert partial["status"]=="screened" and partial["correct"] and partial["trials"]==1
    args.resume_search=True
    monkeypatch.setattr(calibration,"require_exclusive_gpu",lambda:None)
    calibration.run_operator_calibration(args,output,True)
    confirmed=json.loads(output.read_text())[0]
    assert confirmed["status"]=="measured" and confirmed["trials"]==3 and len(measured)==3


def test_search_keeps_completed_screens_when_gpu_becomes_busy(tmp_path, monkeypatch):
    import calibrate_local_hardware as calibration
    for name in ("cubutterfly_bench","cuntt_bench"): (tmp_path/name).write_bytes(b"checkpoint-test")
    workload=tmp_path/"workloads.json"
    workload.write_text(json.dumps(dict(schema="cubutterfly-install-search-v1",scope="checkpoint test",
        workloads=[dict(operator="fft",precision="fp32",logN=12,batch=2)])))
    mappings=[dict(schema_version=1,kind="butterfly",backend="shared-iterative",fft_core="scalar",
                   tile_threads=128,stage_partition=[s,12-s]) for s in (3,4,5)]
    def run(command,check=True):
        if "--list-processing-units" in command: output="precision,logN,threads,ept\n"
        elif "--list-register-tile-mappings" in command: output="logN,local_stages,prefix_threads,prefix_ept,suffix_threads,suffix_ept\n"
        elif "--list-design-points" in command: output="\n".join(map(json.dumps,mappings))
        else:
            mapping=command[command.index("--mapping-json")+1]
            row=dict(operator="fft",precision="fp32",logN=12,batch=2,backend="shared-iterative",fft_core="scalar",
                correct=1,kernel_ms=.01,mapping_json=mapping,warmup=3,repeat=5)
            buffer=io.StringIO(); writer=csv.DictWriter(buffer,fieldnames=list(row));writer.writeheader();writer.writerow(row)
            output=buffer.getvalue()
        return SimpleNamespace(returncode=0,stdout=output,stderr="")
    calls=[]
    def exclusive():
        calls.append(1)
        if len(calls)==3: raise RuntimeError("GPU became busy")
    monkeypatch.setattr(calibration,"run",run)
    monkeypatch.setattr(calibration,"require_exclusive_gpu",exclusive)
    args=SimpleNamespace(build_dir=tmp_path,search_workloads=workload,resume_search=False,search_budget=3,
        search_seconds=0,search_finalists=1,operator_warmup=3,operator_repeat=5,operator_trials=2)
    with pytest.raises(RuntimeError,match="GPU became busy"):
        calibration.run_compiled_search(args,tmp_path/"coverage.json")
    saved=json.loads((tmp_path/"search_measurements.json").read_text())
    assert len(saved)==1 and saved[0]["correct"] and saved[0]["status"]=="screened"
    # A later workload may already be cached when retrying an earlier cell.
    # A second interruption must preserve that evidence as well.
    future=dict(saved[0],name="later-workload-confirmed",status="measured")
    (tmp_path/"search_measurements.json").write_text(json.dumps(saved+[future]))
    args.resume_search=True
    calls.clear()
    with pytest.raises(RuntimeError,match="GPU became busy"):
        calibration.run_compiled_search(args,tmp_path/"coverage.json")
    resumed=json.loads((tmp_path/"search_measurements.json").read_text())
    assert any(row["name"]==future["name"] and row["status"]=="measured" for row in resumed)
    monkeypatch.setattr(calibration,"require_exclusive_gpu",lambda:None)
    result=calibration.run_compiled_search(args,tmp_path/"coverage.json")
    assert len(result)==3 and sum(r["status"]=="measured" for r in result)==1


def test_search_confirmation_is_not_promoted_until_all_trials_finish(tmp_path,monkeypatch):
    import calibrate_local_hardware as calibration
    for name in ("cubutterfly_bench","cuntt_bench"): (tmp_path/name).write_bytes(b"confirmation-resume")
    point=dict(operator="fft",precision="fp32",logN=12,batch=2,backend="shared-iterative",fft_core="scalar",mapping_json="{}")
    workload=tmp_path/"workloads.json"
    workload.write_text(json.dumps(dict(schema="cubutterfly-install-search-v1",scope="test",workloads=[point])))
    monkeypatch.setattr(calibration,"runtime_candidates",lambda *args:[point])
    measurements=[]; checks=[]
    def run(command,check=True):
        if "--list-processing-units" in command: output="precision,logN,threads,ept\n"
        elif "--list-register-tile-mappings" in command: output="logN,local_stages\n"
        else:
            measurements.append(command)
            row=dict(point,correct=1,kernel_ms=.01,warmup=3,repeat=5)
            buffer=io.StringIO(); writer=csv.DictWriter(buffer,fieldnames=list(row));writer.writeheader();writer.writerow(row)
            output=buffer.getvalue()
        return SimpleNamespace(returncode=0,stdout=output,stderr="")
    def exclusive():
        checks.append(1)
        if len(checks)==5: raise RuntimeError("GPU became busy")
    monkeypatch.setattr(calibration,"run",run)
    monkeypatch.setattr(calibration,"require_exclusive_gpu",exclusive)
    args=SimpleNamespace(build_dir=tmp_path,search_workloads=workload,resume_search=False,search_budget=1,
        search_seconds=0,search_finalists=1,operator_warmup=3,operator_repeat=5,operator_trials=3)
    with pytest.raises(RuntimeError,match="GPU became busy"):
        calibration.run_compiled_search(args,tmp_path/"coverage.json")
    partial=json.loads((tmp_path/"search_measurements.json").read_text())[0]
    assert partial["status"]=="screened" and partial["trials"]==2
    monkeypatch.setattr(calibration,"require_exclusive_gpu",lambda:None)
    args.resume_search=True
    result=calibration.run_compiled_search(args,tmp_path/"coverage.json")
    assert result[0]["status"]=="measured" and result[0]["trials"]==3 and len(measurements)==3


def test_install_search_consumes_library_mappings_and_fits_before_ranking(tmp_path, monkeypatch):
    import calibrate_local_hardware as calibration
    for binary in ("cubutterfly_bench","cuntt_bench"): (tmp_path/binary).write_bytes(b"test-binary")
    workload=tmp_path/"workloads.json"
    workload.write_text(json.dumps(dict(schema="cubutterfly-install-search-v1",scope="unit test",
        workloads=[dict(operator="fft",precision="fp32",logN=12,batch=2)])))
    mappings=[dict(schema_version=1,kind="butterfly",backend="shared-iterative",fft_core="scalar",
                   tile_threads=t,local_stages=s,stage_partition=[s,12-s])
              for s in (3,4,5,6) for t in (64,128,256)]
    def run(command,check=True):
        if "--list-processing-units" in command: output="precision,logN,threads,ept\n"
        elif "--list-register-tile-mappings" in command: output="logN,local_stages,prefix_threads,prefix_ept,suffix_threads,suffix_ept\n"
        elif "--list-design-points" in command: output="\n".join(map(json.dumps,mappings))
        else:
            mapping=json.loads(command[command.index("--mapping-json")+1])
            row=dict(operator="fft",precision="fp32",logN="12",batch="2",backend="shared-iterative",fft_core="scalar",
                     correct="1",kernel_ms=str(.01+mapping["tile_threads"]*.0001),mapping_json=json.dumps(mapping),
                     warmup=command[command.index("--warmup")+1],repeat=command[command.index("--repeat")+1])
            output_io=io.StringIO(); writer=csv.DictWriter(output_io,fieldnames=list(row)); writer.writeheader();writer.writerow(row)
            output=output_io.getvalue()
        return SimpleNamespace(returncode=0,stdout=output,stderr="")
    monkeypatch.setattr(calibration,"run",run)
    monkeypatch.setattr(calibration,"require_exclusive_gpu",lambda:None)
    args=SimpleNamespace(build_dir=tmp_path,search_workloads=workload,resume_search=False,search_budget=12,
                         search_seconds=0,search_finalists=2,operator_warmup=3,operator_repeat=5,operator_trials=2)
    audit_path=tmp_path/"coverage.json"
    rows=calibration.run_compiled_search(args,audit_path)
    audit=json.loads(audit_path.read_text())["workloads"][0]
    assert audit["candidate_source"]=="linked-library" and audit["enumerated"]==12
    assert audit["model_updates"][0]["training_rows"]==8
    update = audit["model_updates"][0]
    assert len(update["training_candidate_ids"]) == 8
    assert len(update["coefficients"]) == len(update["feature_means"]) == len(update["feature_scales"]) == len(FEATURE_NAMES)
    assert update["feature_version"] == calibration.FEATURE_VERSION
    assert sum(row["status"]=="measured" for row in rows)==2 and audit["budget_omitted"]==0
    assert audit["complete"]
    args.resume_search=True
    def no_measurements():
        raise AssertionError("completed cell must not be measured again on resume")
    monkeypatch.setattr(calibration,"require_exclusive_gpu",no_measurements)
    resumed=calibration.run_compiled_search(args,audit_path)
    assert len(resumed)==len(rows) and {r["name"] for r in resumed}=={r["name"] for r in rows}
    previous = json.loads(audit_path.read_text())
    assert previous["search_protocol"]["model_update_policy_version"] == calibration.MODEL_UPDATE_POLICY_VERSION
    previous["search_protocol"]["model_update_policy_version"] = "legacy-correct-training-change"
    audit_path.write_text(json.dumps(previous))
    fit_calls = []
    original_fit = calibration.fit

    def counted_fit(training, ridge):
        fit_calls.append(len(training))
        return original_fit(training, ridge)

    monkeypatch.setattr(calibration, "fit", counted_fit)
    resumed = calibration.run_compiled_search(args, audit_path)
    assert len(resumed) == len(rows)
    assert fit_calls == [8, 9, 10, 11]
    current = json.loads(audit_path.read_text())
    assert current["search_protocol"]["model_update_policy_version"] == calibration.MODEL_UPDATE_POLICY_VERSION


def test_model_refits_before_ranking_after_new_valid_feedback(tmp_path, monkeypatch):
    import calibrate_local_hardware as calibration
    for binary in ("cubutterfly_bench", "cuntt_bench"):
        (tmp_path / binary).write_bytes(b"model-feedback")
    workload = tmp_path / "workloads.json"
    workload.write_text(json.dumps(dict(
        schema="cubutterfly-install-search-v1", scope="model feedback",
        workloads=[dict(operator="fft", precision="fp32", logN=12, batch=2)])))
    mappings = [dict(schema_version=1, kind="butterfly", backend="shared-iterative", fft_core="scalar",
                     tile_threads=64 + index, local_stages=6, stage_partition=[6, 6])
                for index in range(11)]
    fit_calls = []
    predict_calls = []

    def fake_fit(training, ridge):
        fit_calls.append([row["median_kernel_ms"] for row in training])
        return ((len(training),), (), ())

    def fake_predict(sample, coefficients, means, scales):
        predict_calls.append(coefficients[0])
        return 0.0

    measured = 0

    def run(command, check=True):
        nonlocal measured
        if "--list-processing-units" in command:
            output = "precision,logN,threads,ept\n"
        elif "--list-register-tile-mappings" in command:
            output = "logN,local_stages,prefix_threads,prefix_ept,suffix_threads,suffix_ept\n"
        elif "--list-design-points" in command:
            output = "\n".join(map(json.dumps, mappings))
        else:
            mapping = json.loads(command[command.index("--mapping-json") + 1])
            measured += 1
            row = dict(operator="fft", precision="fp32", logN="12", batch="2",
                       backend="shared-iterative", fft_core="scalar", correct="1",
                       kernel_ms=str(100.0 if measured == 9 else 1.0),
                       mapping_json=json.dumps(mapping), warmup="3", repeat="5")
            output_io = io.StringIO()
            writer = csv.DictWriter(output_io, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)
            output = output_io.getvalue()
        return SimpleNamespace(returncode=0, stdout=output, stderr="")

    monkeypatch.setattr(calibration, "run", run)
    monkeypatch.setattr(calibration, "require_exclusive_gpu", lambda: None)
    monkeypatch.setattr(calibration, "fit", fake_fit)
    monkeypatch.setattr(calibration, "predict", fake_predict)
    monkeypatch.setattr(calibration, "predicted_sample", lambda point: point)
    args = SimpleNamespace(build_dir=tmp_path, search_workloads=workload, resume_search=False,
                           search_budget=10, search_seconds=0, search_finalists=1,
                           operator_warmup=3, operator_repeat=5, operator_trials=1)

    audit_path = tmp_path / "coverage.json"
    calibration.run_compiled_search(args, audit_path)

    assert [len(rows) for rows in fit_calls] == [8, 9]
    assert 100.0 in fit_calls[1]
    assert predict_calls and set(predict_calls) == {9}
    audit = json.loads(audit_path.read_text())
    assert audit["search_protocol"]["model_update_policy_version"] == \
        calibration.MODEL_UPDATE_POLICY_VERSION


def test_failed_screen_does_not_trigger_redundant_model_refit(tmp_path, monkeypatch):
    import calibrate_local_hardware as calibration
    for binary in ("cubutterfly_bench", "cuntt_bench"):
        (tmp_path / binary).write_bytes(b"model-invalid-feedback")
    workload = tmp_path / "workloads.json"
    workload.write_text(json.dumps(dict(
        schema="cubutterfly-install-search-v1", scope="model invalid feedback",
        workloads=[dict(operator="fft", precision="fp32", logN=12, batch=2)])))
    mappings = [dict(schema_version=1, kind="butterfly", backend="shared-iterative", fft_core="scalar",
                     tile_threads=64 + index, local_stages=6, stage_partition=[6, 6])
                for index in range(12)]
    fit_calls = []

    def fake_fit(training, ridge):
        fit_calls.append(len(training))
        return ((len(training),), (), ())

    measured = 0

    def run(command, check=True):
        nonlocal measured
        if "--list-processing-units" in command:
            output = "precision,logN,threads,ept\n"
        elif "--list-register-tile-mappings" in command:
            output = "logN,local_stages,prefix_threads,prefix_ept,suffix_threads,suffix_ept\n"
        elif "--list-design-points" in command:
            output = "\n".join(map(json.dumps, mappings))
        else:
            mapping = json.loads(command[command.index("--mapping-json") + 1])
            measured += 1
            correct = "0" if measured == 10 else "1"
            row = dict(operator="fft", precision="fp32", logN="12", batch="2",
                       backend="shared-iterative", fft_core="scalar", correct=correct,
                       kernel_ms="1.0", mapping_json=json.dumps(mapping), warmup="3", repeat="5")
            output_io = io.StringIO()
            writer = csv.DictWriter(output_io, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)
            output = output_io.getvalue()
        return SimpleNamespace(returncode=0, stdout=output, stderr="")

    monkeypatch.setattr(calibration, "run", run)
    monkeypatch.setattr(calibration, "require_exclusive_gpu", lambda: None)
    monkeypatch.setattr(calibration, "fit", fake_fit)
    monkeypatch.setattr(calibration, "predict", lambda sample, coefficients, means, scales: 0.0)
    monkeypatch.setattr(calibration, "predicted_sample", lambda point: point)
    args = SimpleNamespace(build_dir=tmp_path, search_workloads=workload, resume_search=False,
                           search_budget=11, search_seconds=0, search_finalists=1,
                           operator_warmup=3, operator_repeat=5, operator_trials=1)

    calibration.run_compiled_search(args, tmp_path / "coverage.json")

    assert fit_calls == [8, 9]


@pytest.mark.parametrize("change_mapping", [False, True])
def test_historical_seed_has_reserved_budget_and_requires_target_confirmation(tmp_path, monkeypatch, change_mapping):
    import calibrate_local_hardware as calibration
    monkeypatch.setenv("CUBUTTERFLY_REGISTRY", str(tmp_path/"empty-registry.json"))
    for binary in ("cubutterfly_bench", "cuntt_bench"):
        (tmp_path/binary).write_bytes(b"seed-target-build")
    workload = dict(operator="fft", precision="fp32", logN=12, batch=8, placement="in-place")
    workload_path = tmp_path/"workloads.json"
    workload_path.write_text(json.dumps(dict(schema="cubutterfly-install-search-v1", scope="seed test", workloads=[workload])))
    seed_mapping = dict(schema_version=1, kind="butterfly", backend="shared-iterative", fft_core="scalar",
                        tile_threads=128, stage_partition=[6, 6])
    library_mapping = dict(seed_mapping, tile_threads=64)
    seed_path = tmp_path/"seeds.json"
    seed_path.write_text(json.dumps(dict(schema="cubutterfly-mapping-seeds-v1", seeds=[dict(
        semantics=dict(workload, batch=2, placement="out-of-place"), mapping=seed_mapping)])))
    timed = []
    def run(command, check=True):
        if "--list-processing-units" in command:
            output = "precision,logN,threads,ept\n"
        elif "--list-register-tile-mappings" in command:
            output = "logN,local_stages\n"
        elif "--list-design-points" in command:
            output = json.dumps(library_mapping)
        else:
            mapping = json.loads(command[command.index("--mapping-json")+1])
            timed.append((mapping, command))
            assert command[command.index("--batch")+1] == "8"
            assert command[command.index("--placement")+1] == "in-place"
            resolved = dict(mapping, tile_threads=256) if change_mapping and mapping == seed_mapping else mapping
            row = dict(workload, backend=mapping["backend"], fft_core="scalar", correct=1,
                       kernel_ms=.01 if mapping == seed_mapping else 1,
                       warmup=command[command.index("--warmup")+1], repeat=command[command.index("--repeat")+1],
                       mapping_json=json.dumps(resolved), runtime_fingerprint="target-build")
            buffer = io.StringIO(); writer = csv.DictWriter(buffer, fieldnames=list(row))
            writer.writeheader(); writer.writerow(row); output = buffer.getvalue()
        return SimpleNamespace(returncode=0, stdout=output, stderr="")
    monkeypatch.setattr(calibration, "run", run)
    monkeypatch.setattr(calibration, "require_exclusive_gpu", lambda:None)
    args = SimpleNamespace(build_dir=tmp_path, search_workloads=workload_path, resume_search=False,
        seed_budget=1, mapping_seeds=[seed_path], search_budget=1, search_seconds=0, search_finalists=1,
        operator_warmup=12, operator_repeat=20, operator_trials=3)
    audit_path = tmp_path/"coverage.json"
    records = calibration.run_compiled_search(args, audit_path)
    cell = json.loads(audit_path.read_text())["workloads"][0]
    assert cell["screened"] == cell["requested_screened_count"] == 2
    assert cell["seed_coverage"]["attempted"] == 1
    assert timed[0][0] == seed_mapping
    assert {p["selection_reason"] for p in cell["candidates"]} == {"historical-seed", "stratified-exploration"}
    historical = next(r for r in records if json.loads(r["configuration"]["mapping_json"]) == seed_mapping)
    if change_mapping:
        assert historical["status"] == "mapping-mismatch" and not historical["correct"]
        assert cell["seed_coverage"]["confirmed"] == 0
    else:
        assert historical["status"] == "measured" and historical["trials"] == 3
        assert historical["median_kernel_ms"] == .01 and cell["seed_coverage"]["confirmed"] == 1
        assert all(s["runtime_fingerprint"] == "target-build" for s in historical["samples"])
    args.resume_search = True
    count = len(timed)
    calibration.run_compiled_search(args, audit_path)
    assert len(timed) == count


def test_compiled_product_preserves_asymmetric_axes_and_budget_coverage():
    units = [{"precision": "fp64", "logN": s, "threads": t, "ept": e}
             for s in (7, 8, 9) for t, e in ((128, 4), (256, 8))]
    points = list(online_candidates(units, {"precision": "fp64", "logN": 16, "batch": 64}))
    assert len(points) == 3 * 2 * 2 * 2 * 2 * 7
    assert {p["local_stages"] for p in points} == {7, 8, 9}
    assert any(p["prefix_threads"] != p["suffix_threads"] and p["prefix_ept"] != p["suffix_ept"] for p in points)
    screened = stratified_candidates(points, 24)
    assert {(p["local_stages"], p["cross_twiddle"], p["shared_layout"]) for p in screened} == {
        (s, t, l) for s in (7, 8, 9) for t in ("table", "recurrence") for l in ("linear", "xor-swizzle")}
    assert len(stratified_candidates(points, 0)) == len(points)
    assert len({candidate_id(p) for p in points}) == len(points)


def test_strategy_representatives_precede_unit_pair_budget_cutoff():
    strategies = [
        ("native", "linear"),
        ("cufftdx-thread", "linear"),
        ("native", "xor"),
        ("cufftdx-thread", "xor"),
    ]
    points = []
    for unit in range(12):
        for codelet, layout in strategies:
            mapping = {
                "backend": "online-reorder", "fft_core": "register-tile",
                "local_stages": 6, "cross_twiddle": "table",
                "shared_layout": "linear", "stage_overlap": 0,
                "prefix_threads": 32 * (unit + 1), "prefix_ept": 8,
                "suffix_threads": 64, "suffix_ept": 8,
                "prefix_codelet": codelet, "prefix_shared_layout": layout,
            }
            points.append({
                "backend": "online-reorder", "fft_core": "register-tile",
                "prefix_threads": mapping["prefix_threads"], "prefix_ept": 8,
                "suffix_threads": 64, "suffix_ept": 8,
                "mapping_json": json.dumps(mapping),
            })
    selected = stratified_candidates(points, len(strategies))
    observed = {
        (json.loads(point["mapping_json"]).get("prefix_codelet", "native"),
         json.loads(point["mapping_json"]).get("prefix_shared_layout", "linear"))
        for point in selected
    }
    assert observed == set(strategies)
    assert ("native", "linear") == (
        json.loads(selected[0]["mapping_json"]).get("prefix_codelet", "native"),
        json.loads(selected[0]["mapping_json"]).get("prefix_shared_layout", "linear"),
    )


def test_factor_exploration_keeps_a_share_beyond_one_seed():
    points=[]
    for core in ("cufftdx-block","register-tile"):
        for groups in (2,3):
            for depth in (0,1,2):
                mapping=dict(factor_partition=[6]*groups,prefetch_depth=depth,data_tiles_per_cta=4)
                points.append(dict(backend="factor-streamed",fft_core=core,mapping_json=json.dumps(mapping)))
    points += [dict(backend="online-reorder",fft_core="register-tile",prefix_threads=32*i,
                    prefix_ept=8,suffix_threads=128,suffix_ept=8) for i in range(1,50)]
    selected=stratified_candidates(points,9)
    assert sum(p["backend"]=="online-reorder" for p in selected)==3
    for core in ("cufftdx-block","register-tile"):
        subset=[p for p in selected if p["backend"]=="factor-streamed" and p["fft_core"]==core]
        assert {json.loads(p["mapping_json"])["prefetch_depth"] for p in subset}=={0,1,2}


def test_embedded_selector_preserves_complete_factor_mapping():
    mapping=dict(backend="factor-streamed",fft_core="register-tile",stage_partition=[24],
                 factor_partition=[8,8,8],factor_ept=16,factor_columns=8,data_tiles_per_cta=4,prefetch_depth=1)
    statements=mapping_statements(dict(mapping_json=json.dumps(mapping)))
    assert len(statements)==1 and "::cubutterfly::apply_serialized_mapping" in statements[0]
    assert 'factor_partition' in statements[0] and 'prefetch_depth' in statements[0]


def test_other_operators_keep_local_stage_and_pipeline_axes():
    for op in ("fwht", "subset-zeta", "superset-zeta", "structured-2x2"):
        points = list(butterfly_candidates(dict(operator=op, precision="fp32", logN=12, batch=4)))
        assert {p["local_stages"] for p in points if p["backend"] == "online-reorder"} == set(range(5, 11))
        assert {p["local_stages"] for p in points if p["backend"] == "shared-iterative"} == set(range(1, 13))
        small = list(butterfly_candidates(dict(operator=op, precision="fp32", logN=8, batch=4)))
        assert {p["stage_space"] for p in small if p["backend"] == "stage-pipeline"} == {1, 2, 4, 8}


def test_uncompiled_dimensions_never_become_executable_candidates():
    units = [{"precision": "fp32", "logN": s, "threads": 256, "ept": 16} for s in (8, 9, 10, 12)]
    points = list(online_candidates(units, {"precision": "fp32", "logN": 20, "batch": 16}))
    assert {p["local_stages"] for p in points} == {8, 10, 12}
    assert not list(online_candidates(units, {"precision": "fp64", "logN": 16}))


def test_fp32_swizzle_is_an_executable_prefix_choice():
    units = [{"precision": "fp32", "logN": s, "threads": 256, "ept": 16}
             for s in (8, 12)]
    points = list(online_candidates(units, {"precision": "fp32", "logN": 20, "batch": 5}))
    assert {p["shared_layout"] for p in points if p["local_stages"] == 8} == {"linear", "xor-swizzle"}
    assert {p["shared_layout"] for p in points if p["local_stages"] == 12} == {"linear"}
    for layout in ("linear", "xor-swizzle"):
        assert any(p["shared_layout"] == layout and p["stage_overlap"] for p in points)
    assert len({candidate_id(p) for p in points}) == len(points)


def test_semantic_key_separates_numeric_and_layout_contracts():
    sample = dict(operator="fft", precision="fp64", logN="16", batch="64", direction="forward",
                  normalization="none", accumulation="native", placement="out-of-place", element_stride="1",
                  batch_stride="65536", stage_matrices="", modulus="0")
    for field,value in dict(precision="fp32",direction="inverse",accumulation="fp32",placement="in-place",
                            element_stride="2",batch_stride="131072",stage_matrices="1:0:0:1",modulus="17").items():
        assert semantic_key(sample) != semantic_key(dict(sample, **{field:value}))
    assert semantic_key(sample)==semantic_key(dict(sample,normalization="inverse"))
    inverse=dict(sample,direction="inverse")
    assert semantic_key(inverse)!=semantic_key(dict(inverse,normalization="inverse"))
    omitted={k:v for k,v in sample.items() if k not in ("modulus","accumulation","direction","batch_stride")}
    assert semantic_key(sample)==semantic_key(omitted)


def test_selector_promotes_arbitrary_measured_names_but_excludes_baselines(tmp_path):
    sample = dict(operator="fft", precision="fp64", placement="out-of-place", logN="8", batch="64",
                  backend="temporal-tile", fft_core="cufftdx-block", compute_unit="auto", kernel_ms="0.2")
    row = dict(name="arbitrary-new-unit-without-cpp-branch", status="measured", correct=True,
               median_kernel_ms=0.2, samples=[sample])
    external = copy.deepcopy(row)
    external.update(name="reference", median_kernel_ms=0.01)
    external["samples"][0]["backend"] = "cufft"
    failed = dict(row, name="bad", correct=False, median_kernel_ms=0.001)
    screened = dict(row, name="screen-only", status="screened", median_kernel_ms=0.001)
    output = tmp_path / "selector.hpp"
    count = write_header(dict(device="test", compute_capability="8.0"),
                         {"candidates": [external, row, failed, screened]}, output)
    assert count == 1
    header = output.read_text()
    assert row["name"] in header
    assert '"reference"' not in header and '"bad"' not in header and '"screen-only"' not in header
    assert 'config.fft_core = parse_fft_core("cufftdx-block")' in header


def test_cost_bytes_respect_value_width_and_physical_groups():
    sample = dict(operator="fft", precision="fp32", N="65536", batch="64", logN="16",
                  decomposition_count="2", execution_group_count="2")
    index = FEATURE_NAMES.index("bytes_million")
    fp32 = feature_vector(sample)[index]
    assert feature_vector(dict(sample, precision="fp64"))[index] == 2 * fp32
    assert feature_vector(dict(sample, operator="fwht"))[index] == fp32 / 2
    assert feature_vector(dict(sample, execution_group_count="1"))[index] == fp32 / 2


def test_cost_schedule_features_preserve_bulk_and_tile_semantics():
    sample = dict(operator="fft", precision="fp32", logN="20", N="1048576", batch="16",
                  execution_group_count="2", batch_tile_count="4")
    bulk, pipelined = feature_vector(sample), feature_vector(dict(sample, stage_overlap="1"))
    assert bulk[FEATURE_NAMES.index("schedule_tiles")] == 1
    assert bulk[FEATURE_NAMES.index("batch_tile_fraction")] == 1
    assert pipelined[FEATURE_NAMES.index("schedule_tiles")] == 4
    assert pipelined[FEATURE_NAMES.index("batch_tile_fraction")] == 0.25


def test_inverse_workload_uses_benchmark_flag():
    assert command_for("bench", {"operator": "fft", "direction": "inverse"}) == ["bench", "--operator", "fft", "--inverse"]


def test_public_mapping_axes_are_not_repeated_as_unsupported_cli_flags():
    from calibration_space import mapping_point
    mapping = dict(schema_version=1, backend="online-reorder", fft_core="register-tile",
        stage_partition=[11,11], prefix_units_per_cta=4, suffix_units_per_cta=4,
        prefix_threads=256, prefix_ept=32, suffix_threads=512, suffix_ept=16,
        local_stage_partitions=[[6,5],[]], exchange_chunks=[0,8],
        prefix_codelet="cufftdx-thread", prefix_shared_layout="xor",
        prefix_codelet_lanes=2, stage_overlap=True, batch_tile_count=2)
    workload = dict(operator="fft", precision="fp32", logN=22, batch=3,
                    direction="inverse", normalization="inverse", element_stride=2)
    command = command_for("bench", mapping_point(workload, mapping))
    assert json.loads(command[command.index("--mapping-json")+1]) == mapping
    for unsupported in ("--prefix-units-per-cta", "--suffix-units-per-cta",
                        "--stage-partition", "--local-stage-partitions", "--exchange-chunks",
                        "--prefix-codelet", "--stage-overlap", "--backend"):
        assert unsupported not in command
    assert "--inverse" in command
    assert command[command.index("--element-stride")+1] == "2"
    assert command[command.index("--batch")+1] == "3"
    assert command[command.index("--normalization")+1] == "inverse"


def test_overlap_is_a_schedule_axis_and_survives_selector_replay():
    units = [{"precision": "fp32", "logN": s, "threads": 256, "ept": 16} for s in (8, 12)]
    points = list(online_candidates(units, {"precision": "fp32", "logN": 20, "batch": 5}))
    assert {p["reorder_columns"] for p in points} == {1}
    assert {p["batch_tile_count"] for p in points if p["stage_overlap"]} == {1, 2, 4}
    point = next(p for p in points if p["stage_overlap"])
    assert "--stage-overlap" in command_for("bench", point)
    sample = {k: str(v) for k, v in point.items()}
    sample.update(stages_per_decomposition="8x12", boundary_twiddles="table",
                  boundary_layouts="direct-strided", boundary_residencies="global-scratch")
    assert "config.stage_overlap = true;" in mapping_statements(sample)
    assert any(p["stage_overlap"] == 0 for p in points)
    assert len({candidate_id(p) for p in points}) == len(points)


def test_boundary_serialization_preserves_prefix_layout_and_asymmetric_groups():
    sample = dict(backend="online-reorder", stages_per_decomposition="11x9", segment_threads="256x512",
                  segment_ept="8x4", group_threads="256x512", group_ept="8x4", boundary_twiddles="recurrence",
                  boundary_layouts="prefix-tiled-transpose", boundary_residencies="global-scratch")
    statements = "\n".join(mapping_statements(sample))
    assert 'parse_direct_boundary("prefix-tiled-transpose")' in statements
    assert "256U, 8U" in statements and "512U, 4U" in statements


def test_register_tile_inventory_and_mixed_core_replay():
    mappings=[dict(logN="20",local_stages="10",prefix_threads="256",prefix_ept="32",suffix_threads="256",suffix_ept="16")]
    points=list(register_tile_candidates(mappings,dict(operator="fft",precision="fp32",logN=20,batch=16)))
    assert len(points)==1 and points[0]["fft_core"]=="register-tile"
    assert not list(register_tile_candidates(mappings,dict(precision="fp64",logN=20)))
    sample=dict(backend="online-reorder",fft_core="register-tile",stages_per_decomposition="10x10",
                segment_threads="256x256",segment_ept="32x16",group_threads="256x256",group_ept="32x16",
                segment_cores="register-tile:cufftdx-block",group_cores="register-tile:cufftdx-block",
                boundary_twiddles="recurrence",boundary_layouts="direct-strided",boundary_residencies="global-scratch")
    text="\n".join(mapping_statements(sample))
    assert '{parse_fft_core("register-tile"), config.local_exchange, 256U, 32U}' in text
    assert '{parse_fft_core("cufftdx-block"), config.local_exchange, 256U, 16U}' in text
    other={**points[0],"fft_core":"cufftdx-block"}
    assert {p["fft_core"] for p in stratified_candidates([other,points[0]],2)}=={"cufftdx-block","register-tile"}


def _resolved_butterfly_mapping(tile_threads=128, stage_partition=(6, 6)):
    return {
        "schema_version": 1,
        "kind": "butterfly",
        "backend": "shared-iterative",
        "compute_unit": "radix2",
        "stage_partition": list(stage_partition),
        "execution_group_mappings": [{"core": "scalar"}],
        "fft_core": "scalar",
        "tile_threads": tile_threads,
        "boundaries": [],
        "segment_mappings": [],
    }


def _run_finalist_selection(tmp_path, monkeypatch, candidates, search_finalists):
    import calibrate_local_hardware as calibration

    for binary in ("cubutterfly_bench", "cuntt_bench"):
        (tmp_path / binary).write_bytes(b"finalist-selection-test")
    workload = tmp_path / "workloads.json"
    workload.write_text(json.dumps(dict(
        schema="cubutterfly-install-search-v1", scope="finalist identity",
        workloads=[dict(operator="fft", precision="fp32", logN=12, batch=2)])))
    points = []
    by_alias = {}
    for candidate in candidates:
        point = dict(operator="fft", precision="fp32", logN=12, batch=2,
                     backend="shared-iterative", fft_core="scalar",
                     mapping_json=json.dumps({"candidate_alias": candidate["alias"]}))
        points.append(point)
        by_alias[candidate["alias"]] = candidate
    monkeypatch.setattr(calibration, "runtime_candidates", lambda *args: points)
    monkeypatch.setattr(calibration, "require_exclusive_gpu", lambda: None)

    def run(command, check=True):
        if "--list-processing-units" in command:
            output = "precision,logN,threads,ept\n"
        elif "--list-register-tile-mappings" in command:
            output = "logN,local_stages,prefix_threads,prefix_ept,suffix_threads,suffix_ept\n"
        else:
            requested = json.loads(command[command.index("--mapping-json") + 1])
            candidate = by_alias[requested["candidate_alias"]]
            sample = dict(operator="fft", precision="fp32", logN="12", batch="2",
                          placement="out-of-place", backend="shared-iterative", fft_core="scalar",
                          correct="1", kernel_ms=str(candidate["kernel_ms"]),
                          mapping_json=json.dumps(candidate["resolved_mapping"], sort_keys=True),
                          device=candidate.get("device", "Test GPU"),
                          compute_capability=candidate.get("compute_capability", "8.0"),
                          global_memory_bytes=str(candidate.get("global_memory_bytes", 4096)),
                          runtime_fingerprint=candidate.get("runtime_fingerprint", "runtime-a"),
                          warmup=command[command.index("--warmup") + 1],
                          repeat=command[command.index("--repeat") + 1])
            output_io = io.StringIO()
            writer = csv.DictWriter(output_io, fieldnames=list(sample))
            writer.writeheader()
            writer.writerow(sample)
            output = output_io.getvalue()
        return SimpleNamespace(returncode=0, stdout=output, stderr="")

    monkeypatch.setattr(calibration, "run", run)
    args = SimpleNamespace(build_dir=tmp_path, search_workloads=workload, resume_search=False,
                           search_budget=len(points), search_seconds=0,
                           search_finalists=search_finalists, operator_warmup=3,
                           operator_repeat=5, operator_trials=1)
    records = calibration.run_compiled_search(args, tmp_path / "coverage.json")
    return {candidate["alias"]: candidate_id(point)
            for candidate, point in zip(candidates, points)}, records


def test_finalist_slots_deduplicate_resolved_aliases_but_keep_distinct_mapping(tmp_path, monkeypatch):
    mapping_a = _resolved_butterfly_mapping(tile_threads=128)
    mapping_b = _resolved_butterfly_mapping(tile_threads=256)
    candidates = [
        {"alias": "alias-fast", "resolved_mapping": mapping_a, "kernel_ms": 0.01},
        {"alias": "alias-slow", "resolved_mapping": mapping_a, "kernel_ms": 0.02},
        {"alias": "different-mapping", "resolved_mapping": mapping_b, "kernel_ms": 0.03},
    ]
    names, records = _run_finalist_selection(tmp_path, monkeypatch, candidates, search_finalists=2)
    measured = {row["name"] for row in records if row["status"] == "measured"}
    assert names["alias-fast"] in measured
    assert names["alias-slow"] not in measured
    assert names["different-mapping"] in measured


def test_finalist_identity_does_not_merge_same_mapping_from_other_hardware(tmp_path, monkeypatch):
    mapping = _resolved_butterfly_mapping()
    candidates = [
        {"alias": "target-gpu", "resolved_mapping": mapping, "kernel_ms": 0.01,
         "device": "Test GPU", "runtime_fingerprint": "runtime-a"},
        {"alias": "other-gpu", "resolved_mapping": mapping, "kernel_ms": 0.02,
         "device": "Other GPU", "runtime_fingerprint": "runtime-b"},
    ]
    names, records = _run_finalist_selection(tmp_path, monkeypatch, candidates, search_finalists=2)
    measured = {row["name"] for row in records if row["status"] == "measured"}
    assert measured == {names["target-gpu"], names["other-gpu"]}


def test_finalist_selection_keeps_incomplete_mappings_conservative(tmp_path, monkeypatch):
    candidates = [
        {"alias": "incomplete-a", "resolved_mapping": {}, "kernel_ms": 0.01},
        {"alias": "incomplete-b", "resolved_mapping": {}, "kernel_ms": 0.02},
    ]
    names, records = _run_finalist_selection(tmp_path, monkeypatch, candidates, search_finalists=2)
    measured = {row["name"] for row in records if row["status"] == "measured"}
    assert measured == {names["incomplete-a"], names["incomplete-b"]}
