"""CPU-only checks for the current paper evidence exporter."""

import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


MODULE = Path(__file__).resolve().parents[1] / "paper/tools/export_current_refresh.py"
SPEC = importlib.util.spec_from_file_location("current_refresh", MODULE)
refresh = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(refresh)


def test_summary_keeps_missing_and_separates_cohorts():
    rows = [dict(cohort="main", baseline="a", speedup=4),
            dict(cohort="main", baseline="a", speedup=0.25),
            dict(cohort="main", baseline="a", speedup=None),
            dict(cohort="supplement", baseline="a", speedup=9)]
    groups = refresh.summarize(rows, ("cohort", "baseline"))
    main = next(g for g in groups if g["cohort"] == "main")
    assert main["geometric_mean"] == pytest.approx(1)
    assert (main["requests"], main["measured"], main["missing"], main["nominal_faster"]) == (3, 2, 1, 1)
    assert main["minimum"] == 0.25
    assert main["maximum"] == 4


@pytest.mark.parametrize("failure", [None, "accepted", "finite", "graph", "timing"])
def test_filter_error_report_policy_preserves_source_and_real_failures(tmp_path, failure):
    graph = dict(distinct_input_outputs=True, previous_output_preserved=True,
                 restored_input_exact=True, same_precision_cpu_errors=[dict(finite=True)] * 2)
    row = dict(accepted=None, formal_performance_eligible=False,
               error=dict(finite=True, max_abs=12.5), timing=dict(median_us=2),
               graph_changing_input_validation=graph)
    own = json.loads(json.dumps(row))
    if failure == "accepted":
        own["accepted"] = False
    elif failure == "finite":
        own["error"]["finite"] = False
    elif failure == "graph":
        own["graph_changing_input_validation"]["previous_output_preserved"] = False
    elif failure == "timing":
        own["timing"] = {}
    name = "image1024_filter31_batch16"
    directory = tmp_path / "fft_filters" / name
    directory.mkdir(parents=True)
    data = dict(contract=dict(image_shape=[1024, 1024], batch=16),
                rows=[dict(own, method="cubutterfly"),
                      dict(row, method="cupy_fftconvolve_native_padding")])
    (directory / (name + ".current_graph_fast.json")).write_text(json.dumps(data))
    pair = refresh.Exporter(tmp_path).applications()["pairs"][0]
    assert pair["formal_performance_eligible"] is (failure is None)
    assert pair["speedup"] == (1 if failure is None else None)
    assert pair["independent_processes"] == 1
    source = pair["source_numerical_reports"]["cubutterfly"]
    assert source["formal_performance_eligible"] is False
    assert source["error"]["max_abs"] == 12.5


def test_actual_storage_controls_precision_and_missing_is_null():
    exporter = refresh.Exporter(".")
    request, pairs = exporter.operator_row(
        dict(id="missing", descriptor=dict(semantics=dict(operator=2, storage=3, shape=[8], batch=1)),
             requested_variants=[dict(regime="fp32")], baselines=[dict(name="dao", status="missing-baseline")]),
        "main66", "fixture")
    assert request["precision"] == "fp64"
    assert request["correctness"] == "error-report-only"
    assert request["own_us"] is None
    assert pairs[0]["baseline_us"] is None
    assert pairs[0]["speedup"] is None


def test_ckks_uses_only_measured_auto_rows_and_strongest_control(tmpdir):
    tmp_path = Path(str(tmpdir))
    directory = tmp_path / "ckks/auto_schedule_36"
    directory.mkdir(parents=True)
    (directory / "method1_n14.metadata.json").write_text(json.dumps(
        dict(status="completed", performance_eligible=True)))
    fields = ("scene", "N", "batch", "baseline", "cubutterfly_ms", "native_ms", "batch_submission",
              "exact_ciphertext", "mismatch_count", "keyswitch_method", "mode")
    base = dict(scene="rotate", N="16384", batch="1", cubutterfly_ms="2", exact_ciphertext="1",
                mismatch_count="0", keyswitch_method="I", mode="cuda_graph")
    rows = [dict(base, baseline="loop", native_ms="6", batch_submission="measured_auto_loop_or_stage_batch"),
            dict(base, baseline="stage", native_ms="3", batch_submission="measured_auto_loop_or_stage_batch"),
            dict(base, baseline="loop", native_ms="100", batch_submission="independent_public_API_loop")]
    for suffix in (".csv", "_population.csv", ".csv.selection.csv", ".csv.samples.csv"):
        with (directory / ("method1_n14" + suffix)).open("w") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    result = refresh.Exporter(tmp_path).applications()
    assert len(result["pairs"]) == 2
    assert len(result["strongest_native_pairs"]) == 1
    assert result["strongest_native_pairs"][0]["speedup"] == 1.5
    assert result["strongest_native_pairs"][0]["selected_baseline"] == "stage"


def test_ntt_trials_aggregate_paired_ratios(tmpdir):
    tmp_path = Path(str(tmpdir))
    directory = tmp_path / "ntt_search_repair_5"
    directory.mkdir()
    trials = [{"cubutterfly": {"median_ms": own}, "gpu-ntt": {"median_ms": baseline}}
              for own, baseline in [(1, 4), (10, 10), (100, 50)]]
    (directory / "results.json").write_text(json.dumps(dict(rows=[dict(
        case_id="ntt", semantics=dict(operator=1, storage=9, shape=[8], batch=1), trials=trials, status="pass")])))
    result = refresh.Exporter(tmp_path).operators()
    assert result["pairs"][0]["speedup"] == 1
    assert result["pairs"][0]["own_us"] == 10000


def test_mechanism_retains_paired_trial_range(tmpdir):
    tmp_path = Path(str(tmpdir))
    directory = tmp_path / "mechanism"
    directory.mkdir()
    comparison = dict(baseline="stagewise", treatment="resident8", status="measured",
                      baseline_median_kernel_ms=10, treatment_median_kernel_ms=10,
                      pairs=[dict(baseline_kernel_ms=baseline, treatment_kernel_ms=own)
                             for baseline, own in [(4, 1), (10, 10), (50, 100)]])
    (directory / "timing.json").write_text(json.dumps(dict(cases=[dict(
        id="example", workload=dict(operator="fft", precision="fp32", logN=12, batch=4),
        comparisons=[comparison])])))
    pair = refresh.Exporter(tmp_path).mechanisms()["pairs"][0]
    assert pair["trial_speedups"] == [4, 1, 0.5]
    assert pair["speedup"] == 1
    assert pair["speedup_min"] == 0.5
    assert pair["speedup_max"] == 4


def test_stale_error_timing_cannot_produce_plotted_speedup():
    exporter = refresh.Exporter(".")
    row = dict(id="failed", status="error", semantics=dict(operator=1, storage=9),
               timing=dict(median_ms=1), baselines=[dict(name="gpu-ntt", status="pass", timing=dict(median_ms=2))])
    request, pairs = exporter.operator_row(row, "main66", "fixture")
    assert request["own_us"] == 1000
    assert pairs[0]["baseline_us"] == 2000
    assert pairs[0]["speedup"] is None
    row["status"] = "pass"
    row["baselines"][0]["status"] = "error"
    assert exporter.operator_row(row, "main66", "fixture")[1][0]["speedup"] is None
    row["baselines"][0]["status"] = "pass"
    row["performance_eligible"] = False
    assert exporter.operator_row(row, "main66", "fixture")[1][0]["speedup"] is None


def test_ckks_without_metadata_is_pending(tmpdir):
    tmp_path = Path(str(tmpdir))
    directory = tmp_path / "ckks/auto_schedule_36"
    directory.mkdir(parents=True)
    row = dict(scene="rotate", N="16384", batch="1", baseline="loop", cubutterfly_ms="2", native_ms="3",
               batch_submission="measured_auto_loop_or_stage_batch", exact_ciphertext="1", mismatch_count="0",
               keyswitch_method="I", mode="cuda_graph")
    with (directory / "method1_n14.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    pair = refresh.Exporter(tmp_path).applications()["pairs"][0]
    assert pair["speedup"] is None
    assert pair["own_ms"] == 2
    assert pair["eligibility"] == "pending-unvalidated"


@pytest.fixture
def cyclic_campaign(tmp_path):
    source = "recovery_followups_20261029/cyclic_mapping/measurement/results.json"
    supplement = "recovery_gpuntt_baseline_20261102/results.json"
    protocol = "20 complete calls per CUDA graph; 7 replays; 10 warmup"
    timing = dict(median_ms=1, samples_ms=[1] * 7, protocol=protocol)
    descriptor = dict(semantics=dict(operator=1, storage=9, word_bits=64, shape=[65536], batch=1))
    original = [dict(id=name, status="pass", descriptor=descriptor, mapping={"backend": "fixture"},
                     timing=timing, baselines=[baseline]) for name, baseline in (
        ("original", dict(name="gpu-ntt", status="pass", support_status="supported",
                          timing=dict(timing, median_ms=0.5))),
        ("missing", dict(name="gpu-ntt", status="missing-baseline", support_status="missing-external-binding")),
        ("unsupported", dict(name="gpu-ntt", status="missing-baseline", support_status="unsupported-semantic",
                             reason="modulus outside arithmetic range")))]
    addition = dict(id="missing", status="pass", descriptor=descriptor, source_mapping=original[1]["mapping"],
                    source_cubutterfly_timing=timing, source_row_sha256=refresh.digest(original[1]),
                    baselines=[dict(name="gpu-ntt", status="pass", support_status="supported",
                                    timing=dict(timing, median_ms=3))])

    def write_report(relative, rows, identity):
        entries = []
        directory = tmp_path / Path(relative).parent
        (directory / "rows").mkdir(parents=True, exist_ok=True)
        for row in rows:
            sha = refresh.digest(row)
            row_path = "rows/" + row["id"] + ".json"
            (directory / row_path).write_text(json.dumps(dict(row=row, sha256=sha)))
            entries.append(dict(id=row["id"], status=row["status"], path=row_path, sha256=sha))
        data = dict(status="complete-with-gaps", rows=entries, rows_sha256=refresh.digest(entries),
                    identity=identity, remaining_case_ids=[])
        (tmp_path / relative).write_text(json.dumps(data))

    def save():
        manifest = dict(cases=[dict(id=r["id"], descriptor=r["descriptor"]) for r in original])
        manifest_path = tmp_path / Path(source).parent.parent / "manifest.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest))
        identity = dict(gpu_uuid="fixture-gpu", manifest_sha256=refresh.digest(manifest), library_sha256="library",
                        protocol={"warmup_calls": 10, "graph_calls": 20, "replays": 7})
        write_report(source, original, identity)
        supp_identity = dict(identity, source_report_sha256=hashlib.sha256((tmp_path / source).read_bytes()).hexdigest(),
                             manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                             backend=dict(root_helper_library=dict(sha256="library")))
        write_report(supplement, [addition], supp_identity)

    save()
    return tmp_path, original, addition, save, source, supplement


def test_cyclic_supplement_is_independent_and_preserves_unpaired(cyclic_campaign):
    directory, original, _, _, _, _ = cyclic_campaign
    data = refresh.Exporter(directory).export()
    cyclic = data["cyclic"]
    assert data["operators"]["requests"] == []
    assert len(cyclic["requests"]) == len(cyclic["pairs"]) == 3
    assert cyclic["summary"] == dict(count=2, geometric_mean=pytest.approx(1.5 ** 0.5),
                                      median=1.75, minimum=0.5, maximum=3, wins=1, losses=1, ties=0)
    assert cyclic["original_pairs"]["count"] == cyclic["supplemented_pairs"]["count"] == 1
    unsupported = next(p for p in cyclic["pairs"] if p["case_id"] == "unsupported")
    assert unsupported["speedup"] is None
    assert unsupported["support_status"] == "unsupported-semantic"
    added = next(r for r in cyclic["requests"] if r["id"] == "missing")
    assert added["original_baselines"] == original[1]["baselines"]
    assert added["mapping"] == original[1]["mapping"]
    assert added["supplement_source"]
    assert data["coverage"]["cyclic"]["measured_requests"] == 3
    assert cyclic["identities"]["original"]["gpu_uuid"] == "fixture-gpu"


@pytest.mark.parametrize("field", ["source_row_sha256", "descriptor", "source_mapping", "source_cubutterfly_timing"])
def test_cyclic_rejects_rehashed_supplement_provenance_tampering(cyclic_campaign, field):
    directory, _, addition, save, _, _ = cyclic_campaign
    addition[field] = "tampered"
    save()
    with pytest.raises(ValueError, match="source row mismatch"):
        refresh.Exporter(directory).cyclic()


@pytest.mark.parametrize("support,status", [("supported", "pass"), ("unsupported-semantic", "missing-baseline")])
def test_cyclic_supplement_cannot_overwrite_existing_baseline(cyclic_campaign, support, status):
    directory, original, addition, save, _, _ = cyclic_campaign
    original[1]["baselines"][0].update(status=status, support_status=support)
    addition["source_row_sha256"] = refresh.digest(original[1])
    save()
    with pytest.raises(ValueError, match="overwrite an existing baseline"):
        refresh.Exporter(directory).cyclic()


@pytest.mark.parametrize("target", ["source-report", "index", "envelope", "payload", "protocol"])
def test_cyclic_rejects_checksum_and_protocol_tampering(cyclic_campaign, target):
    directory, _, addition, save, source, supplement = cyclic_campaign
    if target == "protocol":
        addition["baselines"][0]["timing"]["protocol"] = "different timing"
        save()
    else:
        relative = source if target == "source-report" else supplement
        path = directory / relative
        report = json.loads(path.read_text())
        if target in ("envelope", "payload"):
            path = path.parent / report["rows"][0]["path"]
            report = json.loads(path.read_text())
            if target == "envelope":
                report["sha256"] = "tampered"
            else:
                report["row"]["status"] = "error"
        elif target == "index":
            report["rows"][0]["status"] = "error"
        else:
            report["status"] = "complete"
        path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="hash|checksum|protocol"):
        refresh.Exporter(directory).cyclic()


def test_cyclic_missing_sources_are_explicit(tmp_path):
    cyclic = refresh.Exporter(tmp_path).export()["cyclic"]
    assert cyclic["status"] == "missing"
    assert cyclic["requests"] == []
    assert cyclic["summary"]["count"] == 0
    assert cyclic["remaining_requests"] is None


def application_modules():
    import sys
    tools = MODULE.parent
    if str(tools) not in sys.path:
        sys.path.insert(0, str(tools))
    import export_application_closeout
    import render_current_refresh
    return export_application_closeout, render_current_refresh


def test_frozen_application_closeout_retains_every_cohort_and_negative_result():
    helper, _ = application_modules()
    data = helper.load(MODULE.parents[1] / 'generated/current_refresh/application_closeout.json')
    closure = [r for r in data['records'] if r['cohort'] == 'nov2-application-closure']
    assert len(closure) == 14
    assert len({r['job_id'] for r in closure}) == 9
    quarot = [r for r in closure if 'Llama' in r['scenario']]
    assert len(quarot) == 4 and all(.956 < r['speedup'] < .969 for r in quarot)
    assert {r['variant_order'] for r in quarot} == {'baseline-first', 'treatment-first'}
    assert len([r for r in closure if r['role'] == 'diagnostic']) == 4
    assert len([r for r in closure if 'CuPy' in r['baseline']]) == 4
    assert len([r for r in closure if 'bootstrap' in r['scenario']]) == 2
    fno = [r for r in data['records'] if r['cohort'] == 'historical-fno-hotpath']
    assert len(fno) == 4
    assert [r['speedup'] for r in fno if r['timing_mode'] == 'graph'] == pytest.approx(
        [1.0299315352768919, 1.0564188968444337])
    assert all(r['speedup'] < 1 for r in fno if r['timing_mode'] == 'eager')
    assert all(r['independent_processes'] == 2 for r in fno)
    assert len({r['library_sha256'] for r in data['records']}) == 2
    assert all(len(r['source_sha256']) == 64 for r in data['records'])
    table = helper.cohort_table(data)
    assert 'Scenario & Type & Baseline & Ratio' in table
    rows = [line for line in table.splitlines() if line.startswith(('QuaRot,', 'Filter ', 'Bootstrap,', 'FNO '))]
    assert len(rows) == 8 and all(line.count('&') == 3 for line in rows)
    assert 'QuaRot, B1 & FWHT & QuaRot/Dao & 0.968' in table
    assert 'QuaRot, B4 & FWHT & QuaRot/Dao & 0.957--0.959' in table
    assert 'Bootstrap, B8 & NTT & HEonGPU' in table
    assert 'FNO $64^2$, B4 & FFT & Torch & 1.030' in table
    assert 'fac109' not in table and 'ddb7d7' not in table


@pytest.mark.parametrize('tamper', ['ratio', 'source', 'missing-row'])
def test_application_closeout_rejects_corrupt_frozen_evidence(tmp_path, tamper):
    helper, _ = application_modules()
    data = json.loads((MODULE.parents[1] / 'generated/current_refresh/application_closeout.json').read_text())
    if tamper == 'ratio':
        data['records'][0]['speedup'] = 2
    elif tamper == 'source':
        data['records'][0]['source_sha256'] = '0' * 64
    else:
        data['records'].pop()
    path = tmp_path / 'frozen.json'
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        helper.load(path)


def test_frozen_renderer_never_loads_raw_results(tmp_path, monkeypatch):
    import shutil
    import sys
    _, renderer = application_modules()
    generated = tmp_path / 'paper/generated/current_refresh'
    generated.mkdir(parents=True)
    original = MODULE.parents[1] / 'generated/current_refresh'
    for name in ('snapshot.json', 'fp16_comparison.json', 'application_closeout.json'):
        shutil.copyfile(original / name, generated / name)
    monkeypatch.setattr(renderer, 'ROOT', tmp_path)
    monkeypatch.setattr(renderer, 'load_fp16_acceptance', lambda *a: pytest.fail('raw FP16 read'))
    monkeypatch.setattr(renderer, 'load_fft_l_comparison', lambda *a: pytest.fail('raw FFT-L read'))
    seen = []
    for name in ('operator_plot', 'application_plot', 'mechanism_plot'):
        monkeypatch.setattr(renderer, name, lambda data, *a: seen.append(data))
    monkeypatch.setattr(sys, 'argv', ['renderer', '--figures-only', '--frozen-inputs'])
    renderer.main()
    assert len(seen) == 3
    assert len(seen[0]['fft_l_comparison']['requests']) == 9
    assert len(seen[0]['application_closeout']['records']) == 18
    assert len([r for r in seen[0]['operators']['requests'] if r['cohort'] == 'main66']) == 66
    manifest = json.loads((generated / 'figure_manifest.json').read_text())
    assert manifest['frozen_inputs'] is True
    assert len(manifest['frozen_fp16_sha256']) == 64


def test_source_release_can_redraw_without_results_tree(tmp_path):
    import os
    import shutil
    import subprocess
    import sys
    paper = MODULE.parents[1]
    tools = tmp_path / 'paper/tools'
    tools.mkdir(parents=True)
    for name in ('render_current_refresh.py', 'load_fp16_acceptance.py',
                 'load_fft_l_comparison.py', 'export_application_closeout.py'):
        shutil.copyfile(paper / 'tools' / name, tools / name)
    generated = tmp_path / 'paper/generated/current_refresh'
    generated.mkdir(parents=True)
    for name in ('snapshot.json', 'fp16_comparison.json', 'application_closeout.json'):
        shutil.copyfile(paper / 'generated/current_refresh' / name, generated / name)
    for name in ('motivation_organization_a100.csv', 'application_historical_complete_call.json'):
        shutil.copyfile(paper / 'generated' / name, generated.parent / name)
    subprocess.run([sys.executable, str(tools / 'render_current_refresh.py'),
                    '--figures-only', '--frozen-inputs'], check=True, capture_output=True,
                   env=dict(os.environ, CUDA_VISIBLE_DEVICES=''), cwd=tmp_path)
    assert not (tmp_path / 'results').exists()
    manifest = json.loads((generated / 'figure_manifest.json').read_text())
    assert manifest['figures']['applications']['supplemental_graph_rows'] == 16
    assert manifest['figures']['applications']['historical_records'] == 4
    assert (tmp_path / 'paper/figures/current_refresh_applications.pdf').is_file()
