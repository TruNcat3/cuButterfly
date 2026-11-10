"""Finite screen contracts: no hardware access or timing substitutes."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "results/a100_closure_20261102/mechanism/bounded_search.py"
SPEC = importlib.util.spec_from_file_location("bounded_mechanism", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def mapping(resident=False):
    return {"backend": "shared-iterative", "compute_unit": "radix2", "fft_core": "scalar",
            "local_exchange": "shared", "stage_partition": [8, 8], "tile_threads": 128,
            "local_stage_partitions": [[4, 4], [4, 4]] if resident else [],
            "complex_multiply": "four-mul", "cross_twiddle": "table"}


def test_fixed_leaf_filter_is_conservative():
    assert MODULE.family(mapping(), "fft") == "conventional"
    assert MODULE.family(mapping(True), "fft") == "hybrid"
    for field, value in [("compute_unit", "radix4"), ("fft_core", "cufftdx-block"),
                         ("complex_multiply", "three-mul"), ("backend", "cufft")]:
        changed = dict(mapping(), **{field: value})
        assert MODULE.family(changed, "fft") is None


def test_prepare_freezes_budget_without_gpu(tmp_path, monkeypatch):
    case = {"id": "fft-fixture", "workload": {"operator": "fft", "precision": "fp32", "logN": 16, "batch": 16}}
    (tmp_path / "fft-fixture").mkdir()
    mappings = [dict(mapping(resident), tile_threads=32 * (i + 1))
                for resident in (False, True) for i in range(20)]
    (tmp_path / "fft-fixture/inventory.jsonl").write_text("\n".join(json.dumps(row) for row in mappings))
    monkeypatch.setattr(MODULE, "read", lambda _: {"requests": [case]})
    result = MODULE.prepare(tmp_path)
    assert result[0]["status"] == "prepared_for_screen"
    assert {k: v["scheduled"] for k, v in result[0]["pools"].items()} == {"conventional": 16, "hybrid": 16}
    assert all(v["budget_omitted"] == 4 for v in result[0]["pools"].values())


def test_finalists_reject_resolved_core_drift():
    variants = []
    for side in ("conventional", "hybrid"):
        for i in range(5):
            actual = mapping(side == "hybrid")
            if i == 0:
                actual["compute_unit"] = "radix4"
            variants.append({"id": f"{side}-{i}", "preflight": {"status": "accepted", "resolved_mapping": actual},
                             "trials": [{"performance_accepted": True, "sample": {"kernel_ms": i + 1}}]})
    selected, rejected = MODULE.finalists({"workload": {"operator": "fft"}}, {"cases": [{"variants": variants}]})
    assert len(selected) == 6
    assert len(rejected) == 2
    assert not any(row["id"].endswith("-0") for row in selected)


def test_profile_own_only_prepares_five_operator_only_jobs(tmp_path):
    script = PATH.with_name("profile_fwht.py")
    result = subprocess.run([sys.executable, str(script), "--mode", "prepare", "--operator", "fft",
                             "--own-only", "--output-dir", str(tmp_path),
                             "--mapping-file", str(tmp_path / "selected.json")], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["gpu_queried"] is False
    assert len(manifest["jobs"]) == 5
    assert {row["backend"] for row in manifest["jobs"]} == {"cubutterfly"}
    ncu = next(row["command"] for row in manifest["jobs"] if row["kind"] == "ncu")
    assert ncu[ncu.index("--nvtx-include") + 1] == "cb_operator/"
    assert "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum" in ncu[ncu.index("--metrics") + 1]
    nsys = next(row["command"] for row in manifest["jobs"] if row["kind"] == "nsys")
    assert "--cuda-graph-trace=node" in nsys


def test_profile_rejects_unbounded_selection_before_cuda(tmp_path):
    command = [sys.executable, str(PATH.with_name("profile_fwht.py")),
               "--mode", "worker", "--backend", "cubutterfly", "--worker-mode", "measure",
               "--mapping-file", str(tmp_path / "missing.json"), "--gpu-uuid", "not-a-device",
               "--output", str(tmp_path / "unused.json")]
    result = subprocess.run(command, env=dict(os.environ, CUBUTTERFLY_SEARCH_BUDGET="0"),
                            capture_output=True, text=True)
    assert result.returncode != 0
    assert "requires an explicit positive CUBUTTERFLY_SEARCH_BUDGET" in result.stderr
    assert not (tmp_path / "unused.json").exists()


def test_transform_does_not_apply_ntt_modulus_to_float_operators():
    profile_spec = importlib.util.spec_from_file_location("profile_mechanism", PATH.with_name("profile_fwht.py"))
    profile = importlib.util.module_from_spec(profile_spec)
    profile_spec.loader.exec_module(profile)
    import cubutterfly as cb
    for op in ("fft", "fwht"):
        transform = profile.make_transform(cb, op, 1024, 2)
        assert transform.modulus == cb.Transform().modulus
        assert transform.extents == [1024]
        assert transform.batch == 2
    ntt = profile.make_transform(cb, "ntt", 1024, 2)
    assert ntt.modulus == 576460756061519873
    assert ntt.storage == cb.DataType.UINT64


def test_profile_keeps_software_launch_metrics_and_accepts_wide_kernel_csv(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("profile_metric_regression", PATH.with_name("profile_fwht.py"))
    profile = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(profile)
    assert profile.unavailable_metrics(["launch__registers_per_thread", "dram__bytes.sum", "missing__counter.sum"],
                                       "dram__bytes") == ["missing__counter.sum"]
    rows = profile.ncu_kernel_rows('==PROF== Connected\n"ID","Kernel Name","launch__registers_per_thread"\n"","","register/thread"\n"0","fwht_kernel","32"\n')
    assert len(rows) == 1 and rows[0]["Kernel Name"] == "fwht_kernel"
    assert not profile.ncu_kernel_rows("==WARNING== No kernels were profiled.\n")
    monkeypatch.setattr(profile, "require_no_other_ncu", lambda: None)
    monkeypatch.setenv("TMPDIR", "/tmp")
    private = profile.configure_profiler_tmpdir(tmp_path)
    assert Path(private).parent == tmp_path
    assert os.environ["TMPDIR"] == private
    assert not list(Path(private).glob("writable-*"))
