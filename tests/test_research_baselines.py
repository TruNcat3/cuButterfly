from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import research_baselines as baselines


@pytest.fixture
def paths(tmp_path):
    executable = tmp_path / "cubutterfly_bench"
    executable.write_text("test executable")
    return dict(build_dir=tmp_path, vkfft_binary=executable, gpuntt_binary=executable, fht_python=executable)


def names(workload, paths):
    return {s["name"] for s in baselines.available_baselines(workload, paths)["available"]}


def test_external_contracts_do_not_pair_unsupported_semantics(paths):
    fft = dict(operator="fft", precision="fp32", logN=12, batch=4, placement="out-of-place")
    assert names(fft, paths) == {"cuFFT", "VkFFT", "in-tree-radix2"}
    for changes in (dict(precision="fp64"), dict(direction="inverse"), dict(element_stride=2), dict(batch_stride=4100)):
        assert names(dict(fft, **changes), paths) == {"cuFFT", "in-tree-radix2"}
    fwht = dict(fft, operator="fwht")
    assert "Dao-FWHT" in names(fwht, paths)
    assert "Dao-FWHT" in names(dict(fwht, direction="inverse", normalization="inverse"), paths)
    for changes in (dict(logN=20), dict(precision="fp64"), dict(placement="in-place")):
        assert names(dict(fwht, **changes), paths) == {"in-tree-radix2"}
    zeta = dict(fft, operator="subset-zeta", precision="uint32")
    plan = baselines.available_baselines(zeta, paths)
    assert plan["available"][0]["external"] is False
    assert plan["unavailable"]


def test_ntt_prime_order_and_direction_are_exact(paths):
    ntt = dict(operator="ntt", precision="word64", logN=16, batch=4,
               modulus=baselines.GPU_NTT_MODULUS, input_order="natural", output_order="natural")
    assert names(ntt, paths) == {"GPU-NTT-natural"}
    assert names(dict(ntt, output_order="bit-reversed"), paths) == {"GPU-NTT-bit-reversed"}
    for changes in (dict(precision="word32"), dict(modulus="2013265921"),
                    dict(direction="inverse"), dict(input_order="bit-reversed")):
        assert not names(dict(ntt, **changes), paths)


def test_gpuntt_checks_exact_layout_marker_and_modulus(monkeypatch, paths):
    workload = dict(operator="ntt", precision="word64", logN=16, batch=4,
                    modulus=baselines.GPU_NTT_MODULUS, input_order="natural", output_order="natural")
    spec = baselines.available_baselines(workload, paths)["available"][0]
    protocol = dict(warmup=100, repeat=100, verify_batches=0)
    header = "logN,batch,modulus,warmup,repeat,kernel_ms\n"
    output = header + "16,4,576460756061519873,100,100,0.1\n"
    response = SimpleNamespace(stdout=output, stderr="layout_check,natural_mismatches=17,bit_reversed_mismatches=0\n")
    monkeypatch.setattr(baselines.subprocess, "run", lambda *a, **k: response)
    with pytest.raises(ValueError, match="correctness"):
        baselines.run_baseline(workload, spec, protocol, paths)
    response.stderr = "layout_check,natural_mismatches=0,bit_reversed_mismatches=17\n"
    result = baselines.run_baseline(workload, spec, protocol, paths)
    assert result["verification"]["batches"] == 1
    assert result["sample"]["output_order"] == "natural"
    response.stdout = output.replace("576460756061519873", "576460756061519874")
    with pytest.raises(ValueError, match="modulus mismatch"):
        baselines.run_baseline(workload, spec, protocol, paths)


def test_cufft_retains_strides_and_excludes_auto_selection(monkeypatch, paths):
    workload = dict(operator="fft", precision="fp64", logN=8, batch=4, element_stride=2,
                    batch_stride=519, placement="out-of-place", direction="inverse", normalization="inverse")
    spec = baselines.available_baselines(workload, paths)["available"][0]
    seen = []
    def run(command, **kwargs):
        seen.append(command)
        return SimpleNamespace(stdout="logN,batch,precision,direction,placement,kernel_ms,correct\n8,4,fp64,inverse,out-of-place,0.1,1\n", stderr="")
    monkeypatch.setattr(baselines.subprocess, "run", run)
    result = baselines.run_baseline(workload, spec, dict(warmup=100, repeat=100, verify_batches=0), paths)
    assert "--auto-select" not in seen[0]
    assert seen[0][seen[0].index("--backend") + 1] == "cufft"
    assert seen[0][seen[0].index("--batch-stride") + 1] == "519"
    assert "--inverse" in seen[0]
    assert result["verification"]["batches"] == 4


def test_radix2_log22_uses_shared_iterative_groups_without_legacy_local_stages(monkeypatch, paths):
    workload = dict(operator="fft", precision="fp32", logN=22, batch=1, placement="out-of-place")
    spec = next(s for s in baselines.available_baselines(workload, paths)["available"]
                if s["name"] == "in-tree-radix2")
    protocol = dict(warmup=100, repeat=100, verify_batches=0)
    seen = []

    def run(command, **kwargs):
        seen.append(command)
        return SimpleNamespace(
            stdout="logN,batch,precision,placement,kernel_ms,correct\n22,1,fp32,out-of-place,0.1,1\n",
            stderr="",
        )

    monkeypatch.setattr(baselines.subprocess, "run", run)
    result = baselines.run_baseline(workload, spec, protocol, paths)
    command = seen[0]
    assert result["correct"] is True
    assert command[command.index("--backend") + 1] == "shared-iterative"
    assert command[command.index("--stage-partition") + 1] == "8,8,6"
    assert command[command.index("--tile-threads") + 1] == "128"
    assert command[command.index("--fft-core") + 1] == "scalar"
    assert command[command.index("--compute-unit") + 1] == "radix2"
    assert command[command.index("--local-exchange") + 1] == "shared"
    assert command[command.index("--shared-layout") + 1] == "writer-aligned"
    assert "--local-stages" not in command


@pytest.mark.parametrize("log_n,backend,local_stages", [(1, "temporal-tile", "0"),
                                                         (8, "temporal-tile", "0"),
                                                         (9, "hierarchical", "8"),
                                                         (20, "hierarchical", "8")])
def test_radix2_legacy_log20_and_below_commands_are_unchanged(
        monkeypatch, paths, log_n, backend, local_stages):
    workload = dict(operator="fft", precision="fp32", logN=log_n, batch=1, placement="out-of-place",
                    normalization="none")
    spec = next(s for s in baselines.available_baselines(workload, paths)["available"]
                if s["name"] == "in-tree-radix2")
    protocol = dict(warmup=100, repeat=100, verify_batches=0)
    seen = []

    def run(command, **kwargs):
        seen.append(command)
        return SimpleNamespace(
            stdout=("logN,batch,precision,placement,kernel_ms,correct\n"
                    f"{log_n},1,fp32,out-of-place,0.1,1\n"),
            stderr="",
        )

    monkeypatch.setattr(baselines.subprocess, "run", run)
    baselines.run_baseline(workload, spec, protocol, paths)
    expected = [str(paths["build_dir"] / "cubutterfly_bench"),
                "--operator", "fft", "--precision", "fp32", "--logN", str(log_n),
                "--batch", "1", "--placement", "out-of-place", "--normalization", "none",
                "--backend", backend, "--compute-unit", "radix2", "--local-stages", local_stages,
                "--tile-threads", "128", "--fft-core", "scalar", "--local-exchange", "shared",
                "--warmup", "100", "--repeat", "100", "--verify", "--verify-batches", "0", "--csv"]
    assert seen == [expected]
