"""Bounded low-precision preparation preserves inputs and baseline gaps."""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmark import prepare_fft_l_quick as quick


def test_quick_config_declares_exact_low_precision_domains():
    config = quick.campaign.load_config(ROOT / "config/fft_l_quick_semantic.json")
    population = config["populations"][0]
    assert len(list(quick.campaign._requests(config))) == 32
    assert {tuple(r["transform"][k] for k in ("storage", "compute"))
            for r in population["regimes"]} == {(4, 0), (4, 1), (5, 0), (5, 1)}
    assert population["directions"] == ["forward", "inverse"]
    assert {len(shape) for shape in population["shapes"]} == {1, 2}


@pytest.mark.parametrize("storage", [4, 5])
def test_cpu_preparation_retains_quantized_sources_and_separate_scaled_control(monkeypatch, tmp_path, storage):
    s = dict(operator=0, storage=storage, compute=1, shape=[8], batch=1,
             inverse=0, normalize_inverse=False, placement=0, fft_type=0,
             fft_output_scale=1.0, length_mode=0, input_strides=[1], output_strides=[1],
             input_batch_stride=8, output_batch_stride=8, input_orders=[0], output_orders=[0],
             ntt_input_layout=0, ntt_output_layout=0, modulus=None, word_bits=None, matrices=[[]])
    case = {"id": quick.cpu.runner.digest(s), "descriptor": {"semantics": s},
            "capabilities": {"status": "eligible"}}
    manifest = {"cases": [case], "metrics": {"requested_variants": 1}, "manifest_sha256": "test"}
    monkeypatch.setattr(quick.campaign, "generate_manifest", lambda *args, **kwargs: manifest)
    output = tmp_path / "prepared"
    report = quick.prepare(ROOT / "config/fft_l_quick_semantic.json", "unused.so", output)
    assert len(report["rows"]) == (2 if storage == 4 else 1)
    assert report["protocol"]["float_verdict"] == "error-report"
    assert "not a same-precision oracle" in report["protocol"]["reference"]
    assert not report["protocol"]["timing"]
    assert len({row["id"] for row in report["rows"]}) == len(report["rows"])
    for row in report["rows"]:
        assert row["compute_type"] == "fft-l" and row["compute_precision"] == "fp32"
        assert all(not baseline["supported"] for baseline in row["baselines"].values())
        assert all(baseline["status"] == "unsupported-semantic" for baseline in row["baselines"].values())
        with np.load(output / row["artifact"]["path"]) as arrays:
            decoded = quick.cpu.runner.decode_storage(arrays["storage"], storage)
            np.testing.assert_array_equal(decoded, arrays["source"])
            np.testing.assert_array_equal(quick.cpu.runner.reference(s, arrays["source"]), arrays["reference"])
    assert json.loads((output / "preparation.json").read_text())["status"] == "cpu-prepared"
    with pytest.raises(FileExistsError):
        quick.prepare(ROOT / "config/fft_l_quick_semantic.json", "unused.so", output)


def test_import_does_not_import_torch_or_initialize_gpu():
    subprocess.run([sys.executable, "-c", "import sys; import benchmark.prepare_fft_l_quick; assert 'torch' not in sys.modules"],
                   cwd=ROOT, check=True)


def test_oversized_population_rejected_before_descriptor_work(monkeypatch, tmp_path):
    config = quick.campaign.load_config(ROOT / "config/fft_l_quick_semantic.json")
    config["populations"][0]["batches"].append(8)
    monkeypatch.setattr(quick.campaign, "load_config", lambda _: config)
    def forbidden(*args, **kwargs):
        raise AssertionError("descriptor generation must not start")
    monkeypatch.setattr(quick.campaign, "generate_manifest", forbidden)
    with pytest.raises(ValueError, match="32 requested"):
        quick.prepare("unused", "unused.so", tmp_path / "oversized")
