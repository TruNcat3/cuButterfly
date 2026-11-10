import pathlib
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import sweep_cubutterfly_designs as sweep


def test_visible_gpu_uses_first_cuda_visible_device(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,1")
    assert sweep.visible_gpu() == "2"


def test_exclusive_gpu_rejects_active_compute_process(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")

    class Result:
        def __init__(self, stdout):
            self.stdout = stdout

    def run(command, **kwargs):
        if any("utilization.gpu" in str(item) for item in command):
            return Result("80\n")
        return Result("1234, other-workload, 4096\n")

    monkeypatch.setattr(sweep.subprocess, "run", run)
    try:
        sweep.require_exclusive_gpu()
    except RuntimeError as error:
        assert "other-workload" in str(error)
    else:
        raise AssertionError("active compute process was not rejected")


def test_exclusive_gpu_accepts_idle_device(monkeypatch):
    class Result:
        def __init__(self, stdout):
            self.stdout = stdout

    monkeypatch.setattr(sweep.subprocess, "run", lambda *args, **kwargs: Result(""))
    sweep.require_exclusive_gpu()


def test_exclusive_gpu_ignores_small_idle_context(monkeypatch):
    class Result:
        def __init__(self, stdout):
            self.stdout = stdout

    def run(command, **kwargs):
        if any("utilization.gpu" in str(item) for item in command):
            return Result("0\n")
        return Result("1234, idle-context, 12\n")

    monkeypatch.setattr(sweep.subprocess, "run", run)
    sweep.require_exclusive_gpu()
