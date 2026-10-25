from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import benchmark_weak_regions as runner  # noqa: E402


def test_schema2_config_derives_ntt_and_external_commands():
    config_path = ROOT / "config" / "benchmark_weak_regions_20261022.json"
    config = runner._load_config(config_path)
    case = next(item for item in config["cases"] if item["id"] == "ntt-uint64-65536-b64")
    args = type("Args", (), {
        "benchmark": ROOT / "build-e13-shared" / "cubutterfly_plan_bench",
    })()
    ntt = runner._ntt_command(args, config, case)
    gpuntt = runner._gpuntt_command(args, config, case)
    assert ["--logN", "16"] == ntt[1:3]
    assert ["--batch", "64"] == ntt[3:5]
    assert "--output-order" in ntt and ntt[ntt.index("--output-order") + 1] == "natural"
    assert ["--log-n", "16"] == gpuntt[1:3]
    assert gpuntt[gpuntt.index("--word-bits") + 1] == "64"


def test_csv_parser_preserves_ntt_contract():
    output = (
        "device,backend,logN,N,batch,input_order,output_order,kernel_ms,correct,mapping_json\n"
        "A100,hybrid2d,16,65536,64,natural,natural,0.2,1,{\"backend\":\"hybrid2d\"}\n"
    )
    parsed = runner._parse_csv_timing(output)
    assert parsed["kernel_ms"] == 0.2
    assert parsed["correct"] is True
    assert parsed["input_order"] == "natural"
    assert parsed["mapping_json"]["backend"] == "hybrid2d"


def test_rank_one_fft_uses_auto_lowering_and_multidimensional_fft_uses_xd():
    config = {
        "timing": {"warmup": 0, "repeat": 1, "trials": 1},
    }
    args = type("Args", (), {"benchmark": ROOT / "build-e13-shared" / "cubutterfly_plan_bench"})()
    rank_one = {"id": "fft-1d", "operator": "fft", "shape": [1024], "batch": 4, "precision": "fp32"}
    rank_two = {"id": "fft-2d", "operator": "fft", "shape": [32, 32], "batch": 4, "precision": "fp32"}
    rank_one_command = runner._benchmark_command(args, config, rank_one)
    rank_two_command = runner._benchmark_command(args, config, rank_two)
    assert rank_one_command[rank_one_command.index("--execution-mode") + 1] == "auto"
    assert rank_two_command[rank_two_command.index("--execution-mode") + 1] == "xd"
    cached = runner._benchmark_command(
        args, config, rank_one, cache_path=ROOT / "results" / "cache.json")
    assert cached[cached.index("--cache") + 1].endswith("results/cache.json")


def test_rank_one_xd_is_rejected_as_a_runner_configuration_error():
    config = {
        "schema": 2,
        "cases": [{"id": "fft-1d", "operator": "fft", "shape": [1024],
                   "batch": 4, "precision": "fp32", "execution_mode": "xd"}],
    }
    path = ROOT / "results" / "_tmp_rank1_xd_config.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(__import__("json").dumps(config), encoding="utf-8")
    try:
        try:
            runner._load_config(path)
        except ValueError as exc:
            assert "rank-1" in str(exc)
        else:
            raise AssertionError("rank-1 forced xD should be rejected")
    finally:
        path.unlink()


def test_manifest_search_budget_is_forwarded_to_benchmark_environment():
    config = {"gpu": "2", "search_budget": 24, "xd_tune_max_bytes": 1024}
    env = runner._environment(config, ROOT / "build-e13-shared" / "libcuntt.so")
    assert env["CUBUTTERFLY_SEARCH_BUDGET"] == "24"
    assert env["CUBUTTERFLY_XD_TUNE_MAX_BYTES"] == "1024"


def test_only_filter_selects_declared_cases():
    config = runner._load_config(ROOT / "config" / "benchmark_weak_regions_20261022.json")
    selected = {"fft-fp32-1024x1024-b4", "fft-fp32-512x512-b16"}
    filtered = dict(config)
    filtered["cases"] = [case for case in config["cases"] if case["id"] in selected]
    assert {case["id"] for case in filtered["cases"]} == selected
