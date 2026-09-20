import json
import pathlib
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]/"scripts"))
import compile_module
from compile_module import compile_mapping, render_register_fft, render_shared, render_factor_fft
from benchmark_factor_streamed import make_parser, points, schedule_points


def test_standalone_module_is_one_mapping_with_versioned_entry_and_resources():
    point = dict(logN=21, local_stages=10, prefix_threads=256, prefix_ept=32, suffix_threads=256, suffix_ept=16)
    source = render_register_fft(point, 80, "test")
    assert "launch_prefix<10,8,Inverse>" in source
    assert "launch_grouped_suffix<11,16,2,Inverse>" in source
    assert "cubutterfly_module_v1" in source and "cudaFuncGetAttributes" in source
    assert "CMake" not in source
    with pytest.raises(ValueError):
        render_register_fft(dict(point, local_stages=11), 80, "test")


def test_register_prefix_strategy_defaults_and_rejections():
    point = dict(precision="fp32", logN=21, local_stages=10,
                 prefix_threads=256, prefix_ept=32,
                 suffix_threads=256, suffix_ept=16)
    native = render_register_fft(point, 80, "native")
    assert "launch_prefix<10,8,Inverse>" in native
    assert "prefix<10,8,Inverse>" in native

    xor = render_register_fft(dict(point, prefix_shared_layout="xor"), 80, "xor")
    assert "launch_prefix<10,8,Inverse,register_tile::NativeCodelet,Complex,true>" in xor
    assert "prefix<10,8,Inverse,register_tile::NativeCodelet,Complex,true>" in xor

    dx = render_register_fft(dict(point, prefix_codelet="cufftdx-thread"), 80, "dx")
    assert '#include "fft_thread_codelet.cuh"' in dx
    assert "register_tile::DxCodelet" in dx
    assert "Complex,false>" in dx

    fp64 = dict(point, precision="fp64")
    fp64_native = render_register_fft(fp64, 80, "fp64-native")
    assert "cudaFuncGetAttributes(&a,register_tile::prefix<10,8,Inverse,register_tile::NativeCodelet,Complex,false>)" in fp64_native
    assert "launch_prefix<10,8,Inverse>(" in fp64_native
    fp64_xor = render_register_fft(dict(fp64, prefix_shared_layout="xor"), 80, "fp64-xor")
    assert "Complex,true>" in fp64_xor
    with pytest.raises(ValueError, match="only supported for FP32"):
        render_register_fft(dict(fp64, prefix_codelet="cufftdx-thread"), 80, "bad")
    with pytest.raises(ValueError, match="prefix_codelet"):
        render_register_fft(dict(point, prefix_codelet="unknown"), 80, "bad")
    with pytest.raises(ValueError, match="prefix_shared_layout"):
        render_register_fft(dict(point, prefix_shared_layout="banked"), 80, "bad")


def test_all_operator_specializations_use_one_stage_layout_template():
    for op, precision in (("fft","fp64"),("fwht","bf16"),("structured-2x2","fp16"),
                          ("subset-zeta","uint32"),("superset-zeta","uint32"),("ntt","word32")):
        point=dict(operator=op,precision=precision,logN=12,stage_partition=[3,4,5],threads=128)
        source=render_shared(point,80,"test")
        assert "cubutterfly_module_v2" in source
        assert "cubutterfly_module_launch_group_v2" in source
        assert "shared_iterative_kernel<Operator,12,7,5,1,0>" in source
        assert "static_cast<Value*>(a.io.workspace)+1*extent" in source
        assert "a.io.element_stride" in source
    with pytest.raises(ValueError):
        render_shared(dict(point,stage_partition=[3,4,4]),80,"test")


def test_ntt_shared_specialization_carries_output_order_and_defaults_to_natural():
    point = dict(operator="ntt", precision="word64", logN=12,
                 stage_partition=[6, 6], threads=128)
    natural = render_shared(point, 80, "natural")
    bit_reversed = render_shared(dict(point, output_order="bit-reversed"), 80, "bit-reversed")

    assert "shared_iterative_kernel<Operator,12,0,6,1,0>" in natural
    assert "shared_iterative_kernel<Operator,12,6,6,1,0>" in natural
    assert "shared_iterative_kernel<Operator,12,0,6,1,1>" in bit_reversed
    assert "shared_iterative_kernel<Operator,12,6,6,1,1>" in bit_reversed
    assert natural != bit_reversed


def test_shared_jit_rejects_unknown_or_non_ntt_bit_reversed_output_order():
    ntt = dict(operator="ntt", precision="word64", logN=12,
               stage_partition=[6, 6], threads=128)
    with pytest.raises(ValueError, match="output_order"):
        render_shared(dict(ntt, output_order="appt-static"), 80, "test")
    with pytest.raises(ValueError, match="output_order"):
        render_shared(dict(ntt, output_order="invalid"), 80, "test")
    with pytest.raises(ValueError, match="only supported for NTT"):
        render_shared(dict(ntt, operator="fwht", precision="fp32",
                           output_order="bit-reversed"), 80, "test")


def test_shared_ntt_cache_identity_distinguishes_output_order(tmp_path, monkeypatch):
    calls = []

    def fake_run(command, **kwargs):
        calls.append(list(command))
        if len(command) == 2 and command[1] == "--version":
            return SimpleNamespace(returncode=0, stdout="fake nvcc\n", stderr="")
        pathlib.Path(command[-1]).write_bytes(b"fake module")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(compile_module.subprocess, "run", fake_run)
    point = dict(backend="shared-iterative", operator="ntt", precision="word64",
                 logN=12, stage_partition=[6, 6], threads=128)
    cache = tmp_path / "cache"
    natural = compile_mapping(dict(point, output_order="natural"),
                              root=pathlib.Path(__file__).resolve().parents[1],
                              mathdx=tmp_path / "mathdx", nvcc=pathlib.Path("fake-nvcc"),
                              sm=80, cache=cache)
    bit_reversed = compile_mapping(dict(point, output_order="bit-reversed"),
                                   root=pathlib.Path(__file__).resolve().parents[1],
                                   mathdx=tmp_path / "mathdx", nvcc=pathlib.Path("fake-nvcc"),
                                   sm=80, cache=cache)

    assert natural != bit_reversed
    assert natural.parent != bit_reversed.parent
    assert json.loads((natural.parent / "manifest.json").read_text())["mapping"]["output_order"] == "natural"
    assert json.loads((bit_reversed.parent / "manifest.json").read_text())["mapping"]["output_order"] == "bit-reversed"
    compile_calls = [command for command in calls if len(command) > 2]
    assert len(compile_calls) == 2

    cached = compile_mapping(dict(point, output_order="natural"),
                             root=pathlib.Path(__file__).resolve().parents[1],
                             mathdx=tmp_path / "mathdx", nvcc=pathlib.Path("fake-nvcc"),
                             sm=80, cache=cache)
    assert cached == natural
    assert len([command for command in calls if len(command) > 2]) == 2


def test_register_strategy_cache_identity_distinguishes_physical_choices(tmp_path, monkeypatch):
    calls = []

    def fake_run(command, **kwargs):
        calls.append(list(command))
        if len(command) == 2 and command[1] == "--version":
            return SimpleNamespace(returncode=0, stdout="fake nvcc\n", stderr="")
        pathlib.Path(command[-1]).write_bytes(b"fake module")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(compile_module.subprocess, "run", fake_run)
    point = dict(precision="fp32", logN=21, local_stages=10,
                 prefix_threads=256, prefix_ept=32,
                 suffix_threads=256, suffix_ept=16,
                 fft_core="register-tile")
    cache = tmp_path / "cache"
    root = pathlib.Path(__file__).resolve().parents[1]
    native = compile_mapping(point, root=root, mathdx=tmp_path / "mathdx",
                             nvcc=pathlib.Path("fake-nvcc"), sm=80, cache=cache)
    xor = compile_mapping(dict(point, prefix_shared_layout="xor"), root=root,
                          mathdx=tmp_path / "mathdx", nvcc=pathlib.Path("fake-nvcc"),
                          sm=80, cache=cache)
    dx = compile_mapping(dict(point, prefix_codelet="cufftdx-thread"), root=root,
                         mathdx=tmp_path / "mathdx", nvcc=pathlib.Path("fake-nvcc"),
                         sm=80, cache=cache)
    assert len({native.parent, xor.parent, dx.parent}) == 3
    assert json.loads((xor.parent / "manifest.json").read_text())["mapping"]["prefix_shared_layout"] == "xor"
    assert len([command for command in calls if len(command) > 2]) == 3


def test_factor_policy_validation_precedes_cache_hit(tmp_path, monkeypatch):
    calls = []

    def fake_run(command, **kwargs):
        calls.append(list(command))
        if len(command) == 2 and command[1] == "--version":
            return SimpleNamespace(returncode=0, stdout="fake nvcc\n", stderr="")
        pathlib.Path(command[-1]).write_bytes(b"fake module")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(compile_module.subprocess, "run", fake_run)
    point = dict(backend="factor-streamed", precision="fp32", logN=18,
                 stage_partition=[18], factor_partition=[6, 6, 6],
                 factor_columns=4, factor_ept=8,
                 data_tiles_per_cta=1, prefetch_depth=0)
    cache = tmp_path / "cache"
    root = pathlib.Path(__file__).resolve().parents[1]
    compile_mapping(point, root=root, mathdx=tmp_path / "mathdx",
                    nvcc=pathlib.Path("fake-nvcc"), sm=80, cache=cache)
    for invalid in (None, "dynamic", ["dynamic"], ["dynamic", "invalid", "dynamic"]):
        with pytest.raises(ValueError):
            compile_mapping(dict(point, factor_io_policies=invalid), root=root,
                            mathdx=tmp_path / "mathdx", nvcc=pathlib.Path("fake-nvcc"),
                            sm=80, cache=cache)
    assert len([command for command in calls if len(command) > 2]) == 1


def test_cooperative_prefix_keeps_io_columns_and_shared_capacity():
    point = dict(precision="fp32", logN=22, local_stages=12,
                 prefix_threads=256, prefix_ept=64, suffix_threads=128,
                 suffix_ept=32, prefix_codelet="cufftdx-thread", prefix_shared_layout="xor")
    for lanes in (1, 2, 4):
        source = render_register_fft(dict(point, prefix_codelet_lanes=lanes,
            prefix_threads=256*lanes, prefix_ept=64//lanes), 80, "test")
        tail = f",{lanes}" if lanes != 1 else ""
        specialization = f"12,4,Inverse,register_tile::DxCodelet,Complex,true{tail}"
        assert f"cudaFuncGetAttributes(&a,register_tile::prefix<{specialization}>)" in source
        assert f"launch_prefix<{specialization}>" in source
        assert f"out->threads={256*lanes}; out->dynamic_shared_bytes=(1U<<12)*4*8" in source
    fp64 = render_register_fft(dict(point, precision="fp64", prefix_codelet="native",
        prefix_threads=256, prefix_ept=16, prefix_codelet_lanes=4), 80, "test")
    assert "prefix<12,1,Inverse,register_tile::NativeCodelet,Complex,true,4>" in fp64


def test_cooperative_prefix_validation_and_default_cache_alias(tmp_path, monkeypatch):
    calls = []

    def fake_run(command, **kwargs):
        calls.append(list(command))
        if len(command) == 2:
            return SimpleNamespace(returncode=0, stdout="fake nvcc\n", stderr="")
        pathlib.Path(command[-1]).write_bytes(b"fake module")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(compile_module.subprocess, "run", fake_run)
    point = dict(precision="fp32", logN=22, local_stages=12, fft_core="register-tile",
                 prefix_threads=256, prefix_ept=64, suffix_threads=128, suffix_ept=32)
    kwargs = dict(root=pathlib.Path(__file__).resolve().parents[1], mathdx=tmp_path/"mathdx",
                  nvcc=pathlib.Path("fake-nvcc"), sm=80, cache=tmp_path/"cache")
    original = compile_mapping(point, **kwargs)
    assert compile_mapping(dict(point, prefix_codelet_lanes=1), **kwargs) == original
    assert compile_mapping(dict(point, prefix_codelet_lanes=2,
        prefix_threads=512, prefix_ept=32), **kwargs) != original
    count = len(calls)
    for invalid in (None, True, False, "2", 2.0, 0, 3, 128):
        with pytest.raises(ValueError, match="prefix_codelet_lanes"):
            compile_mapping(dict(point, prefix_codelet_lanes=invalid), **kwargs)
    for changes in (dict(prefix_codelet_lanes=2), dict(prefix_threads=257),
                    dict(prefix_codelet_lanes=8, prefix_threads=2048, prefix_ept=8),
                    dict(local_stages=6, prefix_codelet_lanes=8, prefix_ept=1, prefix_threads=512)):
        with pytest.raises(ValueError):
            compile_mapping(dict(point, **changes), **kwargs)
    with pytest.raises(ValueError, match="only applies"):
        compile_mapping(dict(point, backend="shared-iterative", prefix_codelet_lanes=2), **kwargs)
    assert len(calls) == count, "invalid geometry reached cache/compiler"


def test_factor_partition_is_independent_of_macro_group_and_prefetch():
    point=dict(precision="fp64",logN=24,stage_partition=[24],factor_partition=[8,8,8],
               factor_columns=8,factor_ept=16,data_tiles_per_cta=5,prefetch_depth=2)
    source=render_factor_fft(point,80,"test")
    assert "kernel<F1<Inverse>,24,8,8,8,5,2,Inverse,Complex>" in source
    assert "cubutterfly_module_local_bytes_v1" in source
    assert "group<3" in source
    render_factor_fft(dict(point,stage_partition=[8,16]),80,"test")
    for invalid in (dict(point,stage_partition=[10,14]),dict(point,prefetch_depth=5),dict(point,factor_columns=16)):
        with pytest.raises(ValueError): render_factor_fft(invalid,80,"test")
    with pytest.raises(ValueError): render_factor_fft(point,75,"test")


def test_native_factor_core_keeps_factor_and_data_schedule():
    point=dict(precision="fp64",logN=24,stage_partition=[24],factor_partition=[8,8,8],
               fft_core="register-tile",factor_columns=8,factor_ept=16,data_tiles_per_cta=4,prefetch_depth=1)
    source=render_factor_fft(point,80,"test")
    assert 'NativeFactor<8,16,8,Inverse,Complex>' in source
    assert 'kernel<F2<Inverse>,24,8,16,8,4,1,Inverse,Complex>' in source
    with pytest.raises(ValueError): render_factor_fft(dict(point,factor_partition=[7,9,8]),80,"test")


def test_factor_io_policy_is_per_factor_and_keeps_dynamic_default():
    point=dict(precision="fp32", logN=18, stage_partition=[18], factor_partition=[6,6,6],
               factor_columns=4, factor_ept=8, data_tiles_per_cta=1, prefetch_depth=1)
    default = render_factor_fft(point, 80, "default")
    explicit_dynamic = render_factor_fft(
        dict(point, factor_io_policies=["dynamic", "dynamic", "dynamic"]), 80, "default")
    assert default == explicit_dynamic

    mixed = render_factor_fft(
        dict(point, factor_io_policies=["static-unrolled", "dynamic", "static-unrolled"]),
        80, "mixed")
    static_group0 = "kernel<F0<Inverse>,18,6,0,4,1,1,Inverse,Complex,false,true>"
    dynamic_group1 = "kernel<F1<Inverse>,18,6,6,4,1,1,Inverse,Complex>"
    static_group2 = "kernel<F2<Inverse>,18,6,12,4,1,1,Inverse,Complex,false,true>"
    assert mixed.count(static_group0) >= 2
    assert mixed.count(dynamic_group1) >= 2
    assert mixed.count(static_group2) >= 2

    partial = render_factor_fft(
        dict(point, factor_slices=2,
             factor_io_policies=["static-unrolled", "dynamic", "dynamic"]),
        80, "partial")
    assert "kernel<F0<Inverse>,18,6,0,4,1,1,Inverse,Complex,true,true>" in partial
    with pytest.raises(ValueError, match="empty or match"):
        render_factor_fft(dict(point, factor_io_policies=["dynamic"]), 80, "bad")
    with pytest.raises(ValueError, match="dynamic or static-unrolled"):
        render_factor_fft(dict(point, factor_io_policies=["dynamic", "static", "dynamic"]), 80, "bad")
    with pytest.raises(ValueError, match="must be a list"):
        render_factor_fft(dict(point, factor_io_policies="dynamic"), 80, "bad")


def test_factor_partial_range_entrypoint_is_emitted_and_validated():
    point=dict(precision="fp32",logN=18,stage_partition=[18],factor_partition=[6,6,6],
               factor_columns=4,factor_ept=8,data_tiles_per_cta=1,prefetch_depth=0,
               factor_slices=2,factor_overlap=True)
    source=render_factor_fft(point,80,"test")
    assert "cubutterfly_module_launch_range_v1" in source
    assert "factor_streamed::kernel<F1<Inverse>,18,6,6,4,1,0,Inverse,Complex,true>" in source
    with pytest.raises(ValueError): render_factor_fft(dict(point,factor_slices=3),80,"test")


def test_factor_benchmark_keeps_default_seven_point_grid():
    variants = list(points(18, core="cufftdx-block", factors=[6, 6, 6], columns=8, ept=16))
    assert [name for name, _ in variants] == [
        "tiles1-depth0", "tiles4-depth0", "tiles4-depth1", "tiles4-depth2",
        "tiles16-depth0", "tiles16-depth1", "tiles16-depth2",
    ]
    assert all("factor_slices" not in mapping and "factor_overlap" not in mapping
               for _, mapping in variants)


def test_factor_benchmark_schedule_ablation_emits_serial_and_overlap_controls():
    variants = list(schedule_points(18, core="cufftdx-block", factors=[6, 6, 6],
                                    columns=8, ept=16, data_tiles_per_cta=4,
                                    prefetch_depth=1, factor_slices=(2, 4, 8)))
    assert [name for name, _ in variants] == [
        "whole", "slice2-serial", "slice2-overlap", "slice4-serial", "slice4-overlap",
        "slice8-serial", "slice8-overlap",
    ]
    assert variants[0][1]["factor_slices"] == 1 and not variants[0][1]["factor_overlap"]
    for name, mapping in variants[1:]:
        expected_slices = int(name.split("-")[0][5:])
        assert mapping["factor_slices"] == expected_slices
        assert mapping["factor_overlap"] == name.endswith("-overlap")
        assert mapping["data_tiles_per_cta"] == 4 and mapping["prefetch_depth"] == 1
        render_factor_fft(dict(mapping, logN=18, precision="fp32"), 80, "test")


def test_factor_benchmark_parser_defaults_and_bounded_verification_option():
    args = make_parser().parse_args(["--bench", "bench", "--output", "results"])
    assert not args.schedule_ablation
    assert args.data_tiles_per_cta == 4 and args.prefetch_depth == 1
    assert args.factor_slices == [2, 4, 8] and args.verify_batches == 0
    args = make_parser().parse_args([
        "--bench", "bench", "--output", "results", "--schedule-ablation",
        "--factor-slices", "2", "8", "--verify-batches", "2",
    ])
    assert args.schedule_ablation and args.factor_slices == [2, 8]
    assert args.verify_batches == 2
