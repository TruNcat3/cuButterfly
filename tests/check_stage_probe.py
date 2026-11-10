#!/usr/bin/env python3
"""Check that isolated physical launches recompose all native operators.

The executable checks each batch against the CPU reference before emitting
any timing. Run this GPU check on an exclusive device with the same compile
policy as the intended calibration.
"""
import argparse
import json
import subprocess


def cases():
    operators = (("fft", "fp32"), ("fft", "fp64"), ("fwht", "fp32"),
                 ("fwht", "fp64"), ("subset-zeta", "uint32"),
                 ("superset-zeta", "uint32"), ("xor-zeta", "uint32"),
                 ("structured-2x2", "fp32"), ("ntt", "word32"), ("ntt", "word64"))
    for operator, precision in operators:
        mapping = {"schema_version": 1, "backend": "shared-iterative", "compute_unit": "radix2",
                   "stage_partition": [3, 2, 3]}
        if operator == "ntt":
            mapping.update(kind="ntt", threads_per_block=32, dataflow_layout="hermes-xor")
        else:
            mapping.update(kind="butterfly", tile_threads=32, shared_layout="writer-aligned")
        for inverse in (False, True):
            point = dict(operator=operator, precision=precision, logN=8, batch=3,
                         direction="inverse" if inverse else "forward",
                         normalization="inverse", mapping_json=json.dumps(mapping))
            if operator == "ntt":
                point["modulus"] = 998244353 if precision == "word32" else 576460756061519873
            else:
                point.update(placement="out-of-place", element_stride=2, batch_stride=520)
            if operator == "structured-2x2":
                point["stage_matrix"] = "1,0.25,-0.5,1"
            yield point
    for precision in ("fp16", "bf16"):
        for accumulation in ("native", "fp32"):
            for inverse in (False, True):
                for normalization in ("none", "inverse"):
                    mapping = dict(schema_version=1, kind="butterfly", backend="shared-iterative",
                                   compute_unit="radix2", stage_partition=[3, 2, 3],
                                   tile_threads=32, shared_layout="writer-aligned")
                    yield dict(operator="fwht", precision=precision, accumulation=accumulation,
                               logN=8, batch=3, direction="inverse" if inverse else "forward",
                               normalization=normalization, placement="out-of-place",
                               element_stride=2, batch_stride=520, mapping_json=json.dumps(mapping))
    # The installation's two previously unsupported FP16 physical mappings.
    for log_n, partition, local_partition, threads in (
            (5, [3, 1, 1], [[2, 1], [1], [1]], 64),
            (6, [1, 3, 1, 1], [[1], [3], [1], [1]], 128)):
        mapping = dict(schema_version=1, kind="butterfly", backend="shared-iterative",
                       compute_unit="radix2", stage_partition=partition,
                       local_stage_partitions=local_partition, tile_threads=threads,
                       shared_layout="linear")
        yield dict(operator="fwht", precision="fp16", accumulation="fp32", logN=log_n,
                   batch=3, direction="forward", normalization="none",
                   placement="out-of-place", mapping_json=json.dumps(mapping))
    # Compact-stage has three physical launches even though its public mapping
    # uses an implicit partition. Exercise the sampling path as well as the
    # direct C++ adapter recomposition test.
    for order in ("natural", "bit-reversed"):
        mapping = dict(schema_version=1, kind="ntt", backend="compact-stage",
                       compute_unit="radix2", threads_per_block=256)
        yield dict(operator="ntt", precision="word64", logN=20, batch=4,
                   direction="forward", modulus=576460756061519873,
                   input_order="natural", output_order=order, placement=order,
                   mapping_json=json.dumps(mapping))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True)
    args = parser.parse_args()
    count = 0
    for point in cases():
        completed = subprocess.run([args.binary, "--point-json", json.dumps(point),
                                    "--warmup", "1", "--repeat", "2", "--trials", "2"],
                                   capture_output=True, text=True)
        result = json.loads(completed.stdout)
        if completed.returncode or result.get("status") != "measured" or not result.get("correct"):
            raise RuntimeError(f"stage recomposition failed for {point}: {result}; {completed.stderr}")
        groups = result["groups"]
        mapping = json.loads(point["mapping_json"])
        partition = [5, 6, 9] if mapping["backend"] == "compact-stage" else mapping["stage_partition"]
        assert [group["stage_count"] for group in groups] == partition, (point, groups)
        assert all(group["independent"] and group.get("kernel_launch_count") == 1 for group in groups), (point, groups)
        assert all(group.get("compiler_resources_known") and group.get("compiler_local_resources_known")
                   and not group.get("resource_query_error") for group in groups), (point, groups)
        assert all(len(group["trial_kernel_ms"]) == 2 for group in groups)
        assert len(result["pairs"]) == len(groups) - 1 and len(result["plan_trial_kernel_ms"]) == 2
        if point["precision"] in {"fp16", "bf16"}:
            assert result["verification"]["comparison"] == "exact-finite-values"
            assert result["sample"]["precision"] == point["precision"]
            assert result["sample"]["accumulation"] == point["accumulation"]
        count += 1
    # A separate boundary transpose is not part of either isolated stage.
    for layout in ("direct-strided", "tiled-transpose"):
        mapping = dict(schema_version=1, kind="butterfly", backend="shared-iterative",
                       compute_unit="radix2", stage_partition=[4, 4], tile_threads=32,
                       shared_layout="writer-aligned", boundaries=[
                           dict(twiddle="table", layout=layout, residency="global-scratch")])
        point = dict(operator="fft", precision="fp32", logN=8, batch=3,
                     direction="forward", normalization="none", placement="out-of-place",
                     mapping_json=json.dumps(mapping))
        completed = subprocess.run([args.binary, "--point-json", json.dumps(point),
                                    "--warmup", "1", "--repeat", "2", "--trials", "2"],
                                   capture_output=True, text=True)
        result = json.loads(completed.stdout)
        groups = result.get("groups", [])
        assert [group["stage_count"] for group in groups] == [4, 4], (point, result, completed.stderr)
        if layout == "direct-strided":
            assert completed.returncode == 0 and result.get("status") == "measured" and result.get("correct"), (
                point, result, completed.stderr)
            assert all(group["independent"] and len(group["trial_kernel_ms"]) == 2 for group in groups), result
            assert len(result["pairs"]) == 1 and len(result["plan_trial_kernel_ms"]) == 2, result
            count += 1
        else:
            reason = "separate boundary materialization requires a boundary probe before stage recomposition"
            assert completed.returncode == 2 and result.get("status") == "unavailable" and result.get("correct") is False, (
                point, result, completed.stderr)
            assert all(group["independent"] is False and group.get("reason") == reason for group in groups), result
            assert result["pairs"] == [] and "plan_trial_kernel_ms" not in result, result
            assert all("trial_kernel_ms" not in group for group in groups), result
    print(f"PASS {count} native stage recomposition cases")


if __name__ == "__main__":
    main()
