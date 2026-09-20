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
        assert len(groups) == 3, (point, groups)
        assert all(group["independent"] and group.get("kernel_launch_count") == 1 for group in groups), (point, groups)
        assert all(group.get("compiler_resources_known") and group.get("compiler_local_resources_known")
                   and not group.get("resource_query_error") for group in groups), (point, groups)
        assert all(len(group["trial_kernel_ms"]) == 2 for group in groups)
        assert len(result["pairs"]) == 2 and len(result["plan_trial_kernel_ms"]) == 2
        count += 1
    print(f"PASS {count} native stage recomposition cases")


if __name__ == "__main__":
    main()
