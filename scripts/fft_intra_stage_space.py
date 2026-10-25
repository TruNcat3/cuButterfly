"""Finite CPU diagnostic subset: joint factor geometry and buffering choices.

This is not the definition of the public design space. Accepted points satisfy
known necessary conditions; exact compiled resources and execution are pending.
"""
from __future__ import annotations

import compile_module
import calibration_space
from cufftdx_capabilities import load_database
from research_compile_requests import compile_request


# Keep the explicit one-tile prefetch points in the public diagnostic space.
# Their lowering may collapse to the synchronous path when no successor tile
# exists, but removing the axis here would hide a valid user request from the
# common planner and calibration reports.
SCHEDULES = ((1, 0), (1, 1), (1, 2), (4, 0), (4, 1), (4, 2))


def build_space(case, hardware, mathdx):
    """Return accepted/rejected full combinations without compilation or CUDA.

    ``hardware`` supplies sm (80 units), max_threads_per_block, and
    max_shared_per_block_optin_bytes. Shared lower bounds include the depth+1
    input ring and one reusable work tile when a CTA owns multiple tiles. A
    one-tile CTA has no successor to overlap and is projected as the original
    synchronous single-slot lowering; excess provider scratch remains pending
    and exact compiled resources establish launch feasibility.
    """
    for key in ("sm", "max_threads_per_block", "max_shared_per_block_optin_bytes"):
        if type(hardware.get(key)) is not int or hardware[key] <= 0:
            raise ValueError(f"hardware.{key} must be a positive integer")
    log_n = case["logN"]
    if type(log_n) is not int or log_n not in (16, 20):
        raise ValueError("this bounded diagnostic subset requires logN 16 or 20")
    if case.get("precision") not in ("fp32", "fp64"):
        raise ValueError("this bounded diagnostic subset requires FP32 or FP64")
    if type(case.get("batch")) is not int or case["batch"] <= 0:
        raise ValueError("case.batch must be a positive integer")
    if "shape" in case and case["shape"] != [1 << log_n]:
        raise ValueError("case.shape must retain the requested rank-one FFT extent")
    database = load_database(mathdx)
    partitions = list(dict.fromkeys(((log_n - 10, 10), (log_n // 2, log_n - log_n // 2))))
    result = {"accepted": [], "rejected": []}
    workload = {**case, "operator": "fft"}
    for partition in partitions:
        for columns in (2, 4):
            for ept in (8, 16):
                family = f"factor_f{'x'.join(map(str, partition))}_c{columns}_e{ept}"
                for tiles, depth in SCHEDULES:
                    mapping = dict(kind="butterfly", backend="factor-streamed", fft_core="cufftdx-block",
                                   compute_unit="auto", stage_partition=[log_n], factor_partition=list(partition),
                                   factor_columns=columns, factor_ept=ept, shared_layout="writer-aligned",
                                   cross_twiddle="recurrence", data_tiles_per_cta=tiles, prefetch_depth=depth,
                                   stage_overlap=False, factor_overlap=False, factor_slices=1)
                    entry = dict(provider=f"{family}_t{tiles}_p{depth}", mapping=mapping,
                                 family=family, resource_lower_bounds=None)
                    reasons = []
                    point = calibration_space.mapping_point(workload, mapping)
                    try:
                        request = compile_request(point)
                        compile_module.render_factor_fft(request, hardware["sm"], "cpu-preflight", database)
                        sample = calibration_space.predicted_sample(point)
                    except (ValueError, TypeError, KeyError) as exc:
                        reasons.append({"code": "invalid-mapping-or-provider", "message": str(exc)})
                    else:
                        stages = []
                        for group in sample["execution_groups_json"]:
                            shared, threads = group["live_shared_bytes"], group["threads"]
                            stages.append(dict(log_size=group["stage_count"], threads=threads,
                                               dynamic_shared_lower_bound_bytes=shared))
                            if threads > hardware["max_threads_per_block"]:
                                reasons.append(dict(code="threads-exceed-hardware", group=group["index"],
                                                    requested=threads, limit=hardware["max_threads_per_block"]))
                            if shared > hardware["max_shared_per_block_optin_bytes"]:
                                reasons.append(dict(code="shared-lower-bound-exceeds-hardware", group=group["index"],
                                                    requested=shared, limit=hardware["max_shared_per_block_optin_bytes"]))
                        entry["resource_lower_bounds"] = dict(
                            stages=stages, max_dynamic_shared_lower_bound_bytes=max(
                                group["dynamic_shared_lower_bound_bytes"] for group in stages),
                            basis="depth=0 reuses input/work tile; depth>0 uses (depth + 1) input slots plus max(input_tile, provider scratch)",
                            status="necessary_condition_only", exact_register_shared_resources="pending",
                            compiler_resources_known=False,
                            provider_support=[database.check_both(hardware["sm"] * 10, case["precision"], 1 << size, ept)
                                              for size in partition])
                    if reasons:
                        entry["reasons"] = reasons
                        result["rejected"].append(entry)
                    else:
                        result["accepted"].append(entry)
    return result
