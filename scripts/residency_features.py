#!/usr/bin/env python3
import math


RESOURCE_FEATURE_FIELDS = (
    "warps_per_cta", "temporal_state_words_per_thread", "allocated_registers_per_cta",
    "allocated_shared_bytes_per_cta", "register_file_fraction_per_cta", "register_cta_limit",
    "shared_cta_limit", "thread_cta_limit", "warp_cta_limit", "resident_ctas_per_sm",
    "resident_warps_per_sm", "occupancy_upper_bound", "total_ctas", "grid_waves",
    "grid_concurrent_fraction", "limiting_resource", "hardware_feasible",
    "residency_drop_from_previous", "residency_ratio_from_previous", "occupancy_drop_from_previous",
    "occupancy_ratio_from_previous", "register_residency_drop_from_previous",
    "crosses_single_cta_boundary", "resource_cliff",
)


def ceil_div(value, divisor):
    return (value + divisor - 1) // divisor


def round_up(value, unit):
    return ceil_div(value, unit) * unit


def derive_residency_features(hardware, threads_per_cta, registers_per_thread,
                              shared_bytes_per_cta, total_ctas,
                              temporal_state_words_per_thread=0,
                              previous=None):
    if min(threads_per_cta, registers_per_thread, total_ctas) < 1 or shared_bytes_per_cta < 0:
        raise ValueError("threads, registers, and CTAs must be positive; shared bytes must be nonnegative")

    warp_size = hardware["warp_size"]
    warps_per_cta = ceil_div(threads_per_cta, warp_size)
    registers_per_warp = round_up(
        registers_per_thread * warp_size,
        hardware.get("register_allocation_unit_per_warp", 1),
    )
    allocated_registers = registers_per_warp * warps_per_cta
    allocated_shared = round_up(
        shared_bytes_per_cta,
        hardware.get("shared_allocation_unit_per_cta", 1),
    ) if shared_bytes_per_cta else 0

    legal = (threads_per_cta <= hardware.get("max_threads_per_cta", threads_per_cta) and
             allocated_registers <= hardware.get("max_registers_per_cta", hardware["registers_per_sm"]) and
             allocated_shared <= hardware.get("max_shared_bytes_per_cta", hardware["shared_bytes_per_sm"]))
    limits = {
        "register": hardware["registers_per_sm"] // allocated_registers,
        "shared": (hardware["shared_bytes_per_sm"] // allocated_shared
                   if allocated_shared else hardware["max_ctas_per_sm"]),
        "thread": hardware["threads_per_sm"] // threads_per_cta,
        "warp": hardware["warps_per_sm"] // warps_per_cta,
        "hardware": hardware["max_ctas_per_sm"],
    }
    resident_ctas = min(limits.values()) if legal else 0
    limiting = "+".join(name for name, value in limits.items() if value == resident_ctas) if legal else "illegal"
    resident_warps = resident_ctas * warps_per_cta
    occupancy = resident_warps / hardware["warps_per_sm"]
    concurrent_ctas = hardware["sm_count"] * resident_ctas
    waves = total_ctas / concurrent_ctas if concurrent_ctas else math.inf

    previous_resident = previous["resident_ctas_per_sm"] if previous else resident_ctas
    previous_occupancy = previous["occupancy_upper_bound"] if previous else occupancy
    residency_drop = max(0, previous_resident - resident_ctas)
    residency_ratio = previous_resident / resident_ctas if resident_ctas else math.inf
    occupancy_drop = max(0.0, previous_occupancy - occupancy)
    occupancy_ratio = previous_occupancy / occupancy if occupancy else math.inf
    register_limit_previous = previous["register_cta_limit"] if previous else limits["register"]
    register_residency_drop = max(0, register_limit_previous - limits["register"])

    return {
        "warps_per_cta": warps_per_cta,
        "temporal_state_words_per_thread": temporal_state_words_per_thread,
        "allocated_registers_per_cta": allocated_registers,
        "allocated_shared_bytes_per_cta": allocated_shared,
        "register_file_fraction_per_cta": allocated_registers / hardware["registers_per_sm"],
        "register_cta_limit": limits["register"],
        "shared_cta_limit": limits["shared"],
        "thread_cta_limit": limits["thread"],
        "warp_cta_limit": limits["warp"],
        "resident_ctas_per_sm": resident_ctas,
        "resident_warps_per_sm": resident_warps,
        "occupancy_upper_bound": occupancy,
        "total_ctas": total_ctas,
        "grid_waves": waves,
        "grid_concurrent_fraction": min(1.0, waves),
        "limiting_resource": limiting,
        "hardware_feasible": int(resident_ctas > 0),
        "residency_drop_from_previous": residency_drop,
        "residency_ratio_from_previous": residency_ratio,
        "occupancy_drop_from_previous": occupancy_drop,
        "occupancy_ratio_from_previous": occupancy_ratio,
        "register_residency_drop_from_previous": register_residency_drop,
        "crosses_single_cta_boundary": int(previous_resident > 1 and resident_ctas == 1),
        "resource_cliff": int(occupancy_ratio >= 1.5 or (previous_resident > 1 and resident_ctas == 1)),
    }
