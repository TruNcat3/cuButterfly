#!/usr/bin/env python3
import argparse
import json
import pathlib


LAYOUTS = {"hermes-xor": "DataflowLayout::HermesXor", "linear": "DataflowLayout::Linear"}
STATES = {"inplace": "DataflowStateMode::InPlace", "ping-pong": "DataflowStateMode::PingPong"}
HANDOFFS = {"named-barrier": "StageHandoff::NamedBarrier", "atomic": "StageHandoff::Atomic"}
COMPUTE_UNITS = {"radix2": "ComputeUnit::Radix2", "radix4": "ComputeUnit::Radix4"}


def values(entry, key, default):
    value = entry.get(key, default)
    return value if isinstance(value, list) else [value]


def validate_point(point):
    (word, tile, stages, data, data_time, buffers, layout, state, handoff,
     role_stages, target_ctas, token_interleave) = point
    if word not in (32, 64) or tile not in range(5, 9) or stages not in range(2, 9):
        raise ValueError(f"unsupported generated point: {point}")
    if stages > tile or data not in (8, 16, 32) or data_time not in range(1, 17) or buffers not in (1, 2, 3):
        raise ValueError(f"illegal generated point: {point}")
    if role_stages < 1 or role_stages > stages:
        raise ValueError(f"illegal role fusion: {point}")
    if target_ctas not in (1, 2, 3):
        raise ValueError(f"illegal CTA residency target: {point}")
    if token_interleave not in (1, 2) or token_interleave > data_time:
        raise ValueError(f"illegal token interleave: {point}")
    if token_interleave > 1 and (role_stages == 1 or data_time < role_stages * token_interleave):
        raise ValueError(f"token interleave requires fused roles with enough packet tokens: {point}")
    if word == 64 and data > 16 or word == 32 and data < 16:
        raise ValueError(f"data-space/word-width mismatch: {point}")
    if layout not in LAYOUTS or state not in STATES or handoff not in HANDOFFS:
        raise ValueError(f"unknown generated policy: {point}")


def load_points(path):
    document = json.loads(path.read_text())
    if document.get("schema_version") != 1:
        raise ValueError("unsupported HybridDataflow schema")
    primary = document["primary"]
    points = set()
    for word_text, data_spaces in primary["data_space_by_word"].items():
        word = int(word_text)
        for tile in primary["flow_tile_log_n"]:
            for stages in primary["stage_space_by_tile"][str(tile)]:
                for data in data_spaces:
                    for data_time in primary["data_time"]:
                        for role_stages in primary.get("role_stages", [1]):
                            points.add((word, tile, stages, data, data_time, primary["pipeline_buffers"],
                                        primary["layout"], primary["state"], primary["handoff"], role_stages,
                                        primary.get("target_ctas_per_sm", 1), primary.get("token_interleave", 1)))
    for entry in document.get("ablations", []):
        for word in entry["word_bits"]:
            for data_time in values(entry, "data_time", 1):
                for role_stages in values(entry, "role_stages", 1):
                    for target_ctas in values(entry, "target_ctas_per_sm", 1):
                        for token_interleave in values(entry, "token_interleave", 1):
                            points.add((word, entry["flow_tile_log_n"], entry["stage_space"],
                                        entry["data_space_by_word"][str(word)], data_time,
                                        entry["pipeline_buffers"], entry["layout"], entry["state"], entry["handoff"],
                                        role_stages, target_ctas, token_interleave))
    for point in points:
        validate_point(point)
    return sorted(points)


def load_defaults(path, points):
    document = json.loads(path.read_text())
    defaults = []
    point_set = set(points)
    primary = document["primary"]
    for entry in document.get("defaults", []):
        for word in entry["word_bits"]:
            data_space = entry["data_space_by_word"][str(word)]
            point = (word, entry["flow_tile_log_n"], entry["stage_space"], data_space,
                     entry["data_time"], primary["pipeline_buffers"], primary["layout"],
                     primary["state"], primary["handoff"], entry["role_stages"],
                     entry.get("target_ctas_per_sm", 1), entry.get("token_interleave", 1))
            if point not in point_set:
                raise ValueError(f"default is not a generated point: {point}")
            compute_unit = entry.get("compute_unit", "radix2")
            if compute_unit not in COMPUTE_UNITS:
                raise ValueError(f"unsupported HybridDataflow default compute unit: {compute_unit}")
            defaults.append((word, entry["log_n"], entry["flow_tile_log_n"], entry["stage_space"],
                             data_space, entry.get("batch_min", 1), entry.get("batch_per_sm", 0),
                             entry.get("batch_offset", 0),
                             entry["role_stages"], entry["data_time"], entry.get("target_ctas_per_sm", 1),
                             entry.get("token_interleave", 1), compute_unit))
    return sorted(defaults, key=lambda value: (value[0], value[1], -value[6], -value[7], -value[5]))


def write_header(path, points, defaults=()):
    rows = []
    for (word, tile, stages, data, data_time, buffers, layout, state, handoff,
         role_stages, target_ctas, token_interleave) in points:
        rows.append(f"    {{{word}, {tile}, {stages}, {data}, {data_time}, {buffers}, {role_stages}, {target_ctas}, {token_interleave}, {LAYOUTS[layout]}, {STATES[state]}, {HANDOFFS[handoff]}}}")
    kernels = sorted({(point[1], point[2], point[4], point[9], point[10], point[11]) for point in points})
    kernel_rows = " \\\n".join(f"    X({tile}, {stages}, {data_time}, {role_stages}, {target_ctas}, {token_interleave})" for tile, stages, data_time, role_stages, target_ctas, token_interleave in kernels)
    default_rows = []
    for (word, log_n, tile, stages, data, batch_min, batch_per_sm, batch_offset, role_stages,
         data_time, target_ctas, token_interleave, compute_unit) in defaults:
        default_rows.append(f"    {{{word}, {log_n}, {tile}, {stages}, {data}, {batch_min}, {batch_per_sm}, {batch_offset}, {role_stages}, {data_time}, {target_ctas}, {token_interleave}, {COMPUTE_UNITS[compute_unit]}}}")
    rows_text = ",\n".join(rows)
    default_rows_text = ",\n".join(default_rows)
    path.write_text(f'''// Generated by scripts/generate_hybrid_dataflow.py. Do not edit.
#pragma once
#include <algorithm>
#include <array>
#include "cuntt/ntt.hpp"

#define CUNTT_FOR_EACH_HYBRID_DATAFLOW_KERNEL(X) \\
{kernel_rows}

namespace cuntt::detail {{
struct GeneratedHybridDataflowPoint {{
    std::uint32_t word_bits, flow_tile_log_n, stage_space, data_space, data_time, pipeline_buffers, role_stages, target_ctas_per_sm, token_interleave;
    DataflowLayout layout;
    DataflowStateMode state;
    StageHandoff handoff;
}};
inline constexpr std::array<GeneratedHybridDataflowPoint, {len(points)}> kGeneratedHybridDataflowPoints = {{{{
{rows_text}
}}}};
struct GeneratedHybridDataflowDefault {{
    std::uint32_t word_bits, log_n, flow_tile_log_n, stage_space, data_space;
    std::size_t batch_min;
    std::uint32_t batch_per_sm, batch_offset, role_stages, data_time, target_ctas_per_sm, token_interleave;
    ComputeUnit compute_unit;
}};
inline constexpr std::array<GeneratedHybridDataflowDefault, {len(defaults)}> kGeneratedHybridDataflowDefaults = {{{{
{default_rows_text}
}}}};
inline bool apply_generated_hybrid_dataflow_default(PlanConfig& config, std::uint32_t sm_count) noexcept {{
    for (const auto& point : kGeneratedHybridDataflowDefaults) {{
        const std::size_t threshold = std::max(point.batch_min,
            static_cast<std::size_t>(point.batch_per_sm) * sm_count + point.batch_offset);
        if (point.word_bits == config.word_bits && point.log_n == config.log_n &&
            point.flow_tile_log_n == config.flow_tile_log_n && point.stage_space == config.stage_space &&
            point.data_space == config.data_space && config.batch >= threshold) {{
            config.role_stages = point.role_stages;
            config.data_time = point.data_time;
            config.target_ctas_per_sm = point.target_ctas_per_sm;
            config.token_interleave = point.token_interleave;
            config.compute_unit = point.compute_unit;
            return true;
        }}
    }}
    return false;
}}
inline bool generated_hybrid_dataflow_available(const PlanConfig& config) noexcept {{
    for (const auto& point : kGeneratedHybridDataflowPoints) {{
        if (point.word_bits == config.word_bits && point.flow_tile_log_n == config.flow_tile_log_n &&
            point.stage_space == config.stage_space && point.data_space == config.data_space &&
            point.data_time == config.data_time &&
            point.role_stages == config.role_stages &&
            point.target_ctas_per_sm == config.target_ctas_per_sm &&
            point.token_interleave == config.token_interleave &&
            point.pipeline_buffers == config.pipeline_buffers && point.layout == config.dataflow_layout &&
            point.state == config.dataflow_state_mode && point.handoff == config.stage_handoff) return true;
    }}
    return false;
}}
}}  // namespace cuntt::detail
''')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=pathlib.Path, required=True)
    parser.add_argument("--header", type=pathlib.Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    points = load_points(args.spec)
    defaults = load_defaults(args.spec, points)
    if args.header:
        args.header.parent.mkdir(parents=True, exist_ok=True)
        write_header(args.header, points, defaults)
    elif not args.check:
        raise ValueError("provide --header or --check")


if __name__ == "__main__":
    main()
