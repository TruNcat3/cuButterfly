"""Serialize measured mappings, rather than a hand-maintained candidate-name ABI."""
import json
import math
import re
from hardware_registry import canonical_semantics


ENUM_FIELDS = {
    "backend": "parse_butterfly_backend", "compute_unit": "parse_compute_unit",
    "fft_core": "parse_fft_core", "complex_multiply": "parse_complex_multiply",
    "cross_twiddle": "parse_cross_twiddle_mode", "direct_boundary": "parse_direct_boundary",
    "local_exchange": "parse_local_exchange", "shared_layout": "parse_shared_layout",
    "stage_handoff": "parse_stage_handoff",
}
INT_FIELDS = (
    "tile_threads", "local_stages", "reorder_columns", "stage_space", "warp_stages",
    "pipeline_warps", "prefix_threads", "suffix_threads", "prefix_ept", "suffix_ept",
    "batch_tile_count",
)


def semantic_key(sample):
    """All workload semantics that may affect correctness or measured timing."""
    fields = ("operator", "precision", "placement", "logN", "batch", "modulus",
              "accumulation", "direction", "normalization", "element_stride",
              "batch_stride", "stage_matrices", "input_order", "output_order")
    canonical=canonical_semantics(sample)
    return tuple(str(canonical.get(field, "")) for field in fields)


def _list(sample, key):
    if key == "boundary_layouts" and sample.get(key):
        # The benchmark uses 'x' as its list separator, also present in 'prefix'.
        tokens = re.findall(r"prefix-tiled-transpose|tiled-transpose|direct-strided", sample[key])
        if "x".join(tokens) != sample[key]:
            raise ValueError("unknown measured boundary layout")
        return tokens
    return sample.get(key, "").split("x") if sample.get(key) else []


def mapping_statements(sample):
    if sample.get("mapping_json"):
        # Reuse the public mapping decoder for new lowerings and their axes.
        # The legacy field list below only serves older measurement records.
        mapping=json.dumps(json.loads(sample["mapping_json"]),sort_keys=True,separators=(",",":"))
        return [f"::cubutterfly::apply_serialized_mapping({json.dumps(mapping)}, config);"]
    lines = []
    lines.append(f"config.stage_overlap = {'true' if sample.get('stage_overlap', '0') == '1' else 'false'};")
    for field, parser in ENUM_FIELDS.items():
        if field in sample:
            lines.append(f"config.{field} = {parser}({json.dumps(sample[field])});")
    for field in INT_FIELDS:
        if field in sample:
            lines.append(f"config.{field} = {int(sample[field])}U;")
    matrices = _list(sample, "stage_matrices")
    if matrices:
        values = []
        for matrix in matrices:
            entries = matrix.split(":")
            if len(entries) != 4:
                raise ValueError("invalid measured stage matrix")
            values.append("{" + ",".join(entries) + "}")
        lines.append("config.stage_matrices = {" + ",".join(values) + "};")
    # Record logical segments and physical groups independently. Do not infer
    # that a segment boundary is a kernel launch or an on-chip exchange.
    if sample.get("backend") == "online-reorder":
        stages = [int(v) for v in _list(sample, "stages_per_decomposition")]
        lines.append("config.stage_partition = {" + ",".join(map(str, stages)) + "};")
        for target, thread_key, ept_key in (("segment_mappings", "segment_threads", "segment_ept"),
                                           ("execution_group_mappings", "group_threads", "group_ept")):
            threads, epts = _list(sample, thread_key), _list(sample, ept_key)
            if len(threads) != len(epts):
                raise ValueError("incomplete measured mapping")
            core_key = "segment_cores" if target == "segment_mappings" else "group_cores"
            cores = sample.get(core_key, "").split(":") if sample.get(core_key) else []
            if cores and len(cores) != len(threads):
                raise ValueError("incomplete measured processing-unit cores")
            values = []
            for i, (t, e) in enumerate(zip(threads, epts)):
                core = f"parse_fft_core({json.dumps(cores[i])})" if cores else "config.fft_core"
                values.append(f"{{{core}, config.local_exchange, {int(t)}U, {int(e)}U}}")
            lines.append(f"config.{target} = {{" + ",".join(values) + "};")
        twiddles = _list(sample, "boundary_twiddles")
        layouts = _list(sample, "boundary_layouts")
        residences = _list(sample, "boundary_residencies")
        if not len(twiddles) == len(layouts) == len(residences) == len(stages) - 1:
            raise ValueError("incomplete measured boundary mapping")
        values = [f"{{parse_cross_twiddle_mode({json.dumps(t)}), parse_direct_boundary({json.dumps(l)}), "
                  f"parse_fft_boundary_residency({json.dumps(r)})}}" for t, l, r in zip(twiddles, layouts, residences)]
        lines.append("config.boundaries = {" + ",".join(values) + "};")
    return lines


def semantic_condition(sample):
    inverse = sample.get("direction", "forward") == "inverse"
    conditions = [f"config.inverse == {'true' if inverse else 'false'}",
                  f"config.accumulation == parse_butterfly_accumulation({json.dumps(sample.get('accumulation', 'native'))})",
                  f"config.element_stride == {int(sample.get('element_stride', 1))}ULL",
                  f"(config.batch_stride ? config.batch_stride : ((std::size_t{{1}} << config.log_n) - 1) * config.element_stride + 1) == {int(sample.get('batch_stride', 1 << int(sample['logN'])))}ULL"]
    if inverse:
        conditions.append(f"config.normalize_inverse == {'true' if sample.get('normalization') == 'inverse' else 'false'}")
    matrices = _list(sample, "stage_matrices")
    conditions.append(f"config.stage_matrices.size() == {len(matrices)}")
    for i, matrix in enumerate(matrices):
        values = list(map(float, matrix.split(":")))
        if len(values) != 4 or not all(math.isfinite(v) for v in values):
            raise ValueError("invalid measured stage matrix")
        for field, value in zip(("m00", "m01", "m10", "m11"), values):
            conditions.append(f"config.stage_matrices[{i}].{field} == {value!r}")
    return " && ".join(conditions)


def select_points(operator_result):
    best = {}
    for candidate in operator_result.get("candidates", []):
        if candidate.get("status") != "measured" or not candidate.get("correct") or not candidate.get("samples"):
            continue
        sample = candidate["samples"][0]
        # Whole external libraries remain baselines, never cuButterfly winners.
        if sample.get("backend") == "cufft":
            continue
        latency = float(candidate["median_kernel_ms"])
        if not math.isfinite(latency) or latency <= 0:
            continue
        key = semantic_key(sample)
        if key not in best or latency < best[key]["median_kernel_ms"]:
            best[key] = candidate
    return [best[key] for key in sorted(best)]


def write_header(profile, operator_result, output):
    points = [point for point in select_points(operator_result) if point["samples"][0].get("operator") != "ntt"]
    major, minor = map(int, re.search(r"(\d+)\.(\d+)", str(profile["compute_capability"])).groups())
    target = f"{profile['device']} / sm_{major}{minor}"
    lines = ["// Generated measured mappings; candidate names are opaque identifiers.", "#pragma once",
             "#define CUBUTTERFLY_LOCAL_SELECTOR_DATA 1", "#include <array>", "#include <cstdint>",
             '#include "cuntt/butterfly.hpp"', '#include "cubutterfly/mapping.hpp"', "namespace cuntt::detail {",
             "struct LocalSelectorPoint { const char* operator_name; const char* precision; const char* placement; std::size_t log_n; std::size_t batch; std::uint64_t modulus; const char* candidate; double kernel_ms; };",
             f"inline constexpr const char* kLocalSelectorDeviceName = {json.dumps(profile['device'])};",
             f"inline constexpr const char* kLocalSelectorTarget = {json.dumps(target)};",
             f"inline constexpr int kLocalSelectorComputeMajor = {major};",
             f"inline constexpr int kLocalSelectorComputeMinor = {minor};",
             f"inline constexpr std::uint64_t kLocalSelectorMemoryBytes = {int(profile.get('global_memory_bytes', 0))}ULL;",
             f"inline constexpr std::array<LocalSelectorPoint, {len(points)}> kLocalSelectorPoints = {{"]
    for point in points:
        s = point["samples"][0]
        lines.append("LocalSelectorPoint{" + ",".join(json.dumps(s[k]) for k in ("operator", "precision", "placement")) +
                     f",{int(s['logN'])}ULL,{int(s['batch'])}ULL,{int(s.get('modulus', 0))}ULL," +
                     json.dumps(point["name"]) + f",{point['median_kernel_ms']:.17g}" + "},")
    lines += ["};", "inline bool local_semantics_match(const LocalSelectorPoint& point, const ButterflyConfig& config) {",
              "switch (&point - kLocalSelectorPoints.data()) {"]
    for i, point in enumerate(points):
        s = point["samples"][0]
        if s["operator"] != "ntt":
            lines.append(f"case {i}: return {semantic_condition(s)};")
    lines += ["default: return false;", "}", "}",
              "inline bool apply_measured_local_candidate(const LocalSelectorPoint& point, ButterflyConfig& config) {",
              "switch (&point - kLocalSelectorPoints.data()) {"]
    for i, point in enumerate(points):
        if point["samples"][0]["operator"] == "ntt":
            continue
        lines.append(f"case {i}:")
        lines.extend(mapping_statements(point["samples"][0]))
        lines.append("return true;")
    lines += ["default: return false;", "}", "}", "} // namespace cuntt::detail", ""]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines))
    return len(points)
