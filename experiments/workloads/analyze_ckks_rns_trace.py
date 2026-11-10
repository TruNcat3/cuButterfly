#!/usr/bin/env python3
"""Read-only decomposition of application graphs in the N16384/B8 RNS trace."""
import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import sqlite3


def analyze(database, timings):
    connection = sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)
    rows = connection.execute("""
        SELECT k.graphNodeId,k.correlationId,k.start,k.end,n.value,rn.value,
               k.gridX,k.gridY,k.gridZ,k.blockX,k.blockY,k.blockZ,
               k.dynamicSharedMemory,k.registersPerThread
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        JOIN StringIds n ON n.id=k.demangledName
        JOIN CUPTI_ACTIVITY_KIND_RUNTIME r ON r.correlationId=k.correlationId
        JOIN StringIds rn ON rn.id=r.nameId
        WHERE k.graphNodeId IS NOT NULL AND rn.value LIKE 'cudaGraphLaunch%'
    """).fetchall()
    connection.close()
    graphs = defaultdict(list)
    for row in rows:
        graphs[row[0] >> 32].append(row)
    scenes = {112: "keyswitch", 160: "square_relinearize_rescale",
              272: "square_rescale_keyswitch", 88: "rotate_rows_shift1"}
    with timings.open(newline="") as source:
        csv_rows = {row["scene"]: row for row in csv.DictReader(source)}
    result = {"application_graphs": [], "scene_comparisons": [], "excluded_graphs": {}}
    excluded = Counter()
    excluded_kernel_ns = 0
    for graph_id, kernels in sorted(graphs.items()):
        launches = defaultdict(list)
        for row in kernels:
            launches[row[1]].append(row)
        nodes = len({row[0] for row in kernels})
        if nodes not in scenes or len(launches) < 28:
            excluded[(nodes, len(launches))] += 1
            excluded_kernel_ns += sum(row[3] - row[2] for row in kernels)
            continue
        assert all(len(launch) == nodes for launch in launches.values())
        categories, counts, other_signatures = Counter(), Counter(), Counter()
        for row in kernels:
            name = row[4]
            category = ("rns_first" if "hybrid2d_first_pass" in name else
                        "rns_second" if "hybrid2d_second_pass" in name else
                        "rns_fused_combined" if "cuntt::rns_fused_detail::stage_kernel" in name else
                        "gpu_ntt" if "gpuntt::" in name else "other")
            categories[category] += row[3] - row[2]
            counts[category] += 1
            if category == "other":
                other_signatures[(name, *row[6:])] += 1
        launch_count = len(launches)
        ordered = sorted(launches.values(), key=lambda launch: min(row[2] for row in launch))
        kernel_sums = [sum(row[3] - row[2] for row in launch) for launch in ordered]
        spans = [max(row[3] for row in launch) - min(row[2] for row in launch) for launch in ordered]
        replacement_categories = ("rns_first", "rns_second", "rns_fused_combined")
        replacement_count = sum(counts[key] for key in replacement_categories)
        assert not (replacement_count and counts["gpu_ntt"]), "mixed native/replacement NTT graph"
        assert replacement_count or counts["gpu_ntt"], "application graph has no recognized NTT"
        backend = "cubutterfly" if replacement_count else "native"
        ntt_ns = sum(categories[key] for key in (*replacement_categories, "gpu_ntt"))
        scene = scenes[nodes]
        assert csv_rows[scene]["N"] == "16384" and csv_rows[scene]["batch"] == "8"
        result["application_graphs"].append(dict(
            graph_id=graph_id, scene=scene, backend=backend, nodes=nodes, launches=launch_count,
            category_kernels_per_launch={key: value / launch_count for key, value in counts.items()},
            category_us_per_launch={key: value / launch_count / 1000 for key, value in categories.items()},
            ntt_sum_us=ntt_ns / launch_count / 1000,
            kernel_sum_us=sum(kernel_sums) / launch_count / 1000,
            gpu_span_us=sum(spans) / launch_count / 1000,
            last20_kernel_sum_us=sum(kernel_sums[-20:]) / 20000,
            diagnostic_csv_us=float(csv_rows[scene][backend + "_ms"]) * 1000,
            other_signatures=other_signatures))
    assert len(result["application_graphs"]) == 8
    for scene in scenes.values():
        pair = [graph for graph in result["application_graphs"] if graph["scene"] == scene]
        assert len(pair) == 2
        assert pair[0]["other_signatures"] == pair[1]["other_signatures"]
        by_backend = {graph["backend"]: graph for graph in pair}
        assert set(by_backend) == {"native", "cubutterfly"}
        native, replacement = by_backend["native"], by_backend["cubutterfly"]
        ntt_increase = replacement["ntt_sum_us"] - native["ntt_sum_us"]
        total_increase = replacement["kernel_sum_us"] - native["kernel_sum_us"]
        result["scene_comparisons"].append(dict(
            scene=scene, native_ntt_us=native["ntt_sum_us"],
            replacement_ntt_us=replacement["ntt_sum_us"],
            replacement_over_native_ntt=(replacement["ntt_sum_us"] / native["ntt_sum_us"]
                                         if native["ntt_sum_us"] else None),
            ntt_kernel_increase_us=ntt_increase,
            total_kernel_increase_us=total_increase,
            other_kernel_increase_us=total_increase - ntt_increase,
            ntt_fraction_of_kernel_increase=(ntt_increase / total_increase
                                            if total_increase else None)))
    for graph in result["application_graphs"]:
        del graph["other_signatures"]
    result["excluded_graphs"] = {f"{nodes} nodes/{launches} launches": count
                                 for (nodes, launches), count in sorted(excluded.items())}
    result["excluded_graph_kernel_ms"] = excluded_kernel_ns / 1e6
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database", type=Path)
    parser.add_argument("timings", type=Path)
    arguments = parser.parse_args()
    print(json.dumps(analyze(arguments.database, arguments.timings), indent=2, sort_keys=True))
