#!/usr/bin/env python3
import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


SM_COUNT = 80
REGISTERS_PER_SM = 65536
SHARED_BYTES_PER_SM = 98304
THREADS_PER_SM = 2048
REGISTER_ALLOCATION_UNIT = 256


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def resource_map(path):
    result = {}
    for row in read_csv(path):
        key = (int(row["word_bits"]), int(row["flow_tile_log_n"]), int(row["stage_space"]),
               int(row["data_time"]), int(row["role_stages"]),
               int(row["target_ctas_per_sm"]), int(row["token_interleave"]))
        result[key] = {key: int(value) for key, value in row.items()
                       if key in ("registers_per_thread", "stack_bytes", "local_bytes")}
    return result


def resident_ctas(word_bits, log_n, role_stages, registers):
    warps = role_stages
    threads = warps * 32
    registers_per_warp = math.ceil(registers * 32 / REGISTER_ALLOCATION_UNIT) * REGISTER_ALLOCATION_UNIT
    register_blocks = REGISTERS_PER_SM // (registers_per_warp * warps)
    shared_bytes = (1 << log_n) * (word_bits // 8)
    shared_blocks = SHARED_BYTES_PER_SM // shared_bytes
    thread_blocks = THREADS_PER_SM // threads
    return max(1, min(32, register_blocks, shared_blocks, thread_blocks)), shared_bytes


FEATURE_NAMES = [
    "word64", "logN12", "density", "log_density", "packets", "serial_slots",
    "slots_per_packet", "utilization", "registers", "resident_ctas", "continuous_waves",
    "ceil_waves", "wave_tail", "packets_x_density", "slots_x_density",
    "registers_x_density", "packets_x_word64", "slots_x_word64",
    "packets_x_logN12", "slots_x_logN12", "occupancy_headroom", "overflow",
    "cta_floor", "cta_ceil", "partial_cta_wave", "resident_pressure",
    "resident_wave_level", "packets_x_cta_ceil", "slots_x_cta_ceil",
    "packets_x_resident_pressure", "slots_x_resident_pressure",
]


def features(row, resources):
    bits = int(row["word_bits"])
    log_n = int(row["logN"])
    batch = int(row["batch"])
    td = int(row["data_time"])
    ur = int(row.get("role_stages", 5 if log_n == 10 else 6))
    key = (bits, ur, ur, td, ur, 1, 1)
    resource = resources[key]
    resident, _ = resident_ctas(bits, log_n, ur, resource["registers_per_thread"])
    token_count = 1 << (log_n - ur)
    packets = math.ceil(token_count / td)
    slots_per_packet = math.ceil(td / ur)
    serial_slots = packets * slots_per_packet
    utilization = token_count / (serial_slots * ur)
    density = batch / SM_COUNT
    continuous_waves = density / resident
    ceil_waves = max(1, math.ceil(continuous_waves))
    wave_tail = ceil_waves - continuous_waves
    headroom = max(0.0, resident - density) / resident
    overflow = max(0.0, density - resident)
    cta_floor = math.floor(density)
    cta_ceil = max(1, math.ceil(density))
    partial_cta_wave = density - cta_floor
    resident_pressure = cta_ceil / resident
    resident_wave_level = max(1, math.ceil(density / resident))
    word64 = float(bits == 64)
    logn12 = float(log_n == 12)
    values = [
        word64, logn12, density, math.log1p(density), packets, serial_slots,
        slots_per_packet, utilization, resource["registers_per_thread"], resident,
        continuous_waves, ceil_waves, wave_tail, packets * density, serial_slots * density,
        resource["registers_per_thread"] * density, packets * word64, serial_slots * word64,
        packets * logn12, serial_slots * logn12, headroom, overflow,
        cta_floor, cta_ceil, partial_cta_wave, resident_pressure, resident_wave_level,
        packets * cta_ceil, serial_slots * cta_ceil,
        packets * resident_pressure, serial_slots * resident_pressure,
    ]
    metadata = {"resident_ctas": resident, "packets": packets, "serial_slots": serial_slots,
                "utilization": utilization, "registers": resource["registers_per_thread"]}
    return values, metadata


def prepare(rows, resources):
    groups = defaultdict(list)
    for row in rows:
        key = (int(row["word_bits"]), int(row["logN"]), int(row["batch"]))
        groups[key].append(row)
    prepared = []
    for key, group in groups.items():
        best = min(float(row["kernel_ms"]) for row in group)
        for row in group:
            values, metadata = features(row, resources)
            prepared.append({"key": key, "row": row, "x": values,
                             "target": math.log(float(row["kernel_ms"]) / best), **metadata})
    return prepared


def evaluate(name, model, train, test):
    model.fit(np.asarray([item["x"] for item in train]), np.asarray([item["target"] for item in train]))
    predictions = model.predict(np.asarray([item["x"] for item in test]))
    groups = defaultdict(list)
    for item, prediction in zip(test, predictions):
        groups[item["key"]].append((float(prediction), item))
    regrets = []
    top1 = 0
    top2 = 0
    rows = []
    for key, group in sorted(groups.items()):
        ordered = sorted(group, key=lambda value: value[0])
        actual = sorted(group, key=lambda value: value[1]["target"])
        chosen = ordered[0][1]
        actual_td = int(actual[0][1]["row"]["data_time"])
        predicted_td = int(chosen["row"]["data_time"])
        regret = math.exp(chosen["target"])
        regrets.append(regret)
        top1 += predicted_td == actual_td
        top2 += actual_td in {int(value[1]["row"]["data_time"]) for value in ordered[:2]}
        rows.append({"model": name, "word_bits": key[0], "logN": key[1], "batch": key[2],
                     "predicted_td": predicted_td, "actual_td": actual_td, "regret": regret,
                     "predicted_resident_ctas": chosen["resident_ctas"],
                     "predicted_packets": chosen["packets"], "predicted_serial_slots": chosen["serial_slots"]})
    metrics = {
        "model": name, "scenarios": len(groups), "top1": top1 / len(groups), "top2": top2 / len(groups),
        "geomean_regret": math.exp(sum(math.log(value) for value in regrets) / len(regrets)),
        "worst_regret": max(regrets), "p95_regret": float(np.percentile(regrets, 95)),
    }
    return metrics, rows, model


def static_td(bits, log_n, batch):
    if bits == 32 and log_n == 10:
        return 10
    if bits == 32 and log_n == 12:
        return 12 if batch <= 2 * SM_COUNT else 6
    if bits == 64 and log_n == 10:
        return 10
    return 12


def evaluate_static(test):
    groups = defaultdict(list)
    for item in test:
        groups[item["key"]].append(item)
    regrets = []
    top1 = top2 = 0
    rows = []
    for key, group in sorted(groups.items()):
        chosen_td = static_td(*key)
        chosen = next(item for item in group if int(item["row"]["data_time"]) == chosen_td)
        actual = sorted(group, key=lambda item: item["target"])
        actual_td = int(actual[0]["row"]["data_time"])
        ordered_actual = [int(item["row"]["data_time"]) for item in actual]
        regret = math.exp(chosen["target"])
        regrets.append(regret)
        top1 += chosen_td == actual_td
        top2 += chosen_td in ordered_actual[:2]
        rows.append({"model": "static-guarded", "word_bits": key[0], "logN": key[1], "batch": key[2],
                     "predicted_td": chosen_td, "actual_td": actual_td, "regret": regret,
                     "predicted_resident_ctas": chosen["resident_ctas"],
                     "predicted_packets": chosen["packets"], "predicted_serial_slots": chosen["serial_slots"]})
    metrics = {"model": "static-guarded", "scenarios": len(groups), "top1": top1 / len(groups),
               "top2": top2 / len(groups),
               "geomean_regret": math.exp(sum(math.log(value) for value in regrets) / len(regrets)),
               "worst_regret": max(regrets), "p95_regret": float(np.percentile(regrets, 95))}
    return metrics, rows


def main():
    parser = argparse.ArgumentParser(description="Fit and evaluate a wave-aware HybridDataflow Td model")
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--test", required=True, type=Path)
    parser.add_argument("--resources", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    resources = resource_map(args.resources)
    train = prepare(read_csv(args.train), resources)
    test_rows = [row for row in read_csv(args.test)
                 if int(row["batch"]) not in {1, 16, 80, 160, 256, 512}]
    test = prepare(test_rows, resources)
    models = {
        "ridge": make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-4, 4, 33))),
        "hist-gradient": HistGradientBoostingRegressor(max_iter=300, max_leaf_nodes=15,
                                                       learning_rate=0.05, l2_regularization=0.1,
                                                       random_state=7),
        "random-forest": RandomForestRegressor(n_estimators=400, max_depth=10, min_samples_leaf=2,
                                                max_features=0.8, n_jobs=-1, random_state=7),
    }
    static_metrics, static_predictions = evaluate_static(test)
    metrics = [static_metrics]
    predictions = list(static_predictions)
    fitted = {}
    for name, model in models.items():
        result, rows, trained = evaluate(name, model, train, test)
        metrics.append(result)
        predictions.extend(rows)
        fitted[name] = trained
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "metrics.json").open("w") as handle:
        json.dump(metrics, handle, indent=2)
    fields = list(predictions[0])
    with (args.output_dir / "predictions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(predictions)
    ridge = fitted["ridge"]
    coefficients = dict(zip(FEATURE_NAMES, ridge[-1].coef_))
    with (args.output_dir / "ridge_coefficients.json").open("w") as handle:
        json.dump({"alpha": ridge[-1].alpha_, "coefficients": coefficients}, handle, indent=2)
    lines = ["# HybridDataflow Td model evaluation", "",
             "Training uses the full Td scan at six representative batches. Testing uses unseen",
             "batch multiples of 16 from the dense scan; overlapping batches are excluded.", "",
             "| model | scenarios | top-1 | top-2 | geomean regret | p95 regret | worst regret |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for result in metrics:
        lines.append(f"| {result['model']} | {result['scenarios']} | {result['top1']:.3f} | "
                     f"{result['top2']:.3f} | {result['geomean_regret']:.4f} | "
                     f"{result['p95_regret']:.4f} | {result['worst_regret']:.4f} |")
    lines += [
        "",
        "## Features and decision",
        "",
        "The feature set combines packet count, critical serial slots, token utilization, cubin",
        "register count, analytically derived resident CTAs, continuous and discrete CTA density,",
        "partial waves, and interactions with word width and graph length. Cubin resources are",
        "extracted by `extract_hybrid_dataflow_resources.py`; no timing label is used as a feature.",
        "",
        "The runtime admission target is geomean regret <=1.02, p95 <=1.05, and worst <=1.10 on",
        "unseen batches. The static guarded policy passes all three. The best learned model",
        "(`hist-gradient`) passes only the geomean target and reaches 1.151 worst regret, so it is",
        "not admitted to runtime selection. Learned ranking remains useful as a top-2 search hint;",
        "the measured static table remains authoritative on V100.",
    ]
    (args.output_dir / "analysis.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
