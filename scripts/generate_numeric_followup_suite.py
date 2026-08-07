#!/usr/bin/env python3
import argparse
import csv
import json
import math
import pathlib

from generate_numeric_regime_suite import expand


def _contract_key(case):
    return (case["operator"], case["precision"], case.get("accumulation", "native"))


def _point_distance(case, event, axis):
    log_distance = abs(int(case["logN"]) - int(event["logN"]))
    batch_distance = abs(math.log2(int(case["batch"])) - math.log2(int(event["batch"])))
    if axis == "batch":
        return (log_distance, batch_distance)
    return (batch_distance, log_distance)


def _axis_neighbors(cases, event, mapping):
    matching = [case for case in cases
                if _contract_key(case) == (event["operator"], event["precision"], event["accumulation"])
                and case["mapping_id"] == mapping]
    target_log = int(event["logN"])
    target_batch = int(event["batch"])
    if event["axis"] == "batch":
        line = [case for case in matching if int(case["logN"]) == target_log]
        line.sort(key=lambda case: int(case["batch"]))
        coordinate = lambda case: int(case["batch"])
        target = target_batch
    else:
        line = [case for case in matching if int(case["batch"]) == target_batch]
        line.sort(key=lambda case: int(case["logN"]))
        coordinate = lambda case: int(case["logN"])
        target = target_log
    if not line:
        line = sorted(matching, key=lambda case: _point_distance(case, event, event["axis"]))
        return line[:3]
    lower = [case for case in line if coordinate(case) < target]
    equal = [case for case in line if coordinate(case) == target]
    upper = [case for case in line if coordinate(case) > target]
    selected = lower[-1:] + equal[:1] + upper[:1]
    if not equal:
        selected = sorted(line, key=lambda case: abs(coordinate(case) - target))[:3]
    return selected


def select_followups(suite, events, include_ambiguous=False):
    selected = {}
    retained_events = []
    for event_index, event in enumerate(events, 1):
        if event.get("event_type") != "winner-crossover":
            continue
        if event.get("classification") != "confirmed" and not include_ambiguous:
            continue
        event_id = f"crossover-{event_index:03d}"
        retained_events.append({"id": event_id, **event})
        for mapping in (event["from_mapping"], event["to_mapping"]):
            for case in _axis_neighbors(suite["cases"], event, mapping):
                copy = dict(case)
                copy["tier"] = "full"
                reasons = list(selected.get(case["id"], {}).get("followup_reasons", []))
                reason = f"{event_id}:{event['axis']}:{event['classification']}"
                if reason not in reasons:
                    reasons.append(reason)
                copy["followup_reasons"] = reasons
                selected[case["id"]] = copy
    cases = sorted(selected.values(), key=lambda case: case["id"])
    cell_ids = {case["group"].rsplit("-b", 1)[0] for case in cases}
    cells = [cell for cell in suite.get("cells", []) if cell["id"] in cell_ids]
    return {
        "schema_version": 1, "study": "numeric-regime-confirmed-followups-v0.5",
        "hardware": suite["hardware"], "protocols": suite["protocols"],
        "binaries": suite["binaries"], "source_events": retained_events,
        "cells": cells, "cases": cases,
    }


def read_events(path):
    with path.open(newline="") as source:
        return list(csv.DictReader(source))


def main():
    parser = argparse.ArgumentParser(description="Generate focused full-protocol cases around measured crossovers.")
    parser.add_argument("--space", type=pathlib.Path,
                        default=pathlib.Path("config/v100_numeric_regime_space.json"))
    parser.add_argument("--events", type=pathlib.Path,
                        default=pathlib.Path("results/v100_numeric_regime_events.csv"))
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--include-ambiguous", action="store_true")
    args = parser.parse_args()
    suite = expand(json.loads(args.space.read_text()))
    followups = select_followups(suite, read_events(args.events), args.include_ambiguous)
    if not followups["cases"]:
        raise ValueError("no crossover follow-up cases were selected")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(followups, indent=2) + "\n")
    print(f"events={len(followups['source_events'])} cases={len(followups['cases'])}")


if __name__ == "__main__":
    main()
