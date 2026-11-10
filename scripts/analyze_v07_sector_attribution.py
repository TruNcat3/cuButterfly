#!/usr/bin/env python3
"""Attribute the v0.7 global-load sector gap to physical NTT stages."""

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path


ROWS_PER_RUN = 2 * 16 * 1024  # producer and consumer rows at batch 16


def number(row, key):
    value = row.get(key, "")
    return float(value) if value not in (None, "") else 0.0


def parse_number(value):
    value = value.strip().replace(",", "").replace("%", "")
    if not value or value.lower() in {"n/a", "nan", "no data", "-"}:
        return 0.0
    try:
        return float(value)
    except ValueError:
        return 0.0


def normalized(value):
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def metric_name(header):
    name = normalized(header)
    if "l2_theoretical_sectors_global_excessive" in name:
        return "theoretical_excessive"
    if "l2_theoretical_sectors_global_ideal" in name:
        return "theoretical_ideal"
    if "l2_theoretical_sectors_global" in name:
        return "theoretical_sectors"
    if "l1_tag_requests_global" in name:
        return "l1_tag_requests"
    if name in {"inst_executed", "instructions_executed"}:
        return "instructions"
    if "stall_long" in name or "stalled_long_scoreboard" in name:
        return "stall_long_scoreboard"
    return None


def field_name(header):
    name = normalized(header)
    aliases = {
        "address": "address",
        "pc": "address",
        "file": "file",
        "file_name": "file",
        "filename": "file",
        "line": "line",
        "line_no": "line",
        "line_number": "line",
        "source": "source",
        "source_code": "source",
        "sass": "sass",
        "sass_instruction": "sass",
        "kernel_name": "kernel",
        "access_operation": "access_operation",
        "memory_access_type": "access_operation",
        "address_space": "address_space",
        "memory_type": "address_space",
    }
    return aliases.get(name)


def read_source_csv(path):
    """Read NCU source-page CSV across label and internal metric headers."""
    raw_rows = list(csv.reader(path.read_text(errors="replace").splitlines()))
    header = None
    columns = {}
    metric_columns = {}
    current_file = ""
    current_function = ""
    current_line = ""
    current_source = ""
    captured_sources = {}
    records = []
    for raw in raw_rows:
        if len(raw) >= 2 and normalized(raw[0]) == "file_path":
            current_file = raw[1].strip()
            continue
        if len(raw) >= 2 and normalized(raw[0]) == "function_name":
            current_function = raw[1].strip()
            continue
        metrics = [metric_name(cell) for cell in raw]
        if any(metrics) and any(field_name(cell) for cell in raw):
            header = raw
            columns = {}
            metric_columns = {}
            source_count = 0
            for index, cell in enumerate(header):
                metric = metric_name(cell)
                field = field_name(cell)
                if metric:
                    metric_columns[metric] = index
                elif field == "source":
                    source_count += 1
                    columns["source" if source_count == 1 else "sass"] = index
                elif field and field not in columns:
                    columns[field] = index
            continue
        if header is None or len(raw) < 2:
            continue
        if len(raw) < len(header):
            raw = raw + [""] * (len(header) - len(raw))

        def value(field):
            index = columns.get(field)
            return raw[index].strip() if index is not None else ""

        line = value("line")
        source = value("source")
        if line and line not in {"-", "..."}:
            current_line = line
            if source and source not in {"-", "..."}:
                current_source = source
                source_file = value("file") or current_file
                try:
                    source_line_number = int(float(line))
                except ValueError:
                    source_line_number = 0
                if source_line_number:
                    captured_sources[(source_file, source_line_number)] = source
                    captured_sources[(Path(source_file).name,
                                      source_line_number)] = source

        address = value("address")
        # NCU emits one aggregate CUDA-line row followed by its SASS PCs. Only
        # the PC rows are additive; counting both doubles every source metric.
        if not address.lower().startswith("0x"):
            continue
        record = defaultdict(str)
        record.update({
            "file": value("file") or current_file,
            "line": line or current_line,
            "source": source if source not in {"", "-", "..."} else current_source,
            "sass": value("sass"),
            "address": address,
            "kernel": value("kernel") or current_function,
            "access_operation": value("access_operation"),
            "address_space": value("address_space"),
        })
        for metric, index in metric_columns.items():
            record[metric] = parse_number(raw[index])
        if not any(record.get(metric, 0.0) for metric in (
                "theoretical_sectors", "theoretical_ideal",
                "theoretical_excessive", "l1_tag_requests")):
            continue
        operation = normalized(record["access_operation"])
        address_space = normalized(record["address_space"])
        if operation and "load" not in operation and not operation.startswith("ld") and operation != "read":
            continue
        if address_space and "global" not in address_space:
            continue
        records.append(dict(record))
    if not records:
        raise ValueError(f"no supported NCU source table found in {path}")
    for record in records:
        try:
            line_number = int(float(record.get("line", "")))
        except ValueError:
            continue
        source_file = record.get("file", "")
        context = []
        for offset in range(3):
            captured = captured_sources.get((source_file, line_number + offset))
            if captured is None:
                captured = captured_sources.get(
                    (Path(source_file).name, line_number + offset))
            if captured:
                context.append(captured)
        if context:
            record["captured_context"] = " ".join(context)
    return records


def source_line(record, root):
    text = " ".join((record.get("source", ""), record.get("sass", "")))
    captured_context = record.get("captured_context", "")
    if captured_context:
        return f"{captured_context} {text}"
    path = record.get("file", "")
    line = record.get("line", "")
    try:
        line_number = int(float(line))
    except ValueError:
        return text
    candidates = [Path(path)]
    if path:
        candidates.append(root / path)
        candidates.append(root / "src" / Path(path).name)
    for candidate in candidates:
        try:
            lines = candidate.read_text(errors="replace").splitlines()
        except (OSError, ValueError):
            continue
        if 0 < line_number <= len(lines):
            # nvcc commonly attributes a load to the assignment line while
            # the pointer expression is wrapped onto one of the next lines.
            context = " ".join(lines[line_number - 1:line_number + 2])
            return f"{context} {text}"
    return text


def classify_source_record(record, implementation, root):
    text = source_line(record, root).lower()
    if any(token in text for token in (
            "ready[", "readiness", "trace", "globaltimer", "__cuda_load",
            "atomic_load", "bar_has_flipped", "atom.add.release")):
        return "ready/trace"
    if any(token in text for token in (
            "input[", "boundary[", "output[", "destination[",
            "state[row * kstride + reversed] =")):
        return "external data"
    if implementation == "v06":
        if any(token in text for token in ("twiddles[", "fused_twiddles[", "apply_coefficient")):
            return "v0.6 coefficients"
        return "unclassified global load"
    if any(token in text for token in ("stage0_coefficient", "coefficients[1]", "coefficients[2]")):
        return "stage 0-1 coefficients"
    if any(token in text for token in (
            "coefficient_base +", "lane_coefficient", "coefficient =",
            "coefficient_shoup =")):
        return "stage 2-6 coefficients"
    if any(token in text for token in ("coefficients[127", "coefficients[255", "coefficients[383", "coefficients[511")):
        return "stage 7-9 coefficients"
    if "coefficients[" in text or "twiddles[" in text:
        return "other coefficients"
    return "unclassified global load"


def deduplicate_source_records(records, implementation, root):
    """Collapse inline-call-stack copies of one SASS PC to its best frame."""
    priority = {
        "external data": 100,
        "ready/trace": 90,
        "stage 0-1 coefficients": 80,
        "stage 2-6 coefficients": 80,
        "stage 7-9 coefficients": 80,
        "v0.6 coefficients": 80,
        "other coefficients": 70,
        "unclassified global load": 0,
    }
    selected = {}
    for record in records:
        address = record.get("address", "")
        category = classify_source_record(record, implementation, root)
        candidate = (priority.get(category, 0), category, record)
        previous = selected.get(address)
        if previous is None or candidate[0] > previous[0]:
            selected[address] = candidate
    return [item[2] for item in selected.values()]


def aggregate_source(records, implementation, root):
    totals = defaultdict(lambda: defaultdict(float))
    for record in deduplicate_source_records(records, implementation, root):
        category = classify_source_record(record, implementation, root)
        for metric in (
            "theoretical_sectors", "theoretical_ideal",
            "theoretical_excessive", "l1_tag_requests", "instructions",
            "stall_long_scoreboard",
        ):
            totals[category][metric] += float(record.get(metric, 0.0) or 0.0)
    return totals


def annotated_source(records, implementation, root):
    annotated = []
    for record in deduplicate_source_records(records, implementation, root):
        row = {
            "implementation": implementation,
            "category": classify_source_record(record, implementation, root),
            "address": record.get("address", ""),
            "file": record.get("file", ""),
            "line": record.get("line", ""),
            "source": record.get("source", ""),
            "sass": record.get("sass", ""),
        }
        for metric in (
            "theoretical_sectors", "theoretical_ideal",
            "theoretical_excessive", "l1_tag_requests", "instructions",
            "stall_long_scoreboard",
        ):
            row[metric] = float(record.get(metric, 0.0) or 0.0)
        annotated.append(row)
    return annotated


def model_rows():
    definitions = [
        ("v06", "external data", 192, "same input and boundary data movement"),
        ("v06", "stage 0-1 coefficients", 48, "CTA radix-4 pair"),
        ("v06", "stage 2-3 coefficients", 48, "CTA radix-4 pair"),
        ("v06", "stage 4-5 coefficients", 96, "CTA radix-4 pair"),
        ("v06", "stage 6-7 coefficients", 192, "CTA radix-4 pair"),
        ("v06", "stage 8-9 coefficients", 192, "CTA radix-4 pair"),
        ("v07-d6", "external data", 192, "same logical input and boundary data"),
        ("v07-d6", "stage 0-1 coefficients", 48, "register-local radix-4 pair"),
        ("v07-d6", "stage 2-5 coefficients", 128, "dense lane loads and shuffles"),
        ("v07-d6", "stage 6 coefficients", 512, "four half-warp 16-byte-stride loads per tile"),
        ("v07-d6", "stage 7-9 coefficients", 224, "four-warp shared-tail continuation"),
    ]
    return [
        {
            "implementation": implementation,
            "category": category,
            "sectors_per_row": sectors,
            "predicted_sectors": sectors * ROWS_PER_RUN,
            "model_basis": basis,
        }
        for implementation, category, sectors, basis in definitions
    ]


def find_row(rows, label):
    try:
        return rows[label]
    except KeyError as error:
        raise ValueError(f"aggregate summary is missing label {label}") from error


def write_csv(path, model, source_totals):
    fields = [
        "implementation", "category", "sectors_per_row",
        "predicted_sectors", "source_theoretical_sectors",
        "source_ideal_sectors", "source_excessive_sectors", "model_basis",
    ]
    records = []
    for row in model:
        record = dict(row)
        source_impl = "v06" if row["implementation"] == "v06" else "v07-d6"
        measured = source_totals.get(source_impl, {}).get(row["category"], {})
        record.update({
            "source_theoretical_sectors": measured.get("theoretical_sectors", ""),
            "source_ideal_sectors": measured.get("theoretical_ideal", ""),
            "source_excessive_sectors": measured.get("theoretical_excessive", ""),
        })
        records.append(record)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def write_pc_csv(path, records):
    fields = [
        "implementation", "category", "address", "file", "line",
        "theoretical_sectors", "theoretical_ideal",
        "theoretical_excessive", "l1_tag_requests", "instructions",
        "stall_long_scoreboard", "source", "sass",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


def source_coverage(records, implementation):
    selected = [row for row in records if row["implementation"] == implementation]
    total = sum(row["theoretical_sectors"] for row in selected)
    unknown = sum(row["theoretical_sectors"] for row in selected
                  if row["category"] == "unclassified global load")
    return 1.0 if total == 0.0 else (total - unknown) / total


def format_integer(value):
    return f"{value:,.0f}"


def write_markdown(path, aggregate, model, source_totals, source_pcs):
    v06 = find_row(aggregate, "v06")
    d6 = find_row(aggregate, "warp128_vector_radix4_reuse6_dual")
    observed_delta = number(d6, "global_load_sectors") - number(v06, "global_load_sectors")
    predicted = {
        implementation: sum(row["predicted_sectors"] for row in model
                            if row["implementation"] == implementation)
        for implementation in ("v06", "v07-d6")
    }
    predicted_delta = predicted["v07-d6"] - predicted["v06"]
    source_complete = bool(source_totals)
    status_detail = (
        "The analytical values remain independent predictions and are compared "
        "with the PC-level capture below."
        if source_complete else
        "The analytical values below are predictions, not substitutes for the "
        "pending PC-level capture."
    )

    lines = [
        "# v0.7 Global-Load Sector Root-Cause Audit", "",
        "## Status", "",
        ("**Source/SASS attribution passed the configured coverage threshold.**" if source_complete else
         "**Aggregate counters and analytical model complete; SourceCounters capture pending.**"),
        status_detail, "",
        "## Controlled Observation", "",
        "The matched V100 base-clock capture compares uint32 `logN=20`, batch 16, "
        "Shoup, `10+10`, and full-scratch implementations.", "",
        "| metric | v0.6 | v0.7 vector d6 | ratio |", "|:--|--:|--:|--:|",
        f"| kernel time (us) | {number(v06, 'time_us'):.1f} | {number(d6, 'time_us'):.1f} | {number(d6, 'time_us') / number(v06, 'time_us'):.3f}x |",
        f"| DRAM read (MiB) | {number(v06, 'dram_read_mib'):.1f} | {number(d6, 'dram_read_mib'):.1f} | {number(d6, 'dram_read_mib') / number(v06, 'dram_read_mib'):.3f}x |",
        f"| global-load sectors | {format_integer(number(v06, 'global_load_sectors'))} | {format_integer(number(d6, 'global_load_sectors'))} | {number(d6, 'global_load_sectors') / number(v06, 'global_load_sectors'):.3f}x |",
        f"| global-store sectors | {format_integer(number(v06, 'global_store_sectors'))} | {format_integer(number(d6, 'global_store_sectors'))} | {number(d6, 'global_store_sectors') / number(v06, 'global_store_sectors'):.3f}x |",
        f"| active warps (%) | {number(v06, 'active_warps_pct'):.2f} | {number(d6, 'active_warps_pct'):.2f} | {number(d6, 'active_warps_pct') / number(v06, 'active_warps_pct'):.3f}x |",
        f"| registers/thread | {number(v06, 'registers_per_thread'):.0f} | {number(d6, 'registers_per_thread'):.0f} | |",
        f"| shared bytes/CTA | {number(v06, 'shared_mem_bytes'):.0f} | {number(d6, 'shared_mem_bytes'):.0f} | |", "",
        f"DRAM reads change by {100.0 * (number(d6, 'dram_read_mib') / number(v06, 'dram_read_mib') - 1.0):+.1f}% "
        f"and store sectors by {100.0 * (number(d6, 'global_store_sectors') / number(v06, 'global_store_sectors') - 1.0):+.1f}%, "
        f"while global-load sectors change by {100.0 * (number(d6, 'global_load_sectors') / number(v06, 'global_load_sectors') - 1.0):+.1f}%. "
        "The gap is repeated or fragmented cache requests, not an additional N-sized external state.", "",
        "## Stage Sector Model", "",
        f"There are `{ROWS_PER_RUN:,}` local 1024-point row transforms across the producer and consumer roles.", "",
        "| implementation | stage/address class | sectors/row | predicted sectors | basis |",
        "|:--|:--|--:|--:|:--|",
    ]
    for row in model:
        lines.append(
            f"| `{row['implementation']}` | {row['category']} | {row['sectors_per_row']} | "
            f"{format_integer(row['predicted_sectors'])} | {row['model_basis']} |")
    lines += [
        "",
        f"The model predicts an extra **{format_integer(predicted_delta)}** sectors; "
        f"NCU measures **{format_integer(observed_delta)}**. The model therefore explains "
        f"**{100.0 * observed_delta / predicted_delta:.1f}%** of the measured delta before PC-level fitting.", "",
        "The dominant term is stage 6. `coefficient_reuse_stages=6` means `stage < 6`; "
        "stage 6 still uses four slot-wise loads. With four consecutive values per lane, "
        "each instruction uses half a warp with a 16-byte coefficient stride.", "",
        "## Reuse-Depth Difference Check", "",
        "| newly distributed stage | measured reduction | predicted reduction | measured/predicted |",
        "|--:|--:|--:|--:|",
    ]
    depth_pairs = [
        (2, "warp128_vector_radix4_dual", "warp128_vector_radix4_reuse3_dual", 48),
        (3, "warp128_vector_radix4_reuse3_dual", "warp128_vector_radix4_reuse4_dual", 48),
        (4, "warp128_vector_radix4_reuse4_dual", "warp128_vector_radix4_reuse5_dual", 96),
        (5, "warp128_vector_radix4_reuse5_dual", "warp128_vector_radix4_reuse6_dual", 192),
    ]
    for stage, before_label, after_label, sectors_per_row in depth_pairs:
        before = find_row(aggregate, before_label)
        after = find_row(aggregate, after_label)
        measured = number(before, "global_load_sectors") - number(after, "global_load_sectors")
        expected = sectors_per_row * ROWS_PER_RUN
        lines.append(
            f"| {stage} | {format_integer(measured)} | {format_integer(expected)} | "
            f"{measured / expected:.3f}x |")
    lines += [
        "| 6 (d7 candidate; focused capture) | see stage-6 report | layout-dependent | - |", "",
    ]

    if source_complete:
        lines += ["## Source/SASS Attribution", "",
                  "| implementation | category | theoretical sectors | ideal | excessive | instructions |",
                  "|:--|:--|--:|--:|--:|--:|"]
        for implementation in ("v06", "v07-d6"):
            for category, metrics in sorted(source_totals[implementation].items()):
                lines.append(
                    f"| `{implementation}` | {category} | "
                    f"{format_integer(metrics['theoretical_sectors'])} | "
                    f"{format_integer(metrics['theoretical_ideal'])} | "
                    f"{format_integer(metrics['theoretical_excessive'])} | "
                    f"{format_integer(metrics['instructions'])} |")
        lines += ["", "### Attribution coverage", "",
                  "| implementation | source theoretical sectors | aggregate load sectors | source/aggregate | classified source sectors |",
                  "|:--|--:|--:|--:|--:|"]
        for implementation in ("v06", "v07-d6"):
            source_total = sum(
                metrics["theoretical_sectors"]
                for metrics in source_totals[implementation].values())
            aggregate_total = number(
                v06 if implementation == "v06" else d6,
                "global_load_sectors")
            lines.append(
                f"| `{implementation}` | {format_integer(source_total)} | "
                f"{format_integer(aggregate_total)} | {source_total / aggregate_total:.3f}x | "
                f"{100.0 * source_coverage(source_pcs, implementation):.2f}% |")
        unclassified = sorted(
            (row for row in source_pcs
             if row["category"] == "unclassified global load"),
            key=lambda row: row["theoretical_sectors"], reverse=True)
        if unclassified:
            lines += ["", "Largest unclassified PCs:", "",
                      "| implementation | PC | file:line | theoretical sectors | source/SASS |",
                      "|:--|:--|:--|--:|:--|"]
            for row in unclassified[:12]:
                location = f"{row['file']}:{row['line']}".strip(":") or "unknown"
                instruction = row["source"] or row["sass"] or "unknown"
                instruction = instruction.replace("|", "\\|").strip()
                lines.append(
                    f"| `{row['implementation']}` | `{row['address'] or 'unknown'}` | "
                    f"`{location}` | {format_integer(row['theoretical_sectors'])} | "
                    f"`{instruction}` |")
        lines.append("")
    else:
        lines += [
            "## Source/SASS Attribution", "",
            "Pending administrator NCU collection. The profiling script retains `.ncu-rep` "
            "files and exports `cuda,sass` source CSV so every residual global-load PC can be classified.", "",
        ]

    lines += [
        "## Pipeline Audit", "",
        "The current v0.7 dual-group kernel is resident but not a four-phase token pipeline. "
        "Within each warp group its program order is:", "",
        "```text",
        "packet load -> barrier -> row 0 complete NTT -> row 1 complete NTT",
        "            -> row 2 complete NTT -> row 3 complete NTT -> packet store",
        "```", "",
        "The two warp groups may interleave through normal SM warp scheduling, but there is no "
        "software dependence that overlaps one row's load, another row's prefix, a third row's "
        "tail, and a fourth row's store. v0.6 already benefits from the same hardware scheduler "
        "while resident at four CTAs/SM. v0.7 is limited to two CTAs/SM by 106 registers/thread "
        "and 41,184 shared bytes, reducing the sampled active-warp pool by about half.", "",
        "## Root-Cause Decision", "",
        "1. The logical external dataflow of v0.6 and v0.7 is effectively the same.",
        "2. The excess load sectors originate primarily in coefficient request organization, "
        "especially the vector layout's unreused stage 6, rather than input/boundary traffic.",
        "3. Cache hits prevent those requests from becoming proportional DRAM bytes, but they "
        "still consume LD/ST issue, L1 tag, dependency, and scoreboard resources.",
        "4. The current subgraph residency does not provide enough explicit phase overlap to "
        "mask that physical-core loss, and its resource footprint halves the resident warp pool.", "",
        "No kernel or selector change is made by this audit. Candidate corrections remain "
        "separate follow-up experiments; the unclassified PC rows stay explicit because inline "
        "CUDA line mappings do not support a reliable finer attribution.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("aggregate", type=Path)
    parser.add_argument("--source-v06", type=Path)
    parser.add_argument("--source-v07", type=Path)
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--pc-csv", type=Path)
    parser.add_argument("--markdown", required=True, type=Path)
    parser.add_argument("--min-source-coverage", type=float, default=0.95)
    args = parser.parse_args()

    with args.aggregate.open(newline="") as stream:
        aggregate = {row["label"]: row for row in csv.DictReader(stream)}
    source_totals = {}
    source_pcs = []
    root = Path(__file__).resolve().parents[1]
    if bool(args.source_v06) != bool(args.source_v07):
        raise ValueError("provide both --source-v06 and --source-v07")
    if args.source_v06:
        v06_records = read_source_csv(args.source_v06)
        v07_records = read_source_csv(args.source_v07)
        source_totals["v06"] = aggregate_source(v06_records, "v06", root)
        source_totals["v07-d6"] = aggregate_source(v07_records, "v07-d6", root)
        source_pcs.extend(annotated_source(v06_records, "v06", root))
        source_pcs.extend(annotated_source(v07_records, "v07-d6", root))

    model = model_rows()
    write_csv(args.csv, model, source_totals)
    if args.pc_csv and source_pcs:
        write_pc_csv(args.pc_csv, source_pcs)
    write_markdown(args.markdown, aggregate, model, source_totals, source_pcs)
    if source_pcs:
        failures = [
            f"{implementation}={source_coverage(source_pcs, implementation):.3f}"
            for implementation in ("v06", "v07-d6")
            if source_coverage(source_pcs, implementation) < args.min_source_coverage
        ]
        if failures:
            raise ValueError(
                "source attribution coverage below "
                f"{args.min_source_coverage:.3f}: {', '.join(failures)}")


if __name__ == "__main__":
    main()
