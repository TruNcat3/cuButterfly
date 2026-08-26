import csv
import importlib.util
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "analyze_v07_sector_attribution.py"
SPEC = importlib.util.spec_from_file_location("sector_attribution", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def write_aggregate(path):
    fields = [
        "label", "time_us", "dram_read_mib", "dram_write_mib",
        "global_load_sectors", "global_store_sectors", "active_warps_pct",
        "registers_per_thread", "shared_mem_bytes",
    ]
    values = {
        "v06": (1268.832, 277.755, 165.326, 29793054, 8388611, 47.78, 40, 16416),
        "warp128_vector_radix4_dual": (1424.224, 263.556, 139.315, 53402797, 8349008, 24.8, 106, 41184),
        "warp128_vector_radix4_reuse3_dual": (1489.088, 276.898, 174.223, 51801249, 8419276, 24.68, 103, 41184),
        "warp128_vector_radix4_reuse4_dual": (1475.136, 274.804, 174.412, 50041845, 8329365, 24.55, 105, 41184),
        "warp128_vector_radix4_reuse5_dual": (1385.216, 279.192, 159.841, 46456898, 8432822, 24.69, 103, 41184),
        "warp128_vector_radix4_reuse6_dual": (1345.696, 280.220, 144.663, 39989328, 8408173, 24.73, 106, 41184),
    }
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(fields)
        for label, metrics in values.items():
            writer.writerow((label,) + metrics)


def write_source(path, rows, internal_names=False):
    headers = [
        "File Name", "Line Number", "Address", "Source",
        "Access Operation", "Address Space",
        "L2 Theoretical Sectors Global",
        "L2 Theoretical Sectors Global Ideal",
        "L2 Theoretical Sectors Global Excessive", "Instructions Executed",
    ]
    if internal_names:
        headers = [
            "file", "line", "address", "source_code",
            "memory_access_type", "memory_type",
            "memory_l2_theoretical_sectors_global",
            "memory_l2_theoretical_sectors_global_ideal",
            "derived__memory_l2_theoretical_sectors_global_excessive",
            "inst_executed",
        ]
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["==PROF== Source page"])
        writer.writerow(headers)
        writer.writerows(rows)


def test_source_parser_accepts_ncu_labels_and_classifies_loads(tmp_path):
    source = tmp_path / "source.csv"
    write_source(source, [
        ("ntt.cu", "0", "0x100", "value = input[index]", "Load", "Global", 20, 10, 10, 5),
        ("ntt.cu", "0", "0x100", "inlined wrapper", "Load", "Global", 20, 10, 10, 5),
        ("ntt.cu", "0", "0x108", "lane_coefficient = coefficients[coefficient_base + lane]", "Load", "Global", 40, 20, 20, 8),
        ("ntt.cu", "0", "0x110", "destination[index] = value", "Store", "Global", 99, 99, 0, 4),
    ], internal_names=True)
    records = MODULE.read_source_csv(source)
    assert len(records) == 3
    totals = MODULE.aggregate_source(records, "v07-d6", ROOT)
    assert totals["external data"]["theoretical_sectors"] == 20
    assert totals["stage 2-6 coefficients"]["theoretical_excessive"] == 20


def test_source_parser_deduplicates_inline_frames_and_ignores_line_totals(tmp_path):
    source = tmp_path / "inline.csv"
    rows = [
        ["File Path", str(ROOT / "src/ntt.cu")],
        ["Function Name", "kernel"],
        ["Line No", "Source", "Address", "Source", "Instructions Executed",
         "Address Space", "Access Operation", "L1 Tag Requests Global",
         "L2 Theoretical Sectors Global Excessive",
         "L2 Theoretical Sectors Global", "L2 Theoretical Sectors Global Ideal"],
        ["465", "op.apply(stage, offset, left, right);", "-", "-", "20",
         "Global", "Load", "10", "5", "20", "15"],
        ["", "", "0x100", "LDG.E.SYS R2, [R2]", "10", "Global", "Load",
         "10", "5", "20", "15"],
        ["213", "twiddles[index]", "-", "-", "20", "Global", "Load",
         "10", "5", "20", "15"],
        ["", "", "0x100", "LDG.E.SYS R2, [R2]", "10", "Global", "Load",
         "10", "5", "20", "15"],
    ]
    with source.open("w", newline="") as stream:
        csv.writer(stream).writerows(rows)

    records = MODULE.read_source_csv(source)
    assert len(records) == 2
    totals = MODULE.aggregate_source(records, "v06", ROOT)
    assert totals["v0.6 coefficients"]["theoretical_sectors"] == 20


def test_model_explains_measured_delta_and_generates_report(tmp_path):
    aggregate = tmp_path / "summary.csv"
    source_v06 = tmp_path / "v06_source.csv"
    source_v07 = tmp_path / "v07_source.csv"
    output = tmp_path / "attribution.csv"
    pc_output = tmp_path / "source_pcs.csv"
    markdown = tmp_path / "analysis.md"
    write_aggregate(aggregate)
    write_source(source_v06, [
        ("ntt.cu", "0", "0x100", "value = input[index]", "Load", "Global", 100, 80, 20, 5),
        ("ntt.cu", "0", "0x108", "twiddles[index]", "Load", "Global", 200, 100, 100, 10),
    ])
    write_source(source_v07, [
        ("ntt.cu", "0", "0x200", "value = boundary[index]", "Load", "Global", 100, 80, 20, 5),
        ("ntt.cu", "0", "0x208", "stage0_coefficient = coefficients[0]", "Load", "Global", 50, 25, 25, 3),
        ("ntt.cu", "0", "0x210", "coefficients[511U + offset]", "Load", "Global", 250, 100, 150, 12),
    ])

    subprocess.run([
        sys.executable, str(SCRIPT), str(aggregate),
        "--source-v06", str(source_v06), "--source-v07", str(source_v07),
        "--csv", str(output), "--pc-csv", str(pc_output),
        "--markdown", str(markdown),
    ], check=True)

    text = markdown.read_text()
    assert "Source/SASS attribution passed the configured coverage threshold" in text
    assert "92.6%" in text
    assert "stage 7-9 coefficients" in text
    assert "not a four-phase token pipeline" in text
    assert "100.00%" in text
    rows = list(csv.DictReader(output.open()))
    predicted = {
        implementation: sum(float(row["predicted_sectors"]) for row in rows
                            if row["implementation"] == implementation)
        for implementation in ("v06", "v07-d6")
    }
    assert predicted["v07-d6"] - predicted["v06"] == 11010048
    pc_rows = list(csv.DictReader(pc_output.open()))
    assert {row["category"] for row in pc_rows} >= {
        "external data", "v0.6 coefficients", "stage 0-1 coefficients",
        "stage 7-9 coefficients",
    }
