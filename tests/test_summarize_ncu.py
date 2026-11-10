import csv
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "summarize_ncu.py"


class SummarizeNcuTests(unittest.TestCase):
    def test_skips_derived_summary_but_parses_raw_report(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            raw = directory / "kernel.csv"
            raw.write_text(
                '==PROF== Connected\n'
                '"ID","Kernel Name","Grid Size","Block Size","Metric Name","Metric Value"\n'
                '"1","kernel()","(1,1,1)","(32,1,1)",'
                '"gpu__time_duration.sum","123000"\n'
                '"1","kernel()","(1,1,1)","(32,1,1)",'
                '"l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum","17"\n'
            )
            derived = directory / "summary.csv"
            derived.write_text("label,kernel_id,kernel_name\nold,1,kernel\n")
            output = directory / "out.csv"

            result = subprocess.run(
                ["python3", str(SCRIPT), str(raw), str(derived),
                 "--output", str(output)],
                check=True, capture_output=True, text=True,
            )

            self.assertIn("skip derived CSV", result.stderr)
            with output.open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["label"], "kernel")
            self.assertAlmostEqual(float(rows[0]["time_us"]), 123.0)
            self.assertAlmostEqual(float(rows[0]["local_load_sectors"]), 17.0)

    def test_rejects_unknown_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            unknown = Path(directory) / "unknown.csv"
            unknown.write_text("unexpected,header\n1,2\n")
            result = subprocess.run(
                ["python3", str(SCRIPT), str(unknown)],
                capture_output=True, text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("no supported NCU CSV header", result.stderr)


if __name__ == "__main__":
    unittest.main()
