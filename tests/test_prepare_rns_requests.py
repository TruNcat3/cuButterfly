"""CPU extraction and mock offline-preparation checks."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare_rns_requests.py"
SPEC = importlib.util.spec_from_file_location("prepare_rns_requests", SCRIPT)
requests = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(requests)

CONFIG = {
    "schema_version": 1, "log_n": 12, "inverse": False, "ring": "negacyclic",
    "input_order": "natural", "output_order": "bit_reversed",
    "domains": [{"modulus": 1152921504606830593, "root": 1152921504606830591}],
    "polynomials": [{"domain": 0, "input_offset": 16384, "output_offset": 32768}],
}
MAPPING = {
    "schema_version": 1, "n1_log": 6, "rows_per_block": 1, "threads_per_block": 128,
    "second_rows_per_block": 0, "second_threads_per_block": 0,
    "compute_unit": "radix4", "cross_twiddle": "first_pass",
    "core_family": "fused_ctgs", "arithmetic": "shoup",
    "physical_boundary_traversal": False,
}


def observation():
    return {"schema": requests.OBSERVATION_SCHEMA, "origin": "stage", "placement": "in_place",
            "config": deepcopy(CONFIG), "mapping": deepcopy(MAPPING), "cache_hit": False}


class FakePlan:
    calls = []
    closed = 0
    failure = False

    def __init__(self, library, config):
        self.config = config

    def __enter__(self):
        return self

    def __exit__(self, *error):
        type(self).closed += 1

    def prepare(self, **options):
        type(self).calls.append((deepcopy(self.config), options))
        if type(self).failure:
            raise RuntimeError("required cache entry missing")

    def info(self):
        return {"cache_hit": True, "cache_key": "actual-native-key", "mapping": MAPPING,
                "source": "cache"}


FAKE_RNS = SimpleNamespace(
    Library=lambda path: path, Plan=FakePlan,
    Placement=SimpleNamespace(IN_PLACE=1, SEPARATE=0),
    CachePolicy=SimpleNamespace(PREFER=0, REQUIRE=1, REFRESH=2),
)


class RequestPreparationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "observations.ndjson"
        self.output = self.root / "extracted"
        FakePlan.calls = []
        FakePlan.closed = 0
        FakePlan.failure = False

    def extract(self, rows):
        self.source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        return requests.extract([self.source], self.output)

    def test_complete_identity_and_observation_provenance(self):
        first = observation()
        duplicate = deepcopy(first)
        duplicate.update(origin="scalar", cache_hit=True)
        duplicate["mapping"]["rows_per_block"] = 2
        variants = []
        for mutate in (
            lambda row: row.update(placement="separate"),
            lambda row: row["config"]["domains"][0].update(root=7),
            lambda row: row["config"]["domains"][0].update(modulus=17),
            lambda row: row["config"]["polynomials"][0].update(input_offset=0),
            lambda row: row["config"]["polynomials"][0].update(output_offset=0),
            lambda row: row["config"].update(inverse=True),
        ):
            row = observation()
            mutate(row)
            variants.append(row)
        manifest = self.extract([first, duplicate, *variants])
        self.assertEqual(manifest["observation_count"], 8)
        self.assertEqual(manifest["unique_request_count"], 7)
        self.assertEqual(manifest["count_kind"], requests.COUNT_KIND)
        entry = next(row for row in manifest["requests"] if row["observation_count"] == 2)
        self.assertEqual(entry["origins"], {"scalar": 1, "stage": 1})
        self.assertEqual(entry["cache_hit_observation_count"], 1)
        self.assertEqual(len(entry["observed_mappings"]), 2)
        stored = requests.loads((self.output / entry["request_file"]).read_text())
        self.assertEqual(stored, CONFIG)
        self.assertEqual(entry["id"], requests.request_id(CONFIG, "in_place"))
        self.assertEqual(self.extract([duplicate, first, *variants])["requests"],
                         manifest["requests"])

    def test_strict_observation_validation_before_outputs(self):
        mutations = (
            lambda row: row.update(schema="wrong"),
            lambda row: row.update(origin="execution"),
            lambda row: row.update(placement="auto"),
            lambda row: row.update(cache_hit=1),
            lambda row: row.update(unrecognized=True),
            lambda row: row["config"].update(schema_version=True),
            lambda row: row["config"].update(inverse=0),
            lambda row: row["config"]["domains"][0].update(root=1.0),
            lambda row: row["config"]["domains"][0].update(modulus=1 << 64),
            lambda row: row["config"]["polynomials"][0].update(domain=1),
            lambda row: row["config"]["polynomials"][0].pop("input_offset"),
            lambda row: row["mapping"].update(physical_boundary_traversal=0),
            lambda row: row["mapping"].update(core_family="guess"),
        )
        for mutate in mutations:
            row = observation()
            mutate(row)
            with self.subTest(row=row), self.assertRaises(ValueError):
                self.extract([observation(), row])
        self.assertFalse(self.output.exists())
        self.source.write_text('{"schema":1,"schema":2}\n')
        with self.assertRaisesRegex(ValueError, "duplicate JSON field"):
            requests.extract([self.source], self.output)

    def test_exchange_preserves_legacy_observations_and_request_identity(self):
        legacy = observation()
        rows = [legacy]
        for exchange in ("block", "dependency_scoped"):
            row = observation()
            row["mapping"]["exchange"] = exchange
            rows.append(row)
        manifest = self.extract(rows)
        self.assertEqual(manifest["unique_request_count"], 1)
        entry = manifest["requests"][0]
        self.assertEqual(entry["id"], requests.request_id(CONFIG, "in_place"))
        self.assertEqual(len(entry["observed_mappings"]), 3)
        observed = [item["mapping"] for item in entry["observed_mappings"]]
        self.assertIn(legacy["mapping"], observed)
        self.assertNotIn("exchange", legacy["mapping"])
        for row in rows:
            self.assertIn(row["mapping"], observed)
        self.assertEqual(requests._load_requests(self.output / "manifest.json"),
                         [(entry["id"], "in_place", CONFIG)])

    def test_invalid_exchange_rejected_without_output(self):
        for exchange in ("unknown", "Block", "", None, 0, True, [], {}):
            row = observation()
            row["mapping"]["exchange"] = exchange
            with self.subTest(exchange=exchange), self.assertRaisesRegex(ValueError, "exchange"):
                self.extract([row])
        self.assertFalse(self.output.exists())

    def test_extract_cli_without_loading_cuda(self):
        self.source.write_text(json.dumps(observation()) + "\n")
        driver = """
import ctypes, runpy, sys
def forbidden(*args, **kwargs):
    raise AssertionError('extract loaded a shared library')
ctypes.CDLL = forbidden
script = sys.argv.pop(1)
runpy.run_path(script, run_name='__main__')
"""
        completed = subprocess.run([sys.executable, "-c", driver, str(SCRIPT), "extract",
                                    "--input", str(self.source), "--output", str(self.output)],
                                   capture_output=True, text=True, check=True)
        self.assertIn("1 requests from 1 prepare observations", completed.stdout)

    def test_prepare_defaults_to_all_and_preserves_requests(self):
        other = observation()
        other["placement"] = "separate"
        manifest = self.extract([observation(), other])
        result = requests.prepare(self.output / "manifest.json", "native.so", self.root / "cache",
                                  self.root / "results.json", policy="require", repetitions=9,
                                  rns_module=FAKE_RNS)
        self.assertEqual(len(result["results"]), 2)
        self.assertEqual(result["selected_request_ids"], [row["id"] for row in manifest["requests"]])
        self.assertEqual(FakePlan.closed, 2)
        for config, options in FakePlan.calls:
            self.assertEqual(config, CONFIG)
            self.assertEqual(options["policy"], 1)
            self.assertEqual(options["repetitions"], 9)
        self.assertEqual({options["placement"] for _, options in FakePlan.calls}, {0, 1})
        self.assertTrue(all(row["cache_key"] == "actual-native-key" for row in result["results"]))
        self.assertTrue(all(row["preparation_wall_seconds"] >= 0 for row in result["results"]))
        self.assertEqual(json.loads((self.root / "results.json").read_text()), result)

    def test_explicit_selection_and_failures(self):
        other = observation()
        other["config"]["inverse"] = True
        manifest = self.extract([observation(), other])
        chosen = manifest["requests"][1]["id"]
        result = requests.prepare(self.output / "manifest.json", "native.so", "cache",
                                  self.root / "results.json", request_ids=[chosen], limit=1,
                                  policy="refresh", rns_module=FAKE_RNS)
        self.assertEqual(result["selected_request_ids"], [chosen])
        self.assertEqual(FakePlan.calls[0][1]["policy"], 2)
        FakePlan.failure = True
        result = requests.prepare(self.output / "manifest.json", "native.so", "cache",
                                  self.root / "results.json", policy="require", rns_module=FAKE_RNS)
        self.assertEqual(len(result["results"]), 2)
        self.assertTrue(all(row["status"] == "error" for row in result["results"]))
        self.assertTrue(all("required cache entry missing" in row["error"] for row in result["results"]))
        with self.assertRaisesRegex(ValueError, "unknown request ids"):
            requests.prepare(self.output / "manifest.json", "native.so", "cache",
                             self.root / "results.json", request_ids=["absent"], rns_module=FAKE_RNS)

    def test_tampered_request_rejected_before_native_load(self):
        manifest = self.extract([observation()])
        path = self.output / manifest["requests"][0]["request_file"]
        changed = deepcopy(CONFIG)
        changed["domains"][0]["root"] = 7
        requests.write_json(path, changed)
        with self.assertRaisesRegex(ValueError, "does not match its id"):
            requests.prepare(self.output / "manifest.json", "native.so", "cache",
                             self.root / "results.json", rns_module=FAKE_RNS)
        self.assertEqual(FakePlan.calls, [])


if __name__ == "__main__":
    unittest.main()
