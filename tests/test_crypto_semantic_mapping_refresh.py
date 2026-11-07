"""Synthetic receipt tests: these fixtures are not GPU measurement evidence."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import crypto_semantic_mapping_refresh as adapter
import prepare_crypto_composition_refresh as refresh
import run_async_semantic_campaign as asynchronous
import run_crypto_composition as composition
import run_semantic_campaign as semantic


class SemanticMappingAdapterTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.directory = self.root / "cyclic_mapping"
        self.library = self.root / "libcuntt.so"
        self.library.write_bytes(b"synthetic-library-for-CPU-contract-test")
        _, self.compositions = composition._workloads_document(ROOT / "config/research_campaign/composed.json")
        self.calibration = refresh.calibration_workloads(self.compositions, {})
        composition.write_json(self.root / "calibration_workloads.json", self.calibration)
        self.limits = adapter.budgets(self.calibration)
        composition.write_json(self.directory / "cpu_budget.json", self.limits)
        self.cases = []
        for contract in self.calibration["contracts"]:
            row, bits = contract["semantics"], contract["word_bits"]
            s = dict(operator=1, storage=8 if bits == 32 else 9, compute=3 if bits == 32 else 4,
                shape=[1 << row["logN"]], batch=row["batch"], inverse=row["direction"] == "inverse",
                normalize_inverse=row["normalization"] == "inverse", placement=0,
                input_orders=[0], output_orders=[0], input_strides=[1], output_strides=[1],
                input_batch_stride=1 << row["logN"], output_batch_stride=1 << row["logN"],
                modulus=row["modulus"], word_bits=bits, length_mode=0)
            self.cases.append(dict(id=semantic.digest(s), populations=[contract["id"]],
                                   descriptor=dict(semantics=s, selection=dict(algorithm_policy=0))))
        manifest = dict(schema=adapter.manifest_api.MANIFEST_SCHEMA, cases=self.cases,
            fingerprints=dict(library=dict(path=str(self.library), sha256=composition.sha256(self.library)),
                              sources=[], config={}), invalid_requests=[])
        manifest["manifest_sha256"] = adapter.manifest_api.manifest_hash(manifest)
        composition.write_json(self.directory / "manifest.json", manifest)
        protocol = {**semantic.PROTOCOL, "reference_max_elements": self.limits["reference_max_elements"],
                    "ntt_max_operations": self.limits["ntt_max_operations"]}
        identity = dict(manifest_sha256=semantic.digest(manifest), library_sha256=composition.sha256(self.library),
            gpu_uuid="GPU-test", compile_mode="research", timing=True, protocol=protocol,
            sources=asynchronous.source_identity(), runner_revision=asynchronous.REVISION,
            runtime_environment={"CUBUTTERFLY_SEARCH_BUDGET": "512"})
        self.report = dict(status="complete", identity=identity, remaining_case_ids=[], invalid_request_count=0, rows=[])
        self.rows = []
        self.report_path = self.directory / "measurement/results.json"
        for case in self.cases:
            s = case["descriptor"]["semantics"]
            plan_descriptor = copy.deepcopy(case["descriptor"])
            plan_descriptor["selection"]["algorithm_policy"] = 1
            verdict = dict(exact=True, mismatches=0, elements=s["batch"] * s["shape"][0])
            row = dict(id=case["id"], status="pass", descriptor=case["descriptor"], plan_descriptor=plan_descriptor,
                       mapping=dict(kind="ntt", schema_version=1, contract=s, backend="shared-iterative"),
                       correctness=verdict, mapping_replay_correctness=verdict,
                       selection_scope="public MEASURE", selection_reason="synthetic CPU fixture",
                       timing=dict(protocol=adapter.GRAPH_PROTOCOL, samples_ms=[1] * 7, median_ms=1))
            self.rows.append(row)
            self.report["rows"].append(semantic.write_case_checkpoint(self.report_path.parent, row))
        semantic.write_report(self.report_path, self.report)

    def adapt(self):
        return adapter.adapt(self.root, self.report_path, self.library, "GPU-test")

    def update_row(self, index, row):
        self.report["rows"][index] = semantic.write_case_checkpoint(self.report_path.parent, row)
        semantic.write_report(self.report_path, self.report)

    def test_all_90_convert_without_synthetic_performance(self):
        exported = self.adapt()
        self.assertEqual(len(exported["cells"]), 90)
        self.assertEqual(exported["source"]["original_measurement_identity"], self.report["identity"])
        self.assertEqual(exported["source"]["search_protocol"]["candidate_warmup"], 3)
        self.assertNotIn("kernel_ms", exported["cells"][0]["selected"]["samples"][0])
        self.assertEqual(exported["cells"][0]["provenance"]["source_timing"], self.rows[0]["timing"])
        adapter.validate_source(exported, self.library, "GPU-test")
        portfolios = composition.mapping_portfolios(self.compositions[0], "GPU-test", {}, self.root,
                                                    [self.directory / "mapping_source.json"])
        self.assertEqual(len(portfolios), 2)
        self.assertTrue(all(mapping is not None for mapping in portfolios[1]["mappings"]))

    def test_missing_point_and_wrong_gpu_fail_closed(self):
        self.report["rows"].pop()
        semantic.write_report(self.report_path, self.report)
        with self.assertRaisesRegex(ValueError, "90 unique"):
            self.adapt()
        self.report["identity"]["gpu_uuid"] = "GPU-other"
        semantic.write_report(self.report_path, self.report)
        with self.assertRaisesRegex(ValueError, "identity"):
            self.adapt()

    def test_comparator_rebinding_preserves_mapping_provenance_only(self):
        name = "semantic_gpuntt_baseline.py"
        self.report["identity"]["sources"][name] = "original-comparator-source"
        semantic.write_report(self.report_path, self.report)
        exported = self.adapt()
        self.assertEqual(exported["source"]["original_measurement_identity"], self.report["identity"])
        self.assertEqual(exported["source"]["comparator_only_source_updates"][name],
                         dict(original="original-comparator-source", current=asynchronous.source_identity()[name]))
        self.assertEqual(exported["cells"][0]["selected"]["samples"][0]["mapping_json"], self.rows[0]["mapping"])
        adapter.validate_source(exported, self.library, "GPU-test")
        self.report["identity"]["sources"]["run_semantic_campaign.py"] = "changed-execution"
        semantic.write_report(self.report_path, self.report)
        with self.assertRaisesRegex(ValueError, "execution source"):
            self.adapt()

    def test_recovered_report_path_is_retained_and_verified(self):
        recovered = self.root / "recovery/measurement"
        recovered.mkdir(parents=True)
        for index, row in enumerate(self.rows):
            self.report["rows"][index] = semantic.write_case_checkpoint(recovered, row)
        recovered_report = recovered / "results.json"
        semantic.write_report(recovered_report, self.report)
        exported = adapter.adapt(self.root, recovered_report, self.library, "GPU-test")
        self.assertEqual(exported["source"]["report_path"], str(recovered_report))
        adapter.validate_source(exported, self.library, "GPU-test")
        recovered_report.write_text(recovered_report.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "source changed"):
            adapter.validate_source(exported, self.library, "GPU-test")

    def test_contract_correctness_mapping_and_graph_failures(self):
        changes = [lambda row: row["correctness"].update(exact=False),
                   lambda row: row["mapping"].update(kind="composition"),
                   lambda row: row["mapping"]["contract"].update(word_bits=16),
                   lambda row: row["plan_descriptor"]["semantics"].update(normalize_inverse="wrong"),
                   lambda row: row["descriptor"]["semantics"].update(output_orders=[1]),
                   lambda row: row["timing"].update(protocol="event"),
                   lambda row: row["timing"].update(samples_ms=[1] * 6)]
        for change in changes:
            with self.subTest(change=change):
                row = copy.deepcopy(self.rows[0])
                change(row)
                self.update_row(0, row)
                with self.assertRaises(ValueError):
                    self.adapt()
        self.update_row(0, self.rows[0])
        self.adapt()

    def test_tampered_export_is_not_accepted_as_mapping_source(self):
        exported = self.adapt()
        exported["cells"][0]["selected"]["samples"][0]["mapping_json"]["backend"] = "tampered"
        with self.assertRaisesRegex(ValueError, "differs from.*source"):
            adapter.validate_source(exported, self.library, "GPU-test")

    def test_mapping_context_validates_once_and_rejects_source_changes(self):
        self.adapt()
        path = self.directory / "mapping_source.json"
        with patch.object(adapter, "validate_source", wraps=adapter.validate_source) as validate:
            context = composition.prepare_mapping_context(self.root, [path], "GPU-test")
            for row in self.compositions[:2]:
                composition.mapping_portfolios(row, "GPU-test", {}, self.root, [path], context=context)
            self.assertEqual(validate.call_count, 1)
            composition.validate_mapping_context(context, full_digest=True)
            path.write_text(path.read_text() + "\n")
            with self.assertRaisesRegex(ValueError, "changed"):
                composition.mapping_portfolios(self.compositions[0], "GPU-test", {}, self.root, [path], context=context)

    def test_cpu_readiness_is_a_launch_gate(self):
        adapter.preparation.initialize(self.directory / "prepared", composition.read_json(self.directory / "manifest.json"),
            self.limits["reference_max_elements"], self.limits["ntt_max_operations"])
        with self.assertRaisesRegex(ValueError, "90 not ready"):
            adapter.require_ready(self.root)

    def test_real_historical_ntt_row_shapes_match_adapter(self):
        report = ROOT / "results/install_compare_repair_20261014/measurement/results.json"
        if not report.is_file():
            self.skipTest("historical schema audit receipt is not present in this checkout")
        receipt = adapter.audit_historical_shape(report, self.root)
        self.assertGreater(len(receipt["audited_rows"]), 0)
        self.assertEqual({row["word_bits"] for row in receipt["audited_rows"]}, {32, 64})
        self.assertEqual(receipt["historical_timing_rows_adopted"], 0)


if __name__ == "__main__":
    unittest.main()
