"""CPU-only manifest selection and evidence-wrapper checks."""
from copy import deepcopy
import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_rns_request_comparison", ROOT / "experiments/workloads/run_rns_request_comparison.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)

CONFIG = {
    "schema_version": 1, "log_n": 12, "inverse": False, "ring": "negacyclic",
    "input_order": "natural", "output_order": "bit_reversed",
    "domains": [{"modulus": 1152921504606830593, "root": 1152921504606830591}],
    "polynomials": [{"domain": 0, "input_offset": 0, "output_offset": 0}],
}
GPU = "GPU-00000000-0000-0000-0000-000000000003"
MAPPING = {
    "schema_version": 1, "n1_log": 6, "rows_per_block": 1, "threads_per_block": 128,
    "second_rows_per_block": 0, "second_threads_per_block": 0,
    "compute_unit": "radix4", "cross_twiddle": "first_pass", "core_family": "gpu_ntt_merge",
    "arithmetic": "shoup", "physical_boundary_traversal": False, "exchange": "block",
}


class ComparisonTests(unittest.TestCase):
    def fixture(self, directory, placements=("in_place",)):
        base = Path(directory)
        entries = []
        for index, placement in enumerate(placements):
            config = deepcopy(CONFIG)
            config["polynomials"] *= index + 1
            identity = runner.requests.request_id(config, placement)
            relative = f"requests/{identity}.json"
            runner.requests.write_json(base / relative, config)
            entries.append({"id": identity, "placement": placement, "request_file": relative})
        manifest = base / "manifest.json"
        runner.requests.write_json(manifest, {"schema": runner.requests.MANIFEST_SCHEMA,
            "count_kind": runner.requests.COUNT_KIND, "requests": entries})
        binary = base / "rns_request_compare"
        binary.write_bytes(b"fake binary; never executed")
        argv = ["--manifest", str(manifest), "--binary", str(binary),
                "--output-directory", str(base / "output"), "--cache-dir", str(base / "cache"),
                "--gpu-uuid", GPU]
        for entry in entries:
            argv += ["--request-id", entry["id"]]
        return runner.parser().parse_args(argv), entries

    def fake_run(self, calls, *, returncode=0, status="completed"):
        def execute(command, env, output, *unused):
            calls.append((command, env, output))
            output.with_suffix(".log").write_text("mock run\n")
            output.with_suffix(".metadata.json").write_text(json.dumps({
                "status": status, "returncode": returncode,
                "performance_eligible": returncode == 0,
                "artifacts": {"benchmark_binary": runner.evidence.file_identity(Path(command[0]))}}))
            return returncode
        return execute

    def test_explicit_id_required_and_defaults(self):
        with tempfile.TemporaryDirectory() as directory:
            args, _ = self.fixture(directory)
            self.assertEqual(args.cache_policy, "require")
            self.assertEqual(args.tune_repetitions, 5)
            self.assertEqual(args.repetitions, 20)
            self.assertEqual(args.rounds, 10)
            self.assertIsNone(args.exchange_override)
            self.assertEqual(args.native_dispatch, "modulus-ordered")
            self.assertEqual(args.cache_regime, "warm")
        with patch("sys.stderr"), self.assertRaises(SystemExit):
            runner.parser().parse_args(["--manifest", "m", "--binary", "b",
                "--output-directory", "o", "--cache-dir", "c"])

    def test_preserves_explicit_selection_order_and_exact_config(self):
        with tempfile.TemporaryDirectory() as directory:
            args, entries = self.fixture(directory, ("in_place", "in_place"))
            args.request_ids.reverse()
            calls = []
            with patch.object(runner.evidence, "run_with_evidence", side_effect=self.fake_run(calls)):
                self.assertEqual(runner.execute(args), 0)
            self.assertEqual(len(calls), 2)
            for call, identity in zip(calls, args.request_ids):
                command, env, output = call
                config_path = Path(command[command.index("--request") + 1])
                self.assertEqual(config_path.name, identity + ".json")
                self.assertEqual(command[command.index("--cache-policy") + 1], "require")
                self.assertEqual(command[command.index("--tune-repetitions") + 1], "5")
                self.assertEqual(command[command.index("--repetitions") + 1], "20")
                self.assertNotIn("--exchange-override", command)
                self.assertNotIn("--native-dispatch", command)
                self.assertNotIn("--cache-regime", command)
                self.assertEqual(env["CUDA_VISIBLE_DEVICES"], GPU)
                metadata = json.loads(output.with_suffix(".metadata.json").read_text())
                self.assertEqual(metadata["measurement_kind"], runner.MEASUREMENT_KIND)
                self.assertEqual(metadata["selection_kind"], "automatic_library")
                self.assertEqual(metadata["allowed_mapping_changes"], [])
                self.assertEqual(metadata["native_dispatch"], "modulus-ordered")
                self.assertIn("GPU_NTT_Modulus_Ordered_Inplace", metadata["native_dispatch_description"])
                self.assertEqual(metadata["request_id"], identity)
                self.assertEqual(metadata["status"], "completed")
                for name in ("manifest", "config", "benchmark_source"):
                    self.assertIsNotNone(metadata["request_provenance"][name]["sha256"])
                self.assertIsNotNone(metadata["artifacts"]["benchmark_binary"]["sha256"])

    def test_unknown_or_duplicate_ids_never_launch(self):
        for mode in ("unknown", "duplicate"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                args, _ = self.fixture(directory)
                args.request_ids = ["rns-unknown"] if mode == "unknown" else args.request_ids * 2
                with patch.object(runner.evidence, "run_with_evidence") as launch:
                    with self.assertRaisesRegex(ValueError, "unknown|unique"):
                        runner.execute(args)
                    launch.assert_not_called()

    def test_config_identity_mismatch_never_launches(self):
        with tempfile.TemporaryDirectory() as directory:
            args, entries = self.fixture(directory)
            path = args.manifest.parent / entries[0]["request_file"]
            config = json.loads(path.read_text())
            config["domains"][0]["root"] += 1
            runner.requests.write_json(path, config)
            with patch.object(runner.evidence, "run_with_evidence") as launch:
                with self.assertRaisesRegex(ValueError, "does not match"):
                    runner.execute(args)
                launch.assert_not_called()

    def test_nonstandard_manifest_request_path_is_explicitly_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            args, entries = self.fixture(directory)
            standard_path = args.manifest.parent / entries[0]["request_file"]
            exact_config = json.loads(standard_path.read_text())
            relative = "exported/nested/actual-config.json"
            actual_path = args.manifest.parent / relative
            runner.requests.write_json(actual_path, exact_config)
            shadow = deepcopy(exact_config)
            shadow["domains"][0]["root"] += 1
            runner.requests.write_json(standard_path, shadow)
            manifest = json.loads(args.manifest.read_text())
            manifest["requests"][0]["request_file"] = relative
            runner.requests.write_json(args.manifest, manifest)
            with patch.object(runner.evidence, "run_with_evidence") as launch:
                with self.assertRaisesRegex(ValueError, "request_file must match request id"):
                    runner.execute(args)
                launch.assert_not_called()

    def test_separate_placement_transmitted_and_validated(self):
        with tempfile.TemporaryDirectory() as directory:
            args, _ = self.fixture(directory, ("separate",))
            calls = []
            fake = self.fake_run(calls)

            def launch(command, env, output, *rest):
                with output.open("w", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=("backend", "placement", "mapping",
                        "exact", "input_preserved", "validation_trials", "repeated_input_semantics"))
                    writer.writeheader()
                    for backend in ("cubutterfly", "native-gpuntt"):
                        writer.writerow(dict(backend=backend, placement="separate", mapping=json.dumps(MAPPING),
                            exact=1, input_preserved=1, validation_trials=3,
                            repeated_input_semantics="same_preserved_input"))
                return fake(command, env, output, *rest)

            with patch.object(runner.evidence, "run_with_evidence", side_effect=launch):
                self.assertEqual(runner.execute(args), 0)
            command, _, output = calls[0]
            self.assertEqual(command[command.index("--placement") + 1], "separate")
            metadata = json.loads(output.with_suffix(".metadata.json").read_text())
            self.assertEqual(metadata["request_placement"], "separate")
            self.assertIn("GPU_NTT_Modulus_Ordered;", metadata["native_dispatch_description"])

    def test_separate_missing_evidence_cannot_remain_performance_eligible(self):
        with tempfile.TemporaryDirectory() as directory:
            args, _ = self.fixture(directory, ("separate",))
            calls = []
            with patch.object(runner.evidence, "run_with_evidence", side_effect=self.fake_run(calls)):
                with self.assertRaisesRegex(ValueError, "missing CSV"):
                    runner.execute(args)
            metadata = json.loads(calls[0][2].with_suffix(".metadata.json").read_text())
            self.assertEqual(metadata["status"], "failed_evidence_validation")
            self.assertFalse(metadata["performance_eligible"])

    def test_placement_validation_rejects_wrong_or_incomplete_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.csv"
            base = dict(backend="cubutterfly", placement="separate", exact="1", input_preserved="1",
                        validation_trials="3", repeated_input_semantics="same_preserved_input")
            for changes, message in (({"placement": "in-place"}, "placement"),
                                     ({"input_preserved": "0"}, "validation evidence"),
                                     ({"exact": "0"}, "validation evidence"),
                                     ({"validation_trials": "1"}, "validation evidence"),
                                     ({}, "both backends")):
                with self.subTest(changes=changes):
                    with output.open("w", newline="") as handle:
                        writer = csv.DictWriter(handle, fieldnames=base.keys())
                        writer.writeheader()
                        writer.writerow(dict(base, **changes))
                    with self.assertRaisesRegex(ValueError, message):
                        runner.validate_placement_evidence(output, "separate")

    def test_separate_native_api_descriptions(self):
        for inverse in (False, True):
            config = dict(CONFIG, inverse=inverse)
            description = runner.native_dispatch_description(config, "contiguous", "separate")
            self.assertTrue(description.startswith("GPU_INTT;" if inverse else "GPU_NTT;"))

    def test_failed_run_preserves_error_status_and_stops_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            args, _ = self.fixture(directory, ("in_place", "in_place"))
            calls = []
            with patch.object(runner.evidence, "run_with_evidence", side_effect=self.fake_run(
                    calls, returncode=1, status="blocked_gpu_busy")):
                self.assertEqual(runner.execute(args), 1)
            self.assertEqual(len(calls), 1)
            metadata = json.loads(calls[0][2].with_suffix(".metadata.json").read_text())
            self.assertEqual(metadata["status"], "blocked_gpu_busy")
            self.assertFalse(metadata["performance_eligible"])
            self.assertEqual(metadata["measurement_kind"], runner.MEASUREMENT_KIND)

    def test_existing_later_sidecar_blocks_entire_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            args, entries = self.fixture(directory, ("in_place", "in_place"))
            args.output_directory.mkdir()
            prior = args.output_directory / (entries[1]["id"] + ".metadata.json")
            prior.write_text("prior evidence")
            with patch.object(runner.evidence, "run_with_evidence") as launch:
                with self.assertRaises(FileExistsError):
                    runner.execute(args)
                launch.assert_not_called()
            self.assertEqual(prior.read_text(), "prior evidence")

    def test_invalid_counts_never_launch(self):
        for field, value in (("rounds", 0), ("repetitions", 1001), ("tune_repetitions", 0)):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                args, _ = self.fixture(directory)
                setattr(args, field, value)
                with patch.object(runner.evidence, "run_with_evidence") as launch:
                    with self.assertRaisesRegex(ValueError, "repetitions"):
                        runner.execute(args)
                    launch.assert_not_called()

    def test_shared_runner_collision_does_not_annotate_prior_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            args, entries = self.fixture(directory)

            def collision(command, env, output, *unused):
                output.with_suffix(".metadata.json").write_text("another process owns this evidence")
                raise FileExistsError("concurrent output")

            with patch.object(runner.evidence, "run_with_evidence", side_effect=collision):
                with self.assertRaises(FileExistsError):
                    runner.execute(args)
            metadata = args.output_directory / (entries[0]["id"] + ".metadata.json")
            self.assertEqual(metadata.read_text(), "another process owns this evidence")

    def write_control_csv(self, output, exchange, **changes):
        actual = dict(MAPPING, exchange=exchange, **changes)
        with output.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=("backend", "mapping", "prepared_mapping",
                                                        "selection_kind", "source"))
            writer.writeheader()
            writer.writerow({"backend": "cubutterfly", "mapping": json.dumps(actual),
                             "prepared_mapping": json.dumps(MAPPING),
                             "selection_kind": "component_control", "source": "component_control"})

    def test_exchange_override_is_separate_component_control_with_mapping_evidence(self):
        for exchange in ("block", "dependency_scoped"):
            with self.subTest(exchange=exchange), tempfile.TemporaryDirectory() as directory:
                args, entries = self.fixture(directory)
                args.exchange_override = exchange
                args.output_directory.mkdir()
                automatic = args.output_directory / (entries[0]["id"] + ".csv")
                automatic.write_text("prior automatic evidence")
                calls = []
                fake = self.fake_run(calls)

                def launch(command, env, output, *rest):
                    self.write_control_csv(output, exchange)
                    return fake(command, env, output, *rest)

                with patch.object(runner.evidence, "run_with_evidence", side_effect=launch):
                    self.assertEqual(runner.execute(args), 0)
                command, _, output = calls[0]
                self.assertEqual(command[command.index("--exchange-override") + 1], exchange)
                self.assertEqual(output.name, entries[0]["id"] + f".exchange-{exchange}.csv")
                self.assertEqual(automatic.read_text(), "prior automatic evidence")
                metadata = json.loads(output.with_suffix(".metadata.json").read_text())
                self.assertEqual(metadata["measurement_kind"], "component_control")
                self.assertEqual(metadata["selection_kind"], "component_control")
                self.assertEqual(metadata["allowed_mapping_changes"], ["exchange"])
                self.assertEqual(metadata["exchange_override"], exchange)
                self.assertEqual(metadata["actual_mapping"], dict(MAPPING, exchange=exchange))
                self.assertEqual(metadata["prepared_mapping"], MAPPING)
                with patch.object(runner.evidence, "run_with_evidence") as launch_again:
                    with self.assertRaises(FileExistsError):
                        runner.execute(args)
                    launch_again.assert_not_called()

    def test_exchange_override_invalid_values_never_launch(self):
        for value in ("auto", "Block", "", 0, True):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as directory:
                args, _ = self.fixture(directory)
                args.exchange_override = value
                with patch.object(runner.evidence, "run_with_evidence") as launch:
                    with self.assertRaisesRegex(ValueError, "exchange-override"):
                        runner.execute(args)
                    launch.assert_not_called()
        with patch("sys.stderr"), self.assertRaises(SystemExit):
            runner.parser().parse_args(["--exchange-override", "invalid"])

    def test_control_evidence_rejects_other_mapping_changes_or_wrong_exchange(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "control.csv"
            self.write_control_csv(output, "dependency_scoped", rows_per_block=2)
            with self.assertRaisesRegex(ValueError, "other than exchange"):
                runner.mapping_evidence(output, "dependency_scoped")
            self.write_control_csv(output, "block")
            with self.assertRaisesRegex(ValueError, "does not match"):
                runner.mapping_evidence(output, "dependency_scoped")
            text = output.read_text().replace("component_control", "automatic_library")
            output.write_text(text)
            with self.assertRaisesRegex(ValueError, "component_control"):
                runner.mapping_evidence(output, "block")
            self.write_control_csv(output, "block")
            with output.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            rows[0]["prepared_mapping"] = ""
            with output.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, "original prepared mapping"):
                runner.mapping_evidence(output, "block")

    def test_contiguous_dispatch_transmission_metadata_and_mapping_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            args, entries = self.fixture(directory)
            args.native_dispatch = "contiguous"
            calls = []
            fake = self.fake_run(calls)
            description = runner.native_dispatch_description(CONFIG, "contiguous")

            def launch(command, env, output, *rest):
                with output.open("w", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=("backend", "mapping", "prepared_mapping",
                                                                "native_dispatch", "native_dispatch_description"))
                    writer.writeheader()
                    for backend in ("cubutterfly", "native-gpuntt"):
                        writer.writerow({"backend": backend,
                            "mapping": json.dumps(MAPPING) if backend == "cubutterfly" else description + ";grid=1:1:1",
                            "prepared_mapping": json.dumps(MAPPING), "native_dispatch": "contiguous",
                            "native_dispatch_description": description})
                return fake(command, env, output, *rest)

            with patch.object(runner.evidence, "run_with_evidence", side_effect=launch):
                self.assertEqual(runner.execute(args), 0)
            command, env, output = calls[0]
            self.assertEqual(command[command.index("--native-dispatch") + 1], "contiguous")
            self.assertNotIn("--exchange-override", command)
            self.assertEqual(env["CUBUTTERFLY_RNS_CACHE_POLICY"], "require")
            metadata = json.loads(output.with_suffix(".metadata.json").read_text())
            self.assertEqual(metadata["measurement_kind"], "component_control")
            self.assertEqual(metadata["selection_kind"], "automatic_library")
            self.assertEqual(metadata["native_dispatch"], "contiguous")
            self.assertEqual(metadata["native_dispatch_description"], description)
            self.assertEqual(metadata["native_dispatch_scope"], "semantic_equivalent_not_original_application_dispatch")
            self.assertEqual(metadata["actual_mapping"], metadata["prepared_mapping"])
            self.assertEqual(metadata["allowed_mapping_changes"], [])
            output.write_text(output.read_text().replace("contiguous", "modulus-ordered"))
            with self.assertRaisesRegex(ValueError, "native dispatch"):
                runner.validate_native_evidence(output, "contiguous", description)
            output.write_text("backend,mapping,native_dispatch,native_dispatch_description\n")
            with self.assertRaisesRegex(ValueError, "no native dispatch evidence"):
                runner.validate_native_evidence(output, "contiguous", description)

    def test_contiguous_geometry_uses_physical_rows_and_both_directions(self):
        config = deepcopy(CONFIG)
        config["domains"] *= 2
        config["polynomials"] = [dict(domain=row % 2, input_offset=row * 4096, output_offset=row * 4096)
                                  for row in (3, 1, 0, 2)]
        self.assertEqual(runner.native_dispatch_description(config, "contiguous"),
                         "GPU_NTT_Inplace;PerPolynomial;X_N_plus;batch_size=4;mod_count=2")
        config["inverse"] = True
        self.assertTrue(runner.native_dispatch_description(config, "contiguous").startswith("GPU_INTT_Inplace;"))
        for mutate in (
            lambda c: c["polynomials"][0].update(domain=0),
            lambda c: c["polynomials"][0].update(input_offset=0, output_offset=0),
            lambda c: c["polynomials"][0].update(input_offset=16384, output_offset=16384),
            lambda c: c["polynomials"][0].update(input_offset=1, output_offset=1),
            lambda c: c["polynomials"][0].update(output_offset=0),
            lambda c: c.update(domains=c["domains"] * 3),
            lambda c: c.update(log_n=21),
        ):
            invalid = deepcopy(config)
            mutate(invalid)
            with self.subTest(config=invalid), self.assertRaisesRegex(ValueError, "contiguous native dispatch"):
                runner.native_dispatch_description(invalid, "contiguous")

    def test_invalid_native_dispatch_rejected_before_launch(self):
        for value in ("auto", "Contiguous", "", None, 0):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as directory:
                args, _ = self.fixture(directory)
                args.native_dispatch = value
                with patch.object(runner.evidence, "run_with_evidence") as launch:
                    with self.assertRaisesRegex(ValueError, "native-dispatch"):
                        runner.execute(args)
                    launch.assert_not_called()
        with patch("sys.stderr"), self.assertRaises(SystemExit):
            runner.parser().parse_args(["--native-dispatch", "invalid"])

    def write_cache_csv(self, output, **changes):
        rows = []
        for backend in ("cubutterfly", "native-gpuntt"):
            rows.append(dict(backend=backend, mapping=json.dumps(MAPPING),
                prepared_mapping=json.dumps(MAPPING), source="component_control",
                selection_kind="component_control", measurement_kind="component_control",
                cache_regime="l2-displaced", l2_cache_bytes="1048576",
                cache_displacement_bytes="2097152", cache_scope=runner.DISPLACED_CACHE_SCOPE,
                repetitions="1"))
            rows[-1].update(changes)
        with output.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    def test_cache_regime_validation_never_launches(self):
        for regime, repetitions in (("unknown", 1), ("", 1), (None, 1),
                                     ("l2-displaced", 20), ("l2-displaced", 2)):
            with self.subTest(regime=regime, repetitions=repetitions), tempfile.TemporaryDirectory() as directory:
                args, _ = self.fixture(directory)
                args.cache_regime, args.repetitions = regime, repetitions
                with patch.object(runner.evidence, "run_with_evidence") as launch:
                    with self.assertRaisesRegex(ValueError, "cache-regime|repetitions 1"):
                        runner.execute(args)
                    launch.assert_not_called()
        with patch("sys.stderr"), self.assertRaises(SystemExit):
            runner.parser().parse_args(["--cache-regime", "unknown"])

    def test_l2_displacement_isolated_output_and_component_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            args, entries = self.fixture(directory)
            args.cache_regime, args.repetitions = "l2-displaced", 1
            calls = []
            fake = self.fake_run(calls)

            def launch(command, env, output, *rest):
                self.write_cache_csv(output)
                return fake(command, env, output, *rest)

            with patch.object(runner.evidence, "run_with_evidence", side_effect=launch):
                self.assertEqual(runner.execute(args), 0)
            command, _, output = calls[0]
            self.assertEqual(command[command.index("--cache-regime") + 1], "l2-displaced")
            self.assertEqual(command[command.index("--repetitions") + 1], "1")
            self.assertEqual(output.name, entries[0]["id"] + ".cache-l2-displaced.csv")
            metadata = json.loads(output.with_suffix(".metadata.json").read_text())
            self.assertEqual(metadata["cache_displacement_bytes"], 2097152)
            self.assertEqual(metadata["device_l2_cache_bytes"], 1048576)
            self.assertEqual(metadata["cache_scope"], runner.DISPLACED_CACHE_SCOPE)
            self.assertEqual(metadata["measurement_kind"], "component_control")
            self.assertEqual(metadata["selection_kind"], "component_control")
            self.assertEqual(metadata["actual_mapping"], metadata["prepared_mapping"])
            self.assertEqual(metadata["allowed_mapping_changes"], [])
            self.assertFalse(metadata["headline_eligible"])

    def test_l2_displacement_evidence_rejects_false_or_missing_context(self):
        invalid = ({"cache_regime": "warm"}, {"l2_cache_bytes": "0"},
                   {"l2_cache_bytes": str(1 << 30)}, {"cache_displacement_bytes": "1048576"},
                   {"cache_displacement_bytes": str(2 << 30)}, {"l2_cache_bytes": ""},
                   {"cache_scope": "application_cache_replay"}, {"repetitions": "20"},
                   {"selection_kind": "automatic_library"}, {"prepared_mapping": ""},
                   {"measurement_kind": runner.MEASUREMENT_KIND}, {"source": "cache"})
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "cache.csv"
            for changes in invalid:
                with self.subTest(changes=changes):
                    self.write_cache_csv(output, **changes)
                    with self.assertRaises(ValueError):
                        runner.cache_evidence(output, "l2-displaced")
            self.write_cache_csv(output, backend="cubutterfly")
            with self.assertRaisesRegex(ValueError, "both backends"):
                runner.cache_evidence(output, "l2-displaced")
            output.write_text("backend,mapping\n")
            with self.assertRaisesRegex(ValueError, "both backends"):
                runner.cache_evidence(output, "l2-displaced")

    def test_warm_single_transform_keeps_legacy_binary_compatibility(self):
        with tempfile.TemporaryDirectory() as directory:
            args, _ = self.fixture(directory)
            args.repetitions = 1
            calls = []
            fake = self.fake_run(calls)

            def launch(command, env, output, *rest):
                output.write_text("backend,mapping\n")
                return fake(command, env, output, *rest)

            with patch.object(runner.evidence, "run_with_evidence", side_effect=launch):
                self.assertEqual(runner.execute(args), 0)
            command, _, output = calls[0]
            self.assertNotIn("--cache-regime", command)
            self.assertEqual(command[command.index("--repetitions") + 1], "1")
            metadata = json.loads(output.with_suffix(".metadata.json").read_text())
            self.assertEqual(metadata["cache_regime"], "warm")
            self.assertEqual(metadata["cache_displacement_bytes"], 0)
            self.assertEqual(metadata["measurement_kind"], runner.MEASUREMENT_KIND)

    def test_l2_displacement_does_not_accept_changed_prepared_mapping(self):
        with tempfile.TemporaryDirectory() as directory:
            args, _ = self.fixture(directory)
            args.cache_regime, args.repetitions = "l2-displaced", 1
            fake = self.fake_run([])

            def launch(command, env, output, *rest):
                self.write_cache_csv(output, mapping=json.dumps(dict(MAPPING, rows_per_block=2)))
                return fake(command, env, output, *rest)

            with patch.object(runner.evidence, "run_with_evidence", side_effect=launch):
                with self.assertRaisesRegex(ValueError, "changed the prepared"):
                    runner.execute(args)

    def test_successful_displacement_requires_csv_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            args, _ = self.fixture(directory)
            args.cache_regime, args.repetitions = "l2-displaced", 1
            with patch.object(runner.evidence, "run_with_evidence", side_effect=self.fake_run([])):
                with self.assertRaisesRegex(ValueError, "missing CSV evidence"):
                    runner.execute(args)


if __name__ == "__main__":
    unittest.main()
