"""A complete bootstrap must never be reported with the short-chain contract."""
import copy
import importlib.util
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "experiments/workloads" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


runner = load("run_heongpu_ckks_e2e")
summary = load("summarize_heongpu_ckks_e2e")


class BootstrapContractTests(unittest.TestCase):
    def fixture(self):
        profile = runner.bootstrap_profile()
        meta = {"benchmark_argv": ["bench", "--parameter-set", "bootstrap-v2",
                "--sizes", "65536", "--batches", "1", "--aux-primes", "5", "--graph"]}
        row = dict(scene="regular_bootstrapping_v2", N="65536", batch="1",
                   baseline=summary.LOOP[0], batch_submission=summary.LOOP[1],
                   mode="cuda_graph", parameter_set="bootstrap-v2", keyswitch_method="II",
                   native_ms="2", cubutterfly_ms="1", speedup="2", exact_ciphertext="1",
                   mismatch_count="0", Q_bits=profile["Q_bits"], P_bits=profile["P_bits"],
                   scale="1099511627776", sec_level="none")
        return row, meta

    def test_real_modulus_bits_and_header(self):
        profile = runner.bootstrap_profile()
        self.assertEqual((len(profile["Q_values"]), len(profile["P_values"])), (25, 5))
        # The upstream nominal "60" first Q is slightly ABOVE 2**60.
        self.assertEqual(profile["Q_values"][0], 0x10000000006e0001)
        self.assertEqual(profile["Q_bits"].split(":")[0], "61")
        self.assertEqual(profile["P_bits"], ":".join(["61"] * 5))
        header = runner.bootstrap_header()
        self.assertIn(str(profile["Q_values"][0]) + "ULL", header)
        self.assertIn('q_bits = "' + profile["Q_bits"] + '"', header)

    def test_reduced_profiles_remain_explicit_and_distinct(self):
        v2 = runner.bootstrap_profile("bootstrap_v2_small")
        self.assertEqual(v2["N"], 16384)
        self.assertEqual((len(v2["Q_values"]), len(v2["P_values"])), (25, 5))
        self.assertIn('N = 16384', runner.bootstrap_header("bootstrap_v2_small"))
        v1 = runner.bootstrap_profile("bootstrap_v1_small")
        self.assertEqual(v1["N"], 4096)
        self.assertEqual((len(v1["Q_values"]), len(v1["P_values"])), (0, 0))
        self.assertEqual(v1["Q_bits"].split(":")[0], "60")
        header = runner.bootstrap_header("bootstrap_v1_small")
        self.assertIn("bootstrap_ctos_piece = 3", header)
        self.assertIn('p_bits = "60:60:60"', header)

    def test_complete_bootstrap_parses(self):
        row, meta = self.fixture()
        self.assertEqual(summary.parse_rows([row], meta, "fixture")[0]["parameter_set"], "bootstrap-v2")

    def test_three_v2_sizes_keep_full_modulus_chain(self):
        primary = runner.bootstrap_profile()
        for name, size in (("bootstrap_v2_small", 16384), ("bootstrap_v2_mid", 32768), ("primary", 65536)):
            with self.subTest(profile=name):
                profile = runner.bootstrap_profile(name)
                self.assertEqual(profile["N"], size)
                self.assertIn(f'N = {size}', runner.bootstrap_header(name))
                for key in ("Q_values", "P_values"):
                    self.assertEqual(profile[key], primary[key])
                    self.assertTrue(all((q-1) % (2*size) == 0 for q in profile[key]))
        self.assertEqual(runner.bootstrap_profile()["N"], 65536)

    def test_size_profiles_report_separately(self):
        for name, size in (("bootstrap-v2-small", "16384"), ("bootstrap-v2-mid", "32768")):
            row, meta = self.fixture()
            meta["benchmark_argv"][meta["benchmark_argv"].index("bootstrap-v2")] = name
            meta["benchmark_argv"][meta["benchmark_argv"].index("65536")] = size
            actual = summary.parse_rows([dict(row, N=size, parameter_set=name)], meta, "fixture")[0]
            self.assertEqual(actual["parameter_set"], name)
            with self.assertRaises(ValueError):
                summary.parse_rows([row], meta, "fixture")

    def test_reject_short_chain_label_and_primes(self):
        row, meta = self.fixture()
        for key, value in (("parameter_set", "short-chain"), ("Q_bits", "50:40:40"), ("P_bits", "50:50")):
            invalid = dict(row, **{key: value})
            with self.subTest(key=key), self.assertRaises(ValueError):
                summary.parse_rows([invalid], meta, "fixture")

    def test_reject_bootstrap_shape_substitution(self):
        row, meta = self.fixture()
        invalid = copy.deepcopy(meta)
        invalid["benchmark_argv"][invalid["benchmark_argv"].index("65536")] = "4096"
        with self.assertRaises(ValueError):
            summary.parse_rows([dict(row, N="4096")], invalid, "fixture")

    def test_reject_short_operation_as_bootstrap(self):
        row, meta = self.fixture()
        invalid = copy.deepcopy(meta)
        invalid["benchmark_argv"] += ["--scenes", "keyswitch"]
        with self.assertRaises(ValueError):
            summary.parse_rows([dict(row, scene="keyswitch")], invalid, "fixture")


if __name__ == "__main__":
    unittest.main()
