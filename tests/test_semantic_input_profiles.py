"""CPU-only tests for independently labelled FFT/FWHT input profiles."""
import copy
import importlib.util
import math
import pathlib
import unittest

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = load("profile_test_runner", "run_semantic_campaign.py")
profiles = load("input_profiles", "semantic_input_profiles.py")


def semantics(**changes):
    value = {"operator": 0, "storage": 4, "shape": [4], "batch": 2,
             "inverse": 0, "normalize_inverse": False, "fft_type": 0,
             "fft_output_scale": 1.0, "length_mode": 0,
             "ntt_input_layout": 0, "ntt_output_layout": 0}
    value.update(changes)
    return value


def quantize(value):
    return runner.decode_storage(runner.encode_storage(np.asarray(value, np.complex64), 4), 4)


def apply(s, source, profile=profiles.SCALED_PROFILE):
    return profiles.apply_input_profile(s, source, profile,
                                       encode_storage=runner.encode_storage,
                                       decode_storage=runner.decode_storage)


class InputProfileTests(unittest.TestCase):
    def test_fwht_source_raw_and_profile_identity(self):
        s = semantics(operator=2, storage=0)
        source = runner.make_source(s)
        self.assertEqual(source.dtype, np.dtype("float16"))
        raw, metadata = apply(s, source, profiles.RAW_PROFILE)
        np.testing.assert_array_equal(raw, source)
        self.assertEqual(metadata["input_sha256"], metadata["original_input_sha256"])
        self.assertEqual(profiles.scaled_profile(s), profiles.FWHT_SCALED_PROFILE)
        with self.assertRaisesRegex(ValueError, "match operator"):
            apply(s, source)

    def test_fwht_complete_l1_bounds_all_exact_butterfly_stages(self):
        s = semantics(operator=2, storage=0, shape=[2, 4])
        source = np.array([[[1, -2, 3, -4], [5, -6, 7, -8]],
                           [[60000, -60000, 60000, -60000]] * 2], dtype=np.float16)
        final, metadata = apply(s, source, profiles.FWHT_SCALED_PROFILE)
        self.assertEqual(final.dtype, np.dtype("float16"))
        np.testing.assert_array_equal(final, np.ldexp(source, metadata["exponent"]))
        self.assertGreater(math.ldexp(metadata["original_max_l1_bound"], metadata["exponent"] + 1),
                           metadata["bound_target"])
        self.assertLessEqual(metadata["quantized_max_l1_bound"], 65504 / 4)
        work = final.astype(np.float64).reshape(2, -1)
        for width in (1, 2, 4):
            groups = work.reshape(2, -1, 2 * width)
            left, right = groups[..., :width].copy(), groups[..., width:].copy()
            groups[..., :width], groups[..., width:] = left + right, left - right
            for batch, bound in zip(work, metadata["quantized_batch_l1_bounds"]):
                self.assertLessEqual(np.max(np.abs(batch)), bound)
        np.testing.assert_array_equal(work.reshape(final.shape), runner.reference(s, final))

    def test_fwht_requantization_and_directions_use_final_input(self):
        source = np.array([[2**-24, 60000, -60000, 60000]], dtype=np.float16)
        for inverse in (0, 1):
            for normalize in (False, True):
                for scale in (1.0,):
                    s = semantics(operator=2, storage=0, batch=1, inverse=inverse,
                                  normalize_inverse=normalize, fft_output_scale=scale)
                    final, metadata = apply(s, source, profiles.FWHT_SCALED_PROFILE)
                    self.assertEqual(final[0, 0], 0)
                    self.assertEqual(metadata["newly_zero_components"], 1)
                    self.assertEqual(metadata["newly_zero_elements"], 1)
                    expected = np.array([[sum(final[0].astype(np.float64)),
                                          final[0, 0].astype(np.float64) - final[0, 1] + final[0, 2] - final[0, 3],
                                          final[0, 0].astype(np.float64) + final[0, 1] - final[0, 2] - final[0, 3],
                                          final[0, 0].astype(np.float64) - final[0, 1] - final[0, 2] + final[0, 3]]])
                    # fft_output_scale belongs to FFT; the frozen FWHT oracle
                    # applies only its public inverse normalization.
                    expected /= 4 if inverse and normalize else 1
                    np.testing.assert_array_equal(runner.reference(s, final), expected)

    def test_fwht_rejects_frozen_scale_contract_gap(self):
        source = np.zeros((2, 4), np.float16)
        for scale in (-1.0, 0.0, 0.5):
            for profile in (profiles.RAW_PROFILE, profiles.FWHT_SCALED_PROFILE):
                with self.subTest(scale=scale, profile=profile), self.assertRaisesRegex(
                        ValueError, "FWHT input controls require fft_output_scale=1.*contract gap"):
                    apply(semantics(operator=2, storage=0, fft_output_scale=scale), source, profile)

    def test_fwht_rejects_other_operators_storage_and_dtype(self):
        for changes in ({"operator": 1}, {"operator": 3}, {"operator": 4}, {"operator": 5},
                        {"storage": 2}, {"fft_type": 1}, {"shape": [3]},
                        {"length_mode": 1}, {"fft_output_scale": -2}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                apply(semantics(**dict({"operator": 2, "storage": 0}, **changes)),
                      np.zeros((2, 4), np.float16), profiles.FWHT_SCALED_PROFILE)
        with self.assertRaises(ValueError):
            apply(semantics(operator=2, storage=0), np.zeros((2, 4), np.float32),
                  profiles.FWHT_SCALED_PROFILE)

    def test_raw_preserves_source_and_semantics(self):
        s = semantics()
        original_s = copy.deepcopy(s)
        source = runner.make_source(s)
        original = source.copy()
        final, report = apply(s, source, profiles.RAW_PROFILE)
        self.assertEqual(final.tobytes(), source.tobytes())
        self.assertFalse(np.shares_memory(final, source))
        self.assertEqual(source.tobytes(), original.tobytes())
        self.assertEqual(s, original_s)
        self.assertEqual(report["exponent"], 0)
        self.assertEqual(report["input_sha256"], report["original_input_sha256"])

    def test_scaled_never_enlarges_small_or_zero_source(self):
        for source in (runner.make_source(semantics()), np.zeros((2, 4), np.complex64)):
            final, report = apply(semantics(), source)
            self.assertEqual(report["factor"], 1.0)
            self.assertEqual(final.tobytes(), source.tobytes())
            self.assertTrue(np.all(np.abs(final.real) <= np.abs(source.real)))
            self.assertTrue(np.all(np.abs(final.imag) <= np.abs(source.imag)))

    def test_all_batches_and_multidimensional_bound_choose_maximal_exponent(self):
        s = semantics(shape=[2, 2])
        source = quantize([[[1, 2], [3, 4]], [[60000+60000j]*2]*2])
        before = source.copy()
        final, report = apply(s, source)
        exponent = report["exponent"]
        self.assertLess(exponent, 0)
        self.assertEqual(report["factor"], 2.0**exponent)
        self.assertGreater(report["original_batch_l1_bounds"][1],
                           report["original_batch_l1_bounds"][0])
        self.assertLessEqual(report["scaled_prequantization_max_l1_bound"], report["bound_target"])
        self.assertGreater(math.ldexp(report["original_max_l1_bound"], exponent + 1),
                           report["bound_target"])
        self.assertTrue(all(x <= report["bound_target"] for x in report["quantized_batch_l1_bounds"]))
        np.testing.assert_array_equal(source, before)
        self.assertEqual(report["headroom_fraction"], 0.25)
        self.assertEqual(report["bound_target"], 65504 / 4)
        np.testing.assert_array_equal(final, quantize(source * report["factor"]))

    def test_requantization_zero_counts_hash_and_reference(self):
        s = semantics()
        source = quantize([[2**-24, 60000, 60000, 60000], [0, 1j*2**-24, 0, 0]])
        final, report = apply(s, source)
        self.assertEqual(final[0, 0], 0)
        self.assertEqual(final[1, 1], 0)
        self.assertEqual(report["newly_zero_elements"], 2)
        self.assertEqual(report["newly_zero_components"], 2)
        self.assertEqual(report["quantized_zero_elements"], 5)
        self.assertNotEqual(report["input_sha256"], report["original_input_sha256"])
        np.testing.assert_array_equal(runner.reference(s, final), np.fft.fftn(final, axes=(1,)))
        self.assertFalse(np.array_equal(runner.reference(s, final),
                                        runner.reference(s, source) * report["factor"]))

    def test_exact_boundary_and_inverse_normalization(self):
        for normalize in (False, True):
            s = semantics(shape=[1], batch=1, inverse=1, normalize_inverse=normalize)
            final, report = apply(s, quantize([[16376]]))
            self.assertLessEqual(report["quantized_max_l1_bound"], report["bound_target"])
            self.assertLessEqual(report["factor"], 1)
            self.assertTrue(np.isfinite(final).all())

    def test_rejects_unsupported_or_invalid_semantics(self):
        for changes in ({"operator": 1}, {"fft_type": 1}, {"storage": 6},
                        {"length_mode": 1}, {"shape": [3]}, {"shape": []},
                        {"batch": 0}, {"fft_output_scale": 2},
                        {"fft_output_scale": float("nan")}, {"normalize_inverse": 1}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                apply(semantics(**changes), np.zeros((2, 4), np.complex64))
        with self.assertRaises(ValueError):
            apply(semantics(), np.zeros((2, 4), np.complex64), "unknown")

    def test_rejects_nonfinite_unquantized_incomplete_or_wrong_dtype(self):
        for source in (np.full((2, 4), np.nan, np.complex64),
                       np.full((2, 4), np.inf, np.complex64),
                       np.full((2, 4), 0.1, np.complex64),
                       np.zeros((1, 4), np.complex64),
                       np.zeros((2, 4), np.complex128)):
            with self.subTest(dtype=source.dtype, shape=source.shape), self.assertRaises(ValueError):
                apply(semantics(), source)


if __name__ == "__main__":
    unittest.main()
