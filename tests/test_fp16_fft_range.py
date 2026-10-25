"""CPU tests for the native FP16 range diagnostic."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np

SPEC = importlib.util.spec_from_file_location(
    "fp16_range", Path(__file__).resolve().parents[1] / "scripts/diagnose_fp16_fft_range.py")
diagnostic = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diagnostic)


class NativeDitRangeTests(unittest.TestCase):
    def test_impulse_natural_order_and_source_preserved(self):
        source = np.zeros(16, dtype=np.complex64)
        source[1] = 1 + 1j
        before = source.copy()
        real, imag, stages = diagnostic.simulate_native_dit(source)
        actual = np.empty(16, dtype=np.complex64)
        actual.real, actual.imag = real, imag
        np.testing.assert_allclose(actual, np.fft.fft(source), atol=.002, rtol=.002)
        np.testing.assert_array_equal(source, before)
        self.assertEqual([s["nonfinite_complex"] for s in stages], [0] * 4)

    def test_known_butterfly_overflow_then_nan_propagation(self):
        real, imag, stages = diagnostic.simulate_native_dit(
            np.full(4, 40000, dtype=np.complex64))
        self.assertEqual(stages[0]["nonfinite_real_products"], [0, 0, 0, 0])
        self.assertEqual(stages[0]["nonfinite_complex"], 2)
        self.assertEqual(stages[1]["nonfinite_complex"], 2)
        self.assertGreater(np.isnan(imag).sum(), 0)  # Inf * zero twiddle.
        self.assertTrue(np.isinf(real[0]))

    def test_length_guard(self):
        for length in (0, 1, 3):
            with self.assertRaises(ValueError):
                diagnostic.simulate_native_dit(np.zeros(length, dtype=np.complex64))


if __name__ == "__main__":
    unittest.main()
