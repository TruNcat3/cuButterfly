import importlib.util
from pathlib import Path
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("phase_correlation", ROOT / "experiments/tools/run_phase_correlation_graph_20261013.py")
APP = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(APP)


class PhaseCorrelationTests(unittest.TestCase):
    def test_independent_batch_and_changing_translations(self):
        reference, inputs, expected = APP.make_inputs(torch, 32, 4, 2)
        results = APP.cpu_reference(torch, reference, inputs)
        self.assertFalse(torch.equal(reference[0], reference[1]))
        self.assertFalse(torch.equal(inputs[0], inputs[1]))
        for (shifts, correlation), target in zip(results, expected):
            self.assertEqual(shifts.dtype, torch.int64)
            self.assertTrue(torch.equal(shifts, target))
            self.assertTrue(torch.isfinite(correlation).all())

    def test_zero_spectrum_epsilon_and_half_period_convention(self):
        reference = torch.zeros(2, 8, 8, dtype=torch.float32)
        reference[1, 0, 0] = 1
        moving = reference.clone()
        moving[1] = torch.roll(reference[1], (4, -3), (-2, -1))
        shifts, correlation = APP.cpu_reference(torch, reference, [moving])[0]
        self.assertTrue(torch.equal(shifts, torch.tensor([[0, 0], [4, 3]])))
        self.assertTrue(torch.isfinite(correlation).all())
        self.assertEqual(APP.EPSILON, torch.finfo(torch.float32).eps)


if __name__ == "__main__":
    unittest.main()
