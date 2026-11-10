import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("periodic_poisson", ROOT / "experiments/tools/run_periodic_poisson_graph_20261013.py")
APP = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(APP)


class PeriodicPoissonTests(unittest.TestCase):
    def test_event_primary_summary_keeps_wall_diagnostic(self):
        timings = {backend + "_" + mode: {
            "cuda_event_ms": [device], "cuda_event_median_ms": device,
            "wall_ms": [wall], "wall_median_ms": wall}
            for mode in ("eager", "graph")
            for backend, device, wall in (("cubutterfly", 2, 5), ("torch", 3, 6), ("cupy", 4, 7))}
        result = APP.summarize_complete_call_timings(timings, 16)
        self.assertEqual(result["speedup"]["torch_graph"], 1.5)
        self.assertEqual(result["speedup_cuda_event"]["cupy_graph"], 2)
        self.assertEqual(result["speedup_wall"]["torch_graph"], 1.2)
        own = timings["cubutterfly_graph"]
        self.assertEqual(own["batch_throughput_per_second"], 8000)
        self.assertEqual(own["batch_wall_throughput_per_second"], 3200)
        self.assertEqual(own["per_rhs_cuda_event_ms"], 0.125)
        self.assertEqual(own["per_rhs_wall_ms"], 0.3125)
        self.assertEqual(own["cuda_event_ms"], [2])
        self.assertEqual(own["wall_ms"], [5])

    def test_event_brackets_complete_call_on_declared_stream(self):
        operations, clock = [], [0]
        class Stream:
            def synchronize(self):
                operations.append("stream_sync")
        stream = Stream()
        class Event:
            def record(self, recorded_stream):
                self.asserted_stream = recorded_stream
                self.value = clock[0]
                operations.append("event")
                assert recorded_stream is stream
            def synchronize(self):
                operations.append("event_sync")
        def complete_call(value):
            for operation, duration in (("input_refresh", 2), ("solve_replay", 5), ("output_clone", 3)):
                operations.append(operation)
                clock[0] += duration
            return object()
        with patch.object(APP.measure.__globals__["time"], "perf_counter", side_effect=[0, .02, .03, .05]):
            timing = APP.measure(complete_call, ["first", "changed"], stream, Event, 1, 2,
                                 lambda begin, end: end.value - begin.value)
        self.assertEqual(timing["cuda_event_ms"], [10, 10])
        self.assertAlmostEqual(timing["wall_median_ms"], 20)
        self.assertEqual(operations[4:], ["event", "input_refresh", "solve_replay", "output_clone", "event", "event_sync"] * 2)

    def test_single_fourier_mode_and_zero_mode(self):
        size = 32
        k2, multiplier = APP.spectral_coefficients(torch, size)
        self.assertEqual(multiplier[0, 0].item(), 0)
        spectrum = torch.zeros(2, size, size // 2 + 1, dtype=torch.complex64)
        spectrum[:, 2, 3] = 7 + 2j
        forward, inverse = APP.cpu_operators(torch, size)
        expected = inverse(spectrum)
        rhs = inverse(spectrum * k2) + 4
        for norm in ("backward", "forward"):
            forward, inverse = APP.cpu_operators(torch, size, norm)
            solution = APP.solve(rhs, forward, inverse, multiplier)
            torch.testing.assert_close(solution, expected, rtol=2e-5, atol=1e-8)
            self.assertEqual(solution.dtype, torch.float32)

    def test_manufactured_and_independent_inputs(self):
        inputs, analytical = APP.make_inputs(torch, 32, 4)
        _, multiplier = APP.spectral_coefficients(torch, 32)
        forward, inverse = APP.cpu_operators(torch, 32)
        solution = APP.solve(inputs[0], forward, inverse, multiplier)
        diagnostic = APP.errors(torch, solution, analytical)
        self.assertTrue(diagnostic["finite"])
        self.assertGreater(diagnostic["max_abs"], 0)
        self.assertFalse(torch.equal(inputs[1], inputs[2]))
        self.assertFalse(torch.equal(inputs[1][0], inputs[1][1]))
        self.assertTrue(torch.isfinite(solution).all())


if __name__ == "__main__":
    unittest.main()
