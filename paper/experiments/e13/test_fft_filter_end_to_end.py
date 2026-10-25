"""Bounded CPU semantics tests, independent direct-convolution oracle."""
import unittest
import numpy as np
from scipy.signal import convolve2d
from run_fft_filter_end_to_end import cpu_convolve, error_distribution, padding_shape, same_crop, rotate_methods, require_valid_reports
from run_fft_filter_end_to_end import validate_graph_replay
from run_fft_filter_end_to_end import runtime_contract, method_call_contract


class RuntimeContract(unittest.TestCase):
    def test_cached_graph_overrides_preparation_without_mutation(self):
        prepared = {'batch': 16, 'cubutterfly_padding': [4096, 4096],
                    'filter_contract': 'spectrum recomputed', 'cpu_fp32_error': {'finite': True}}
        actual = runtime_contract(prepared, (2100, 2100), cached_filter=True,
                                  execution_mode='graph')
        self.assertEqual(actual['cubutterfly_padding'], [2100, 2100])
        self.assertEqual(actual['batch'], 16)
        self.assertIn('cached outside timed calls', actual['filter_contract'])
        self.assertNotIn('cpu_fp32_error', actual)
        self.assertEqual(prepared['filter_contract'], 'spectrum recomputed')
        self.assertEqual(prepared['cubutterfly_padding'], [4096, 4096])
        boundary = method_call_contract((2100, 2100), cached_filter=True, execution_mode='graph')
        self.assertIn('image_device_input_refresh', boundary['timing_scope'])
        self.assertIn('kernel_device_input_refresh', boundary['timing_scope'])
        self.assertIn('independent_output_copy', boundary['timing_scope'])
        self.assertNotIn('filter_fft', boundary['timing_scope'])
        self.assertIn('filter_spectrum_setup', boundary['excluded'])

    def test_uncached_eager_and_implicit_single_batch(self):
        actual = runtime_contract({}, (16, 16), cached_filter=False, execution_mode='eager')
        self.assertEqual(actual['batch'], 1)
        self.assertIn('recomputed once', actual['filter_contract'])
        boundary = method_call_contract((12, 12), cached_filter=False, execution_mode='eager')
        self.assertEqual(boundary['fft_padding'], [12, 12])
        self.assertIn('filter_fft', boundary['timing_scope'])
        self.assertNotIn('image_device_input_refresh', boundary['timing_scope'])
        self.assertNotIn('filter_spectrum_setup', boundary['excluded'])


class GraphLifetime(unittest.TestCase):
    def test_replay_consumes_new_inputs_and_keeps_previous_output(self):
        original = np.arange(6, dtype=np.float32)
        changed = original + 3
        class Replay:
            def set_inputs(self, value):
                self.value = value.copy()
            def __call__(self):
                return self.value * 2
        result = validate_graph_replay(Replay(), original, changed, original * 2,
                                       changed * 2, np.asarray)
        self.assertTrue(result['previous_output_preserved'])
        self.assertTrue(result['restored_input_exact'])

    def test_rejects_returning_overwritten_capture_storage(self):
        original = np.arange(6, dtype=np.float32)
        class AliasingReplay:
            output = np.empty(6, dtype=np.float32)
            def set_inputs(self, value):
                self.value = value.copy()
            def __call__(self):
                np.multiply(self.value, 2, out=self.output)
                return self.output
        with self.assertRaisesRegex(RuntimeError, 'lifetime'):
            validate_graph_replay(AliasingReplay(), original, original + 3,
                                  original * 2, (original + 3) * 2, np.asarray)


class MethodOrder(unittest.TestCase):
    def test_three_processes_balance_three_methods(self):
        methods = [('cubutterfly', object()), ('cupy', object()), ('torch', object())]
        orders = [rotate_methods(methods, i) for i in range(1, 4)]
        for position in range(3):
            self.assertEqual({order[position][0] for order in orders}, {'cubutterfly', 'cupy', 'torch'})
        for order in orders:
            self.assertEqual(dict(order), dict(methods))
        self.assertEqual([name for name, _ in methods], ['cubutterfly', 'cupy', 'torch'])

    def test_two_methods_alternate_and_invalid_index_fails(self):
        methods = ['cubutterfly', 'cupy']
        self.assertEqual([rotate_methods(methods, i) for i in range(1, 4)],
                         [methods, list(reversed(methods)), methods])
        with self.assertRaises(ValueError):
            rotate_methods(methods, 0)


class ConvolutionSemantics(unittest.TestCase):
    def test_reporting_without_tolerance_does_not_hide_explicit_failures(self):
        row = {'method': 'example', 'error': {'finite': True}, 'accepted': None}
        require_valid_reports([row])
        with self.assertRaises(RuntimeError):
            require_valid_reports([dict(row, accepted=False)])
        with self.assertRaises(RuntimeError):
            require_valid_reports([dict(row, error={'finite': False})])

    def test_batch_has_independent_images_and_shared_filter(self):
        images = np.arange(3*5*7, dtype=np.float64).reshape(3,5,7) - 11
        kernel = np.array([[1.,-2.,4.], [8.,3.,-7.]])
        expected = np.stack([convolve2d(image, kernel, mode='same') for image in images])
        for forward_norm in (False, True):
            actual = cpu_convolve(images, kernel, forward_norm=forward_norm)
            self.assertEqual(actual.shape, images.shape)
            np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-12)

    def test_error_report_rejects_broadcast_across_batch(self):
        with self.assertRaises(ValueError):
            error_distribution(np.ones((3,5,7)), np.ones((5,7)))

    def test_asymmetric_kernel_is_convolution_not_correlation(self):
        image = np.arange(35, dtype=np.float64).reshape(5, 7) - 9
        kernel = np.array([[1., -2., 4.], [8., 3., -7.]])
        expected = convolve2d(image, kernel, mode='same')
        np.testing.assert_allclose(cpu_convolve(image, kernel), expected, atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(cpu_convolve(image, kernel, forward_norm=True), expected, atol=1e-12, rtol=1e-12)

    def test_impulse_boundaries_and_odd_crop(self):
        image = np.zeros((7, 6)); image[0, 0] = 1; image[-1, -1] = 2
        kernel = np.arange(15, dtype=float).reshape(3, 5)
        np.testing.assert_allclose(cpu_convolve(image, kernel), convolve2d(image, kernel, mode='same'), atol=1e-12)

    def test_filter_larger_than_image_and_even_crop(self):
        image = np.array([[2., -3.], [5., 1.]])
        kernel = np.arange(24, dtype=float).reshape(4, 6)
        full = convolve2d(image, kernel, mode='full')
        np.testing.assert_array_equal(same_crop(full, image.shape, kernel.shape), convolve2d(image, kernel, mode='same'))
        np.testing.assert_allclose(cpu_convolve(image, kernel, forward_norm=True), convolve2d(image, kernel, mode='same'), atol=1e-12)

    def test_padding_prevents_circular_aliasing(self):
        self.assertEqual(padding_shape((512,512),(31,31)), (1024,1024))
        self.assertEqual(padding_shape((2048,2048),(127,127)), (4096,4096))

    def test_nonfinite_error_is_explicit(self):
        error = error_distribution(np.array([np.nan, np.inf]), np.ones(2))
        self.assertFalse(error['finite'])
        self.assertEqual(error['nonfinite_actual_count'], 2)
        self.assertIsNone(error['relative_l2'])


if __name__ == '__main__':
    unittest.main()
