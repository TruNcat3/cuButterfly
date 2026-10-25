"""CPU-only campaign protocol and full logical/physical validation tests."""
import copy
import importlib.util
import pathlib
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('campaign_runner', ROOT / 'scripts/run_semantic_campaign.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def semantics(**changes):
    result = {'schema': 'cubutterfly-semantic-v1', 'operator': 0, 'storage': 6,
              'compute': 1, 'shape': [4], 'batch': 3, 'inverse': 0,
              'normalize_inverse': False, 'placement': 0, 'fft_type': 0,
              'fft_output_scale': 1.0, 'length_mode': 0, 'input_strides': [2],
              'output_strides': [3], 'input_batch_stride': 13,
              'output_batch_stride': 17, 'input_orders': [1], 'output_orders': [0],
              'ntt_input_layout': 0, 'ntt_output_layout': 0, 'modulus': None,
              'word_bits': None, 'matrices': [[]]}
    result.update(changes)
    return result


class CampaignTests(unittest.TestCase):
    def test_gpu_uuid_prefix_does_not_change_identity(self):
        target = 'GPU-2971e500-d3f6-d7e6-e5f2-cb67a06fff02'
        self.assertTrue(runner.same_gpu_uuid(target, target[4:]))
        self.assertTrue(runner.same_gpu_uuid(target, target))
        self.assertFalse(runner.same_gpu_uuid(target, 'GPU-0eae5f07-33de-321b-70a9-ffeae49309e6'))
        with self.assertRaises(ValueError):
            runner.same_gpu_uuid(target, '')

    def test_selection_preserves_semantics(self):
        original = {'semantics': semantics(), 'selection': {'execution_mode': 0,
                    'algorithm_policy': 2, 'explicit_algorithm': 'old'}}
        for timing in (False, True):
            new = runner.selection_descriptor(original, timing)
            self.assertEqual(new['semantics'], original['semantics'])
            self.assertEqual(new['selection']['algorithm_policy'], int(timing))
            self.assertEqual(original['selection']['explicit_algorithm'], 'old')
        bad = copy.deepcopy(original)
        bad['semantics']['batch'] = 1
        with self.assertRaises(AssertionError):
            runner.assert_semantics(original, bad)

    def test_full_check_last_batch_and_nan(self):
        expected = np.zeros((3, 1024), np.float32)
        actual = expected.copy()
        actual[-1, -1] = 1
        with self.assertRaises(AssertionError):
            runner.check_full(actual, expected, 2)
        for value in (float('nan'), float('inf')):
            actual[-1, -1] = value
            with self.assertRaises(AssertionError):
                runner.check_full(actual, expected, 2)

    def test_uint64_does_not_floatcast(self):
        expected = np.array([2**63 + 7], np.uint64)
        actual = expected + np.uint64(1)
        self.assertEqual(float(actual[0]), float(expected[0]))
        with self.assertRaises(AssertionError):
            runner.check_full(actual, expected, 9)
        self.assertTrue(runner.check_full(expected, expected, 9)['exact'])
        with self.assertRaises(AssertionError):
            runner.check_full(expected.astype(np.float64), expected, 9)

    def test_error_report_never_claims_float_pass_or_relaxes_integer(self):
        actual, expected = np.array([1.1], np.float32), np.array([1.], np.float64)
        metrics = runner.check_full(actual, expected, 2, 'error-report')
        self.assertEqual(metrics['verdict'], 'accuracy-review-required')
        self.assertEqual(metrics['mismatches'], 1)
        self.assertEqual(metrics['atol'], 3e-5)
        with self.assertRaises(AssertionError):
            runner.check_full(actual, expected, 2)
        with self.assertRaises(AssertionError):
            runner.check_full(np.array([3], np.uint64), np.array([2], np.uint64), 9, 'error-report')

    def test_all_element_distribution_discrete_quantiles_and_bins(self):
        values = np.array([0, 1e-16, 1e-15, 1, 100], np.float64)
        result = runner.distribution(values)
        self.assertEqual(result['elements'], 5)
        self.assertEqual(result['zero_count'], 1)
        self.assertEqual(result['quantiles']['0'], 0)
        self.assertEqual(result['quantiles']['0.25'], 1e-16)
        self.assertEqual(result['quantiles']['0.5'], 1e-15)
        self.assertEqual(result['quantiles']['0.75'], 1)
        self.assertEqual(result['quantiles']['1'], 100)
        self.assertEqual(sum(result['positive_log_bin_counts']), 4)
        self.assertEqual(result['positive_log_bin_counts'][0], 0)
        self.assertEqual(result['positive_log_bin_counts'][1], 1)
        self.assertEqual(result['positive_log_bin_counts'][2], 1)
        metrics = runner.check_full(np.array([2., 4., 1.]), np.array([1., 2., 0.]), 3, 'error-report')
        self.assertEqual(metrics['error_distributions']['absolute']['elements'], 3)
        self.assertEqual(metrics['error_distributions']['relative_nonzero_reference']['elements'], 2)
        self.assertEqual(metrics['error_distributions']['absolute_zero_reference']['elements'], 1)
        self.assertAlmostEqual(metrics['l2_relative'], np.sqrt(6 / 5))

    def test_lossless_artifacts_and_resume_rejects_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            actual = np.array([1 + 2j, 4 + 5j, 9 + 0j], np.complex64)
            expected = np.array([1 + 1j, 2 + 0j, 0 + 0j], np.complex128)
            artifact = runner.retain_error_artifact(pathlib.Path(directory) / 'errors', 'abc', 'cufft', actual, expected)
            with np.load(artifact['path'], allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved['actual'], actual)
                np.testing.assert_array_equal(saved['reference'], expected)
                self.assertEqual(saved['actual'].dtype, actual.dtype)
                self.assertEqual(saved['reference'].dtype, expected.dtype)
                np.testing.assert_array_equal(saved['nonzero_reference_flat_indices'], [0, 1])
                np.testing.assert_array_equal(saved['zero_reference_flat_indices'], [2])
                self.assertEqual(saved['abs_error_zero_reference'][0], 9)
            row = {'id': 'abc', 'status': 'accuracy-review-required',
                   'baselines': [{'name': 'cufft', 'status': 'accuracy-review-required', 'error_artifact': artifact}]}
            index = runner.write_case_checkpoint(directory, row)
            report = {'identity': {}, 'rows': [index], 'rows_sha256': runner.digest([index])}
            self.assertEqual(runner.validate_resume(report, {}, directory, {'abc'}), {'abc'})
            pathlib.Path(artifact['path']).write_bytes(b'corrupt')
            with self.assertRaises(ValueError):
                runner.validate_resume(report, {}, directory, {'abc'})
            pathlib.Path(artifact['path']).unlink()
            with self.assertRaises(ValueError):
                runner.validate_resume(report, {}, directory, {'abc'})

    def test_dao_contract_routing_preserves_exact_semantics_before_allocation(self):
        import sys
        calls = []

        def support(s):
            calls.append(s)
            supported = s['operator'] == 2 and s['compute'] == 1 and s['storage'] in (0, 1, 2)
            return {'supported': supported, 'status': 'supported' if supported else 'unsupported-semantic',
                    'reason': 'original storage and FP32 arithmetic required', 'metadata': {'storage': s['storage']}}

        fake_dao = types.SimpleNamespace(support=support)
        with patch.dict(sys.modules, {'semantic_dao_baseline': fake_dao}):
            for storage in (0, 1, 2):
                s = semantics(operator=2, storage=storage, compute=1, placement=1)
                self.assertTrue(runner.baseline_contract('daofht', s)['supported'])
                self.assertIs(calls[-1], s)
            s = semantics(operator=2, storage=1, compute=0)
            result = runner.baseline_contract('daofht', s)
            self.assertEqual(result['status'], 'unsupported-semantic')
            fake_torch = types.SimpleNamespace(OutOfMemoryError=type('OutOfMemoryError', (Exception,), {}))
            rows = runner.run_baselines({'descriptor': {'semantics': s}}, fake_torch, None,
                                        None, None, None, None, False, None)
            dao = next(row for row in rows if row['name'] == 'daofht')
            self.assertEqual(dao['status'], 'missing-baseline')
            self.assertEqual(dao['support_status'], 'unsupported-semantic')
            self.assertEqual(dao['support_metadata']['storage'], 1)
        self.assertFalse(runner.baseline_contract('cufft', semantics(storage=6, compute=2))['supported'])
        self.assertTrue(runner.baseline_contract('cufft', semantics(storage=6, compute=1))['supported'])

    def test_gpuntt_contract_routes_exact_semantics_and_library(self):
        import sys
        calls = []

        def support(s, librarypath):
            calls.append((s, librarypath))
            supported = s['operator'] == 1 and s['storage'] in (8, 9) and s['compute'] == (3 if s['storage'] == 8 else 4)
            return {'supported': supported, 'status': 'supported' if supported else 'unsupported-semantic',
                    'reason': 'exact unsigned compute contract', 'metadata': {}}

        fake = types.SimpleNamespace(DEFAULT_LIBRARY='default-bridge', support=support)
        with patch.dict(sys.modules, {'semantic_gpuntt_baseline': fake}):
            for storage, compute, bits in ((8, 3, 32), (9, 4, 64)):
                s = semantics(operator=1, storage=storage, compute=compute, word_bits=bits, modulus=17)
                self.assertTrue(runner.baseline_contract('gpu-ntt', s, gpuntt_library='requested-bridge')['supported'])
                self.assertIs(calls[-1][0], s)
                self.assertEqual(calls[-1][1], 'requested-bridge')
            s['compute'] = 1
            self.assertFalse(runner.baseline_contract('gpu-ntt', s)['supported'])
            self.assertEqual(calls[-1][1], 'default-bridge')

    def test_unsigned_layout_transfers_preserve_words_as_byte_rows(self):
        import torch
        for storage, dtype, bits in ((8, torch.uint32, 32), (9, torch.uint64, 64)):
            s = semantics(operator=1, storage=storage, compute=3 if bits == 32 else 4,
                          modulus=17, word_bits=bits)
            values = (np.arange(12, dtype=np.uint64) + np.uint64(2**(bits - 1))).astype(runner.DTYPES[storage]).reshape(3, 4)
            width = runner.DTYPES[storage].itemsize
            raw = runner.physical_bytes(s, 'input', values, 40 * width)
            input_tensor = torch.from_numpy(raw).view(dtype)
            source = torch.empty((3, 4), dtype=dtype)
            output = torch.empty_like(source)
            initial = np.full(48 * width, 0xA5, np.uint8)
            physical_output = torch.from_numpy(initial.copy()).view(dtype)
            transfer_input, transfer_source, transfer_output, transfer_result = runner.baseline_transfer_views(
                torch, s, input_tensor, source, physical_output, output)
            self.assertEqual(transfer_input.dtype, torch.uint8)
            self.assertEqual(transfer_input.shape[1], width)
            torch.index_select(transfer_input, 0, torch.from_numpy(runner.layout_indices(s, 'input')), out=transfer_source)
            np.testing.assert_array_equal(source.view(torch.uint8).numpy().view(runner.DTYPES[storage]).reshape(3, 4), values)
            output.view(torch.uint8).copy_(source.view(torch.uint8))
            transfer_output.index_copy_(0, torch.from_numpy(runner.layout_indices(s, 'output')), transfer_result)
            result_raw = physical_output.view(torch.uint8).numpy()
            self.assertTrue(runner.check_full(runner.output_from_bytes(s, result_raw), values, storage, 'error-report')['exact'])
            runner.check_padding(s, result_raw, initial)

    def test_layout_full_byte_storage_and_padding(self):
        s = semantics()
        source = runner.make_source(s)
        raw = runner.physical_bytes(s, 'input', source, 40 * 8)
        np.testing.assert_array_equal(raw.view(np.complex64)[runner.layout_indices(s, 'input')], source.reshape(-1))
        expected = runner.reference(s, source)
        output = runner.physical_bytes(s, 'output', expected, 48 * 8)
        self.assertEqual(expected.dtype, np.dtype('complex128'))
        np.testing.assert_array_equal(runner.output_from_bytes(s, output), expected.astype(np.complex64).reshape(-1))
        initial = np.full(output.size, 0xA5, np.uint8)
        runner.check_padding(s, output, initial)
        output[8] ^= 1  # Element 1 is padding for output stride 3.
        with self.assertRaises(AssertionError):
            runner.check_padding(s, output, initial)

    def test_real_fft_side_widths_and_embedding(self):
        s = semantics(storage=2, fft_type=1, shape=[3, 5], length_mode=1,
                      input_strides=[7, 1], output_strides=[8, 1],
                      input_orders=[0, 0], output_orders=[0, 0], matrices=[[], []])
        self.assertEqual(runner.side_extents(s, 'input'), [3, 5])
        self.assertEqual(runner.side_extents(s, 'output'), [4, 5])
        self.assertEqual(runner.side_dtype(s, 'input').itemsize, 4)
        self.assertEqual(runner.side_dtype(s, 'output').itemsize, 8)
        expected = runner.reference(s, runner.make_source(s))
        self.assertEqual(expected.shape, (3, 4, 5))

    def test_embedding_float_oracle_uses_high_precision_arithmetic(self):
        source = np.array([[2**24, 1, -(2**24)]], np.float32)
        expected = np.array([[1, -1, 2**25 + 1, 2**25 - 1]], np.float64)
        for op in (2, 5):
            s = semantics(operator=op, storage=2, batch=1, shape=[3], length_mode=1,
                          matrices=[[[1., 1., 1., -1.]]] if op == 5 else [[]])
            reference = runner.reference(s, source)
            self.assertEqual(reference.dtype, np.dtype('float64'))
            np.testing.assert_array_equal(reference, expected)
            self.assertNotEqual(float(reference[0, 2]), float(np.float32(reference[0, 2])))

    def test_low_precision_storage_is_raw_exact(self):
        for storage in (1, 4, 5):
            s = semantics(operator=2, storage=storage)
            source = runner.make_source(s)
            raw = runner.physical_bytes(s, 'output', source, 48 * runner.side_dtype(s, 'output').itemsize)
            np.testing.assert_array_equal(runner.output_from_bytes(s, raw), source.reshape(-1))
        halfway = np.array([1 + 2**-8, 1 + 3 * 2**-8], np.float32)
        np.testing.assert_array_equal(runner.encode_storage(halfway, 1), np.array([0x3F80, 0x3F82], np.uint16))

    def test_complex_decode_preserves_independent_nonfinite_components(self):
        components = np.array([[1., np.inf], [-0., -np.inf], [np.inf, 2.],
                               [3., np.nan], [-0., -0.]], dtype=np.float32)
        for storage in (4, 5):
            packed = (components.astype(np.float16) if storage == 4 else
                      runner.encode_storage(components, 1)).view(np.dtype('V4')).reshape(-1)
            with np.errstate(invalid='raise'):
                decoded = runner.decode_storage(packed, storage)
            np.testing.assert_array_equal(decoded.real, components[:, 0])
            np.testing.assert_array_equal(decoded.imag, components[:, 1])
            np.testing.assert_array_equal(np.signbit(decoded.real), np.signbit(components[:, 0]))
            np.testing.assert_array_equal(np.signbit(decoded.imag), np.signbit(components[:, 1]))

    def test_range_diagnostics_distinguish_expected_and_extra_nonfinite(self):
        actual = np.empty(3, np.complex64)
        actual.real = [np.inf, np.nan, 1.]
        actual.imag = [0., 1., -np.inf]
        expected = np.array([70000., 2.+1j, 1.-80000j], np.complex128)
        report = runner.floating_range_report(actual, expected, 4)
        self.assertEqual(report['storage_max_finite'], 65504)
        self.assertEqual(report['reference_exceeds_max_finite_elements'], 2)
        self.assertEqual(report['actual_nonfinite_elements'], 3)
        self.assertEqual(report['actual_nonfinite_at_reference_in_range'], 1)
        self.assertEqual(report['components']['real']['actual_nan'], 1)
        with self.assertRaises(AssertionError):
            runner.check_full(actual, expected, 4, 'error-report')
        self.assertIsNone(runner.floating_range_report(np.array([1], np.uint64), np.array([1], np.uint64), 9))

    def test_oracle_guard_is_gap(self):
        s = semantics()
        with self.assertRaises(runner.Gap) as context:
            runner.reference(s, runner.make_source(s), max_elements=11)
        self.assertEqual(context.exception.status, 'validation-unavailable')
        s = semantics(operator=1, storage=9, modulus=17, word_bits=64)
        with self.assertRaises(runner.Gap) as context:
            runner.reference(s, runner.make_source(s), ntt_max_operations=1)
        self.assertEqual(context.exception.status, 'validation-unavailable')
        s['ntt_input_layout'] = 1
        with self.assertRaises(runner.Gap) as context:
            runner.reference(s, runner.make_source(s))
        self.assertEqual(context.exception.status, 'runner-gap')

    def test_moderate_ntt_uses_full_scalable_reference(self):
        s = semantics(operator=1, storage=9, modulus=576460756061519873,
                      word_bits=64, shape=[65536], batch=1)
        source = np.ones((1, 65536), dtype=np.uint64)
        actual = runner.reference(s, source)
        expected = np.zeros_like(source)
        expected[0, 0] = 65536
        np.testing.assert_array_equal(actual, expected)

    def test_integer_zeta_inverse_and_ntt(self):
        for op in (3, 4):
            s = semantics(operator=op, storage=9)
            source = runner.make_source(s)
            forward = runner.reference(s, source)
            inverse = runner.reference(dict(s, inverse=1), forward)
            np.testing.assert_array_equal(inverse, source)
        s = semantics(operator=1, storage=9, modulus=17, word_bits=64)
        source = runner.make_source(s)
        forward = runner.reference(s, source)
        inverse = runner.reference(dict(s, inverse=1, normalize_inverse=True), forward)
        np.testing.assert_array_equal(inverse, source)
        s['modulus'] = 1152921504606584833
        source = runner.make_source(s)
        self.assertEqual(int(source.reshape(-1)[0]), s['modulus'] - 1)
        self.assertGreater(int(source.max()), 2**53)
        self.assertFalse(np.array_equal(source[0], source[1]))
        forward = runner.reference(s, source)
        inverse = runner.reference(dict(s, inverse=1, normalize_inverse=True), forward)
        self.assertTrue(runner.check_full(inverse, source, 9)['exact'])

    def test_resume_checksum_identity_and_missing_never_pass(self):
        rows = [{'id': 'a', 'status': 'pass'}, {'id': 'b', 'status': 'runner-gap'},
                {'id': 'c', 'status': 'runtime-error'}]
        report = {'identity': {'library': 'x'}, 'rows': rows, 'rows_sha256': runner.digest(rows)}
        self.assertEqual(runner.validate_resume(report, report['identity']), {'a', 'b'})
        with self.assertRaises(ValueError):
            runner.validate_resume(report, {'library': 'changed'})
        report['rows'][1]['status'] = 'pass'
        with self.assertRaises(ValueError):
            runner.validate_resume(report, report['identity'])

    def test_unsupported_and_capability_gap_before_oracle(self):
        import json
        import sys
        descriptor = {'semantics': semantics(), 'selection': {}}
        case = {'id': 'case', 'descriptor': descriptor, 'populations': ['conformance']}

        class Error(RuntimeError):
            def __init__(self, status):
                self.status = status

        class Descriptor:
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
            def validate(self):
                pass
            def json(self):
                return json.dumps(descriptor)
            def capabilities_json(self):
                return json.dumps({'status': 'missing-lowering', 'reason': 'honest gap'})

        fake_cb = types.SimpleNamespace(Error=Error, Descriptor=types.SimpleNamespace(from_json=lambda *args: Descriptor()))
        fake_torch = types.SimpleNamespace(OutOfMemoryError=type('OutOfMemoryError', (Exception,), {}))
        with patch.dict(sys.modules, {'cubutterfly': fake_cb}), patch.object(runner, 'make_source', side_effect=AssertionError('oracle should not run')):
            row = runner.run_case(case, None, fake_torch, None)
            self.assertEqual(row['status'], 'missing-lowering')
            self.assertEqual(row['capabilities']['reason'], 'honest gap')
            fake_cb.Descriptor.from_json = lambda *args: (_ for _ in ()).throw(Error(3))
            self.assertEqual(runner.run_case(case, None, fake_torch, None)['status'], 'unsupported')
            fake_cb.Descriptor.from_json = lambda *args: (_ for _ in ()).throw(Error(1))
            self.assertEqual(runner.run_case(case, None, fake_torch, None)['status'], 'runtime-error')

    def test_fingerprints_and_calibration_are_strict(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / 'library'
            path.write_bytes(b'first')
            item = {'path': str(path), 'sha256': runner.sha256_file(path)}
            manifest = {'fingerprints': {'library': item, 'sources': [item]}}
            runner.validate_fingerprints(manifest, path)
            path.write_bytes(b'changed')
            with self.assertRaises(ValueError):
                runner.validate_fingerprints(manifest, path)
        identity = {'library_sha256': 'hash', 'gpu_name': 'GPU', 'gpu_total_memory': 8,
                    'compute_capability': '8.0', 'compile_mode': 'research'}
        calibration = {'schema': 'cubutterfly-local-calibration-v1', 'status': 'ready-with-measured-mappings',
                       'measurement_identity': {'library_sha256': 'hash', 'verify_batches': 0, 'compile_mode': 'research'},
                       'device': 'GPU', 'global_memory_bytes': 8, 'compute_capability': '8.0',
                       'hardware_rates_remeasured': True, 'inventory_complete': False}
        self.assertFalse(runner.validate_calibration(calibration, identity)['inventory_complete'])
        with self.assertRaises(ValueError):
            runner.validate_calibration(calibration, dict(identity, library_sha256='other'))

    def test_atomic_case_checkpoint_tampering_missing_and_foreign(self):
        import json
        with tempfile.TemporaryDirectory() as directory:
            row = {'id': 'abc', 'status': 'pass', 'mapping': {'kind': 'native'}, 'baselines': []}
            index = runner.write_case_checkpoint(directory, row)
            report = {'identity': {'a': 1}, 'rows': [index], 'rows_sha256': runner.digest([index])}
            self.assertEqual(runner.validate_resume(report, report['identity'], directory, {'abc'}), {'abc'})
            with self.assertRaises(ValueError):
                runner.validate_resume(report, report['identity'], directory, {'other'})
            path = pathlib.Path(directory) / index['path']
            checkpoint = json.loads(path.read_text())
            checkpoint['row']['mapping']['kind'] = 'tampered'
            path.write_text(json.dumps(checkpoint))
            with self.assertRaises(ValueError):
                runner.validate_resume(report, report['identity'], directory, {'abc'})
            path.unlink()
            with self.assertRaises(ValueError):
                runner.validate_resume(report, report['identity'], directory, {'abc'})

    def test_pending_accuracy_is_available_and_never_pass_or_gap(self):
        with tempfile.TemporaryDirectory() as directory:
            row = {'id': 'pending', 'status': 'accuracy-review-required',
                   'baselines': [{'status': 'accuracy-review-required'}, {'status': 'pass'}, {'status': 'not-applicable'}]}
            index = runner.write_case_checkpoint(directory, row)
            self.assertEqual(index['baseline_gaps'], 0)
            self.assertEqual(index['baseline_pending_accuracy'], 1)
            report = {'rows': [index], 'remaining_case_ids': [], 'invalid_request_count': 0}
            self.assertEqual(runner.completion_status(report), 'complete-pending-accuracy')
            summary = runner.status_summary([index])
            self.assertEqual(summary['cases']['pass'], 0)
            self.assertEqual(summary['cases']['pending-accuracy'], 1)
            self.assertEqual(summary['baselines']['pass'], 1)
            self.assertEqual(summary['baselines']['pending-accuracy'], 1)
            self.assertEqual(summary['baselines']['not-applicable'], 1)
            row['baselines'].extend([{'status': 'missing-baseline'}, {'status': 'baseline-error'}])
            index = runner.write_case_checkpoint(directory, row)
            self.assertEqual(index['baseline_gaps'], 1)
            self.assertEqual(index['baseline_errors'], 1)
            report['rows'] = [index]
            self.assertEqual(runner.completion_status(report), 'complete-with-gaps-and-pending-accuracy')

    def test_unrelated_baseline_domains_are_not_applicable(self):
        for op in range(6):
            applicable = {0: {'cufft', 'vkfft'}, 1: {'gpu-ntt'}, 2: {'daofht'}}.get(op, set())
            for name in ('cufft', 'vkfft', 'daofht', 'gpu-ntt'):
                if name in applicable:
                    continue
                self.assertEqual(runner.baseline_contract(name, semantics(operator=op))['status'], 'not-applicable')

    def test_calibration_registry_bound_and_mode_conflict(self):
        import os
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / 'registry'
            profile = pathlib.Path(directory) / 'profile'
            cost_model = pathlib.Path(directory) / 'cost_model.json'
            path.write_bytes(b'registry')
            profile.write_bytes(b'profile')
            cost_model.write_text('{"status":"calibrated-local-warning", "usable_for_unmeasured_ranking":false, "whole_plan_validation":{"covered_count":48,"checks":["not-summary"]}}')
            calibration = {'measurement_identity': {'compile_mode': 'research'},
                           'registry_path': str(path), 'hardware_rates_source': str(profile),
                           'cost_model': str(cost_model)}
            with patch.dict(os.environ, {'CUBUTTERFLY_COMPILE_MODE': 'research'}):
                bound = runner.bind_calibration_environment(calibration)
                self.assertEqual(os.environ['CUBUTTERFLY_REGISTRY'], str(path))
                self.assertEqual(bound['registry_sha256'], runner.sha256_file(path))
                self.assertEqual(bound['cost_model_sha256'], runner.sha256_file(cost_model))
                self.assertFalse(bound['model_evidence']['usable_for_unmeasured_ranking'])
                self.assertEqual(bound['model_evidence']['whole_plan_validation'], {'covered_count': 48})
            with patch.dict(os.environ, {'CUBUTTERFLY_COMPILE_MODE': 'production'}):
                with self.assertRaises(ValueError):
                    runner.bind_calibration_environment(calibration)


if __name__ == '__main__':
    unittest.main()
