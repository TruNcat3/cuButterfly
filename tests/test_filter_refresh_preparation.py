import argparse
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('filter_refresh', ROOT / 'scripts/prepare_filter_refresh.py')
refresh = importlib.util.module_from_spec(spec)
spec.loader.exec_module(refresh)


class FilterRefreshTests(unittest.TestCase):
    def report_fixture(self, directory):
        protocol = dict(warmup=2, repeats=3, samples=5, torch_control=True, atol=None, rtol=None,
                        process_index=1, plan_only=False, padding_policy='fast',
                        cache_filter_spectrum=True, require_cache=False, execution_mode='graph')
        case = dict(id='case', directory=str(directory), protocol=protocol, padding=[1080, 1080])
        preparation = dict(id='case', batch=16, image_shape=[1024, 1024], filter_shape=[31, 31],
                           sha256='input', artifact='case.npz', reference_padding=[1080, 1080])
        (directory / 'manifest.json').write_text(json.dumps(dict(contracts=[preparation])))
        identity = dict(library='/library', library_sha256='library-sha', runner_sha256='runner-sha',
                        adapter_sha256='adapter-sha', gpu_uuid='GPU-test', torch='torch-test',
                        compile_mode='research', algorithm_policy='default')
        binding = {'identity': {'runtime': {key: identity[key] for key in ('torch', 'compile_mode', 'algorithm_policy')}}}
        expected = {key: identity[key] for key in ('library', 'library_sha256', 'runner_sha256', 'adapter_sha256', 'gpu_uuid')}
        expected.update(prepared_binding_sha256='binding-sha', requested=protocol)
        api = refresh.runner_module()
        error = dict(finite=True, rms_error=0.1, relative_l2=0.1, reference_rms=1.0, reference_abs_max=2.0,
                     absolute_quantiles={str(q): 0.1 for q in (0.5, 0.9, 0.99, 0.999, 1.0)},
                     relative_quantiles={str(q): 0.1 for q in (0.5, 0.9, 0.99, 0.999, 1.0)})
        methods = ['cubutterfly', 'cupy_fftconvolve_native_padding', 'torch_cufft_same_padding_diagnostic']
        rows = [dict(method=method, batch=16, accepted=None, timing_status='measured_complete_call',
                     error=copy.deepcopy(error), call_contract=api.method_call_contract([1080, 1080], cached_filter=True, execution_mode='graph'),
                     timing=dict(samples_us=[10.] * 5, median_us=10., protocol='cuda_events_cuda_graph_replay', warmup_calls=2, calls_per_sample=3),
                     graph_changing_input_validation=dict(distinct_input_outputs=True, previous_output_preserved=True,
                                                         restored_input_exact=True, same_precision_cpu_errors=[copy.deepcopy(error), copy.deepcopy(error)]))
                for method in methods]
        report = dict(schema='fft-filter-complete-call-v2', identity=identity, requested=protocol,
                      prepared_binding_sha256='binding-sha', preparation_contract=preparation,
                      contract=api.runtime_contract(preparation, [1080, 1080], cached_filter=True, execution_mode='graph'),
                      complete_filter_call=True, execution_mode='graph', graph_capture_outside_timing=True,
                      filter_spectrum_cached=True, actual_cubutterfly_padding=[1080, 1080], cupy_native_fft_padding=[1080, 1080],
                      graph_call_contract=dict(input_device_copy_included=True, independent_output_copy_included=True),
                      validation_reference=dict(dtype='float32', fft_dtype='complex64', padding=[1080, 1080], precision_promotion=False),
                      method_order=methods, rows=rows)
        return case, binding, expected, report

    def test_frozen_receipt_protocols(self):
        config = refresh.read_json(refresh.DEFAULT_CONFIG)
        self.assertEqual([case['id'] for case in config['cases']],
                         ['image1024_filter31_batch16', 'image2048_filter31_batch16'])
        self.assertEqual([(case['padding'], case['warmup'], case['repeats'], case['samples'])
                          for case in config['cases']],
                         [([1080, 1080], 2, 3, 5), ([2100, 2100], 2, 2, 4)])

    def test_cpu_reference_and_changed_input_keep_fp32(self):
        api = refresh.runner_module()
        rng = np.random.default_rng(7)
        image = rng.standard_normal((2, 8, 8)).astype(np.float32)
        kernel = rng.standard_normal((3, 3)).astype(np.float32)
        reference = api.cpu_fp32_convolve(image, kernel, (10, 10))
        changed = api.cpu_fp32_convolve(np.roll(image, (13, -7), (-2, -1)).copy(), kernel, (10, 10))
        self.assertEqual(reference.dtype, np.float32)
        self.assertEqual(reference.shape, image.shape)
        self.assertFalse(np.array_equal(reference, changed))

    def test_forbidden_asset_symlink(self):
        with tempfile.TemporaryDirectory() as temp:
            asset = Path(temp) / 'asset'
            asset.symlink_to('/sdb/missing')
            with self.assertRaisesRegex(RuntimeError, '/sdb'):
                refresh.safe_path(asset / 'input.npz')

    def test_protocol_mismatch_stops_before_cuda(self):
        case = {'id': 'case', 'directory': '/tmp/filter-case', 'protocol': {'warmup': 2}}
        args = argparse.Namespace(prepared_binding=Path('/tmp/binding'), library=Path('/tmp/library'),
                                  output=Path('/tmp/filter-case'), contract=['case'], warmup=3)
        with patch.object(refresh, 'validate_prepared', return_value={'cases': [case]}):
            with self.assertRaisesRegex(RuntimeError, 'timing/protocol mismatch'):
                refresh.validate_runner_binding(args)

    def test_library_hash_mismatch_stops(self):
        with tempfile.TemporaryDirectory() as temp:
            library = Path(temp) / 'library'
            library.write_bytes(b'new library')
            with patch.object(refresh, 'read_json', return_value={'identity': {'library_sha256': 'old'}}):
                with self.assertRaisesRegex(RuntimeError, 'library identity mismatch'):
                    refresh.validate_prepared(Path(temp) / 'prepared.json', library)

    def test_complete_report_is_reused_without_modification(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            case, binding, expected, report = self.report_fixture(directory)
            path, intent = directory / 'report.json', directory / 'intent.json'
            path.write_text(json.dumps(report))
            original = path.read_bytes()
            self.assertTrue(refresh._existing_report(path, intent, expected, binding, case))
            self.assertEqual(path.read_bytes(), original)

    def test_incomplete_same_binding_report_is_archived(self):
        for mutation in ('missing_control', 'nonfinite', 'graph_restore', 'timing', 'malformed'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temp:
                directory = Path(temp)
                case, binding, expected, report = self.report_fixture(directory)
                if mutation == 'missing_control':
                    report['rows'].pop()
                elif mutation == 'nonfinite':
                    report['rows'][0]['error']['finite'] = False
                elif mutation == 'graph_restore':
                    report['rows'][0]['graph_changing_input_validation']['restored_input_exact'] = False
                elif mutation == 'malformed':
                    report['rows'] = [None, None, None]
                else:
                    report['rows'][0]['timing']['samples_us'].pop()
                path, intent = directory / 'report.json', directory / 'intent.json'
                path.write_text(json.dumps(report))
                intent.write_text(json.dumps(expected))
                original = path.read_bytes()
                self.assertFalse(refresh._existing_report(path, intent, expected, binding, case))
                self.assertFalse(path.exists())
                archived, = directory.glob('incomplete_reports/*/report.json')
                self.assertEqual(archived.read_bytes(), original)

    def test_unknown_report_identity_is_preserved_and_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            case, binding, expected, report = self.report_fixture(directory)
            report['identity']['library_sha256'] = 'unknown'
            path = directory / 'report.json'
            path.write_text(json.dumps(report))
            original = path.read_bytes()
            with self.assertRaisesRegex(RuntimeError, 'unknown prior report identity'):
                refresh._existing_report(path, directory / 'intent.json', expected, binding, case)
            self.assertEqual(path.read_bytes(), original)

    def test_truncated_report_requires_matching_persisted_intent(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            case, binding, expected, _ = self.report_fixture(directory)
            path, intent = directory / 'report.json', directory / 'intent.json'
            path.write_text('{truncated')
            with self.assertRaisesRegex(RuntimeError, 'without matching run intent'):
                refresh._existing_report(path, intent, expected, binding, case)
            intent.write_text(json.dumps(expected))
            self.assertFalse(refresh._existing_report(path, intent, expected, binding, case))
            self.assertTrue(next(directory.glob('incomplete_reports/*/report.json')).exists())

    def test_zero_exit_without_report_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            case, binding, expected, _ = self.report_fixture(directory)
            binding['cases'] = [case]
            with patch.object(refresh, 'validate_prepared', return_value=binding), \
                 patch.object(refresh, '_report_identity', return_value=expected), \
                 patch.object(refresh.subprocess, 'run') as run:
                with self.assertRaisesRegex(RuntimeError, 'without the expected report'):
                    refresh.run_prepared(directory / 'prepared.json', directory / 'library', 'GPU-test')
            run.assert_called_once()

    def test_temporary_report_is_archived_with_matching_intent(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            case, binding, expected, _ = self.report_fixture(directory)
            path, intent = directory / 'report.json', directory / 'intent.json'
            temporary = directory / 'report.json.tmp'
            temporary.write_text('{partial atomic write')
            intent.write_text(json.dumps(expected))
            self.assertFalse(refresh._existing_report(path, intent, expected, binding, case))
            self.assertEqual(next(directory.glob('incomplete_reports/*/report.json.tmp')).read_text(), '{partial atomic write')

    def test_code_rebind_never_loads_or_recomputes_reference_arrays(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            case, _, _, report = self.report_fixture(directory)
            frozen = dict(id=case['id'], source='frozen-source')
            config = directory / 'config.json'
            config.write_text(json.dumps(dict(cases=[frozen])))
            identity = dict(config=str(config), code={'helper': 'old'})
            case['source'] = dict(archive_sha256='unchanged')
            binding = dict(identity=identity, cases=[case])
            path = directory / 'prepared.json'
            path.write_text(json.dumps(binding))
            checkpoint_path = directory / 'cpu_prepared.json'
            checkpoint_path.write_text(json.dumps(dict(contract=report['preparation_contract'], resume_key=dict(
                identity=identity, source=case['source'], protocol=frozen))))
            with patch.dict(refresh.os.environ, {'CUDA_VISIBLE_DEVICES': ''}), \
                 patch.object(refresh, 'validate_prepared', return_value=binding), \
                 patch.object(refresh, 'code_identity', return_value={'helper': 'new'}), \
                 patch.object(refresh.np, 'load', side_effect=AssertionError('must not load arrays')):
                refresh.rebind_helper_code(path, directory / 'library')
            updated = json.loads(path.read_text())
            self.assertEqual(updated['identity']['code'], {'helper': 'new'})
            self.assertEqual(json.loads(checkpoint_path.read_text())['resume_key']['identity']['code'], {'helper': 'new'})
            self.assertTrue(next(directory.glob('code_rebind_history/*/prepared.json')).exists())

    def test_second_case_interruption_does_not_rerun_completed_first_case(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            first, second = directory / 'first', directory / 'second'
            first.mkdir()
            second.mkdir()
            first_case, binding, expected, report = self.report_fixture(first)
            second_case = dict(first_case, id='second', directory=str(second))
            report_path = first / 'case.current_graph_fast.json'
            report_path.write_text(json.dumps(report))
            binding['cases'] = [first_case, second_case]
            with patch.object(refresh, 'validate_prepared', return_value=binding), \
                 patch.object(refresh, '_report_identity', return_value=expected), \
                 patch.object(refresh.subprocess, 'run', side_effect=RuntimeError('GPU interrupted')) as run:
                with self.assertRaisesRegex(RuntimeError, 'GPU interrupted'):
                    refresh.run_prepared(directory / 'prepared.json', directory / 'library', 'GPU-test')
            run.assert_called_once()
            self.assertIn(str(second), run.call_args.args[0])
            self.assertTrue(report_path.exists())


if __name__ == '__main__':
    unittest.main()
