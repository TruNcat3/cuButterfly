"""CPU preparation and provenance checks for the independent range control."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TEST_GPU = 'GPU-2971e500-d3f6-d7e6-e5f2-cb67a06fff02'
sys.path.insert(0, str(ROOT / 'scripts'))
spec = importlib.util.spec_from_file_location('fft_range_control', ROOT / 'scripts/run_fft_range_control.py')
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)


class RangeControlTests(unittest.TestCase):
    def fixture(self, directory, n=16, **changes):
        s = {'schema': 'cubutterfly-semantic-v1', 'operator': 0, 'storage': 4,
             'compute': 0, 'shape': [n], 'batch': 1, 'inverse': 0,
             'normalize_inverse': False, 'placement': 0, 'fft_type': 0,
             'fft_output_scale': 1.0, 'length_mode': 0, 'input_strides': [1],
             'output_strides': [1], 'input_batch_stride': n, 'output_batch_stride': n,
             'input_orders': [0], 'output_orders': [0], 'ntt_input_layout': 0,
             'ntt_output_layout': 0, 'modulus': None, 'word_bits': None, 'matrices': [[]]}
        s.update(changes)
        artifact = directory / 'original.npz'
        np.savez(artifact, actual=np.array([np.inf]), reference=np.array([1]))
        descriptor = {'schema': 'cubutterfly-descriptor-v1', 'semantics': s,
                      'selection': {'algorithm_policy': 0, 'execution_mode': 0, 'explicit_algorithm': ''}}
        row = {'id': c.r.digest(s), 'descriptor': descriptor, 'status': 'correctness-failure',
               'mapping': {'contract': s, 'kind': 'butterfly'},
               'error_artifact': {'path': str(artifact), 'sha256': c.r.sha256_file(artifact)}}
        (directory / 'rows').mkdir()
        path = directory / 'rows' / (row['id'] + '.json')
        path.write_text(json.dumps({'row': row, 'sha256': c.r.digest(row)}))
        library = directory / 'library.so'
        library.write_bytes(b'cpu-only-library-fingerprint')
        indices = [{'id': row['id'], 'sha256': c.r.digest(row),
                    'path': str(path.relative_to(directory))}]
        (directory / 'results.json').write_text(json.dumps({
            'schema': 'cubutterfly-semantic-results-v1', 'rows': indices,
            'rows_sha256': c.r.digest(indices),
            'identity': {'library_sha256': c.r.sha256_file(library), 'gpu_uuid': TEST_GPU}}))
        for name in ('registry.json', 'rates.json', 'model.json'):
            (directory / name).write_text('{}')
        calibration = {'measurement_identity': {'compile_mode': 'research',
                                                'library_sha256': c.r.sha256_file(library)},
                       'registry_path': str(directory / 'registry.json'),
                       'hardware_rates_source': str(directory / 'rates.json'),
                       'cost_model': str(directory / 'model.json')}
        manifest = directory / 'calibration.json'
        manifest.write_text(json.dumps(calibration))
        return types.SimpleNamespace(row=path, library=library, output=directory / 'control',
                                     gpu_uuid=TEST_GPU, calibration_manifest=manifest,
                                     vkfft_library=None, gpuntt_library=None), row

    def test_prepare_complete_source_reference_and_exact_semantics(self):
        with tempfile.TemporaryDirectory() as temp:
            args, original = self.fixture(Path(temp))
            before = args.row.read_bytes()
            report, source, expected, _ = c.prepare(args)
            self.assertEqual(report['descriptor'], original['descriptor'])
            self.assertEqual(report['mapping'], original['mapping'])
            self.assertEqual(report['original_semantic_id'], original['id'])
            self.assertNotEqual(report['id'], original['id'])
            self.assertEqual(args.row.read_bytes(), before)
            np.testing.assert_array_equal(expected, c.r.reference(original['descriptor']['semantics'], source))
            with np.load(report['source_artifact']['path']) as arrays:
                np.testing.assert_array_equal(arrays['source'], source)
                np.testing.assert_array_equal(arrays['reference'], expected)
                np.testing.assert_array_equal(c.r.decode_storage(arrays['storage'], 4), source)
            self.assertEqual(report['status'], 'prepared')
            evidence = report['source_evidence']
            self.assertEqual(evidence['original_measurement_identity']['gpu_uuid'], args.gpu_uuid)
            self.assertEqual(evidence['original_report_sha256'],
                             c.r.sha256_file(Path(temp) / 'results.json'))

    def test_fwht_prepare_preserves_real_fp16_and_compute_contract(self):
        for compute in (0, 1):
            with self.subTest(compute=compute), tempfile.TemporaryDirectory() as temp:
                args, original = self.fixture(Path(temp), operator=2, storage=0, compute=compute)
                report, source, expected, _ = c.prepare(args)
                self.assertEqual(source.dtype, np.dtype('float16'))
                self.assertEqual(report['descriptor'], original['descriptor'])
                self.assertEqual(report['mapping'], original['mapping'])
                self.assertEqual(report['input_profile']['profile'], c.profiles.FWHT_SCALED_PROFILE)
                np.testing.assert_array_equal(expected, c.r.reference(original['descriptor']['semantics'], source))
                with np.load(report['source_artifact']['path']) as arrays:
                    self.assertEqual(arrays['storage'].dtype, np.dtype('float16'))
                    np.testing.assert_array_equal(arrays['storage'], source)
                from semantic_dao_baseline import support
                contract = support(original['descriptor']['semantics'], inventory={
                    'available': True, 'contract_verified': True})
                self.assertEqual(contract['supported'], compute == 1)
                if compute == 0:
                    self.assertIn('cannot be substituted', contract['reason'])

    def test_prepare_rejects_unsupported_operator_before_output(self):
        with tempfile.TemporaryDirectory() as temp:
            args, _ = self.fixture(Path(temp), operator=3)
            with self.assertRaisesRegex(ValueError, 'only C2C FFT or standard FWHT'):
                c.prepare(args)
            self.assertFalse(args.output.exists())

    def test_fwht_prepare_rejects_scale_contract_gap_before_output(self):
        with tempfile.TemporaryDirectory() as temp:
            args, _ = self.fixture(Path(temp), operator=2, storage=0, fft_output_scale=0.5)
            with self.assertRaisesRegex(ValueError, 'FWHT input controls require fft_output_scale=1'):
                c.prepare(args)
            self.assertFalse(args.output.exists())

    def test_original_measurement_identity_and_row_index_are_required(self):
        for field, value, message in (('library_sha256', 'different', 'library identity'),
                                      ('gpu_uuid', 'GPU-0eae5f07-33de-321b-70a9-ffeae49309e6', 'GPU identity'),
                                      ('row_checksum', 'different', 'row checksum'),
                                      ('rows_sha256', 'different', 'index checksum')):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temp:
                args, _ = self.fixture(Path(temp))
                path = Path(temp) / 'results.json'
                measurement = json.loads(path.read_text())
                if field == 'row_checksum':
                    measurement['rows'][0]['sha256'] = value
                    measurement['rows_sha256'] = c.r.digest(measurement['rows'])
                elif field == 'rows_sha256':
                    measurement[field] = value
                else:
                    measurement['identity'][field] = value
                path.write_text(json.dumps(measurement))
                with self.assertRaisesRegex(ValueError, message):
                    c.prepare(args)
                self.assertFalse(args.output.exists())

    def test_baseline_failures_preserve_own_status_and_fail_overall(self):
        for status in ('baseline-error', 'runtime-error', 'correctness-failure', 'resource-limit'):
            report = {'status': 'accuracy-review-required', 'baselines': [{'status': status}]}
            self.assertEqual(c.finalize_status(report), 1)
            self.assertEqual(report['cubutterfly_status'], 'accuracy-review-required')
            self.assertEqual(report['status'], 'incomplete-error')
        report = {'status': 'correctness-failure', 'baseline_stop': {'reason': 'unsafe'}}
        self.assertEqual(c.finalize_status(report), 1)
        self.assertEqual(report['cubutterfly_status'], 'correctness-failure')
        report = {'status': 'accuracy-review-required', 'baselines': [{'status': 'missing-baseline'}]}
        self.assertEqual(c.finalize_status(report), 0)
        report = {'status': 'accuracy-review-required', 'baselines': [
            {'status': 'missing-baseline', 'support_status': 'external-binding-error'}]}
        self.assertEqual(c.finalize_status(report), 1)
        self.assertEqual(report['cubutterfly_status'], 'accuracy-review-required')
        self.assertEqual(report['status'], 'incomplete-error')

    def test_cli_baseline_failure_returns_nonzero_and_persists_own_status(self):
        with tempfile.TemporaryDirectory() as temp:
            args, _ = self.fixture(Path(temp))
            def fail_baseline(_args, report, *_rest):
                report.update(status='accuracy-review-required', baselines=[{'status': 'baseline-error'}])
            argv = []
            for name in ('row', 'library', 'output', 'gpu_uuid', 'calibration_manifest'):
                argv.extend(['--' + name.replace('_', '-'), str(getattr(args, name))])
            with patch.object(c, 'execute_control', side_effect=fail_baseline):
                self.assertEqual(c.main(argv), 1)
            report = json.loads((args.output / 'results.json').read_text())
            self.assertEqual(report['status'], 'incomplete-error')
            self.assertEqual(report['cubutterfly_status'], 'accuracy-review-required')

    def test_profile_identity_binds_profile_and_source(self):
        metadata = {'profile': 'raw-stress', 'input_sha256': 'original'}
        first = c.input_profile_id('semantic', metadata)
        for field, value in (('profile', c.profiles.SCALED_PROFILE), ('input_sha256', 'changed')):
            altered = dict(metadata, **{field: value})
            self.assertNotEqual(first, c.input_profile_id('semantic', altered))

    def test_changed_source_evidence_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            args, original = self.fixture(Path(temp))
            Path(original['error_artifact']['path']).write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'artifact checksum'):
                c.prepare(args)
            self.assertFalse(args.output.exists())

    def test_row_checksum_and_semantic_id_are_verified(self):
        with tempfile.TemporaryDirectory() as temp:
            args, original = self.fixture(Path(temp))
            altered = copy.deepcopy(original)
            altered['id'] = 'incorrect'
            args.row.write_text(json.dumps({'row': altered, 'sha256': c.r.digest(altered)}))
            with self.assertRaisesRegex(ValueError, 'semantic ID'):
                c.prepare(args)
            args.row.write_text(json.dumps({'row': original, 'sha256': 'incorrect'}))
            with self.assertRaisesRegex(ValueError, 'row checksum'):
                c.prepare(args)

    def test_existing_output_is_rejected_without_writes(self):
        with tempfile.TemporaryDirectory() as temp:
            args, _ = self.fixture(Path(temp))
            args.output.mkdir()
            sentinel = args.output / 'keep'
            sentinel.write_text('original')
            with self.assertRaises(FileExistsError):
                c.prepare(args)
            self.assertEqual(sentinel.read_text(), 'original')

    def test_prepare_cli_never_imports_torch_or_checks_gpu(self):
        with tempfile.TemporaryDirectory() as temp:
            args, _ = self.fixture(Path(temp))
            code = """
import builtins, sys
sys.path.insert(0, sys.argv.pop(1))
old = builtins.__import__
def checked(name, *args, **kwargs):
    if name.split('.')[0] in ('torch', 'cubutterfly'):
        raise AssertionError('GPU module imported during preparation: ' + name)
    return old(name, *args, **kwargs)
builtins.__import__ = checked
import run_fft_range_control as c
c.r.exclusive_gpu = lambda uuid: (_ for _ in ()).throw(AssertionError('GPU guard called'))
raise SystemExit(c.main(sys.argv[1:]))
"""
            command = [sys.executable, '-c', code, str(ROOT / 'scripts'), '--prepare-only']
            for name in ('row', 'library', 'output', 'gpu_uuid', 'calibration_manifest'):
                command.extend(['--' + name.replace('_', '-'), str(getattr(args, name))])
            result = subprocess.run(command, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((args.output / 'results.json').is_file())

    def test_guard_checks_before_and_after_failure(self):
        with patch.object(c.r, 'exclusive_gpu') as guard:
            with self.assertRaisesRegex(RuntimeError, 'operation'):
                c.guarded('GPU-test', lambda: (_ for _ in ()).throw(RuntimeError('operation')))
            self.assertEqual(guard.call_count, 2)


if __name__ == '__main__':
    unittest.main()
