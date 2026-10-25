import csv
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import os
import json

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('ncu_wrapper',ROOT/'paper/tools/run_bottleneck_profiles_20261012.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class Tests(unittest.TestCase):
    def record(self):
        return dict(label='fft-linear',expected_kernel_groups=2,argv=['ncu','--metrics','duration,bytes','--log-file','/old/global.csv'])

    def fixture(self,path,ids=('0','1'),missing=False,value='1,234'):
        with path.open('w',newline='') as stream:
            stream.write('==PROF== Profiling preamble\n')
            writer=csv.writer(stream)
            writer.writerow(['ID','Kernel Name','Metric Name','Metric Unit','Metric Value'])
            for index in ids:
                for metric in ['duration'] if missing else ['duration','bytes']:
                    writer.writerow([index,'shared_iterative_kernel<float>',metric,'byte',value])

    def test_two_stage_capture_and_wide_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'capture.csv';self.fixture(path)
            rows=module.parse_metrics(path,self.record())
            self.assertEqual(len(rows),2)
            self.assertEqual(rows[0]['metrics']['bytes']['value'],1234)
            output=Path(directory)/'summary.csv';module.write_csv(output,rows)
            with output.open() as stream: result=list(csv.DictReader(stream))
            self.assertEqual(len(result),2)
            self.assertEqual(result[1]['bytes__unit'],'byte')
            self.assertIn('not normal timing',result[0]['timing_kind'])

    def test_ncu_native_wide_header(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'native.csv'
            path.write_text(
                'ID,Kernel Name,gpu__time_duration.sum,dram__bytes_read.sum\n'
                '0,shared_iterative_kernel,12,100\n'
                '1,shared_iterative_kernel,14,200\n')
            record=self.record()
            record['argv'][record['argv'].index('--metrics')+1] = \
                'gpu__time_duration.sum,dram__bytes_read.sum'
            rows=module.parse_metrics(path,record)
            self.assertEqual([row['metrics']['gpu__time_duration.sum']['value']
                              for row in rows],[12.0,14.0])

    def test_missing_or_invalid_capture_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'capture.csv'
            for kwargs in [dict(ids=()),dict(ids=('0',)),dict(ids=('0','1','2')),dict(missing=True),dict(value='N/A'),dict(value='nan')]:
                self.fixture(path,**kwargs)
                with self.assertRaises(ValueError): module.parse_metrics(path,self.record())

    def test_log_path_rewritten_without_mutating_prepared_command(self):
        record=self.record();argv=module.command_for(record,Path('/guard/attempt'))
        self.assertEqual(argv[-1],'/guard/attempt/fft-linear.ncu.csv')
        self.assertEqual(record['argv'][-1],'/old/global.csv')

    def test_hash_mismatch_fails_before_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'source';path.write_text('changed')
            with self.assertRaises(ValueError):module.verify(dict(hashes={str(path):'bad'},records=[]))

    def test_failed_capture_keeps_failed_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory);spec=path/'spec.json'
            record=self.record();record['environment']={}
            spec.write_text(json.dumps(dict(records=[record],claim_boundary='not normal timing')))
            with patch.object(module,'verify'),patch.dict(os.environ,XD_REPORT_DIR=str(path),CUDA_VISIBLE_DEVICES=module.GPU), \
                 patch.object(module.subprocess,'run',return_value=type('Result',(),{'returncode':0})()):
                self.assertEqual(module.child(spec),1)
            summary=json.loads((path/'profile_summary.json').read_text())
            self.assertEqual(summary['status'],'failed')
            self.assertEqual(summary['profiles'][0]['status'],'failed')
            self.assertTrue((path/'fft-linear.stdout.log').exists())


if __name__=='__main__': unittest.main()
