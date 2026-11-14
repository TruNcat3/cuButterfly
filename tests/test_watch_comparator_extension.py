import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
import watch_comparator_extension as watcher


class Clock:
    now = 100.0
    def time(self):
        return self.now
    def sleep(self, seconds):
        self.now += max(seconds, .001)


class WatchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.campaign = self.root / 'campaign'
        self.campaign.mkdir()
        (self.campaign / 'preparation.json').write_text(json.dumps(dict(config=dict(gpu_uuid='GPU-one'), cases=[dict(id='a') , dict(id='b') ])))
        self.state = self.root / 'state'
        self.summary = unittest.mock.Mock()
        self.summary_patch = patch.object(watcher.summarize_comparator_extension, 'write_outputs', self.summary)
        self.summary_patch.start()
        self.addCleanup(self.summary_patch.stop)
    def tearDown(self):
        self.temp.cleanup()
    def row(self, case='a', **updates):
        directory = self.campaign / 'rows'
        directory.mkdir(exist_ok=True)
        (directory / (case + '.json')).write_text(json.dumps(dict(row=dict(dict(status='pass', baselines=[]), **updates))))
    def run_watch(self, snapshot, launch=None, seconds=5):
        clock = Clock()
        until = f'1970-01-01T00:01:{40+seconds:02d}+00:00'
        with patch.object(watcher.time, 'time', clock.time), patch.object(watcher.time, 'monotonic', clock.time), patch.object(watcher.time, 'sleep', clock.sleep), patch.object(watcher, 'gpu_snapshot', return_value=snapshot), patch.object(watcher.subprocess, 'Popen', side_effect=launch) as popen, patch.object(watcher, 'stop_child'):
            code = watcher.watch([self.campaign], until, self.state, poll_seconds=1, idle_seconds=0)
            return code, popen
    def test_timezone_required(self):
        with self.assertRaises(ValueError):
            watcher.deadline_value('2026-11-14T10:00:00')
    def test_busy_wait_is_bounded(self):
        code, popen = self.run_watch(dict(pids=[99], gpu_utilization=10, memory_utilization=0, memory_used_mib=10))
        self.assertEqual(code, 0)
        popen.assert_not_called()
        self.assertEqual(json.loads((self.state/'status.json').read_text())['status'], 'deadline')
    def test_success_and_resume_use_runner(self):
        self.row()
        def launch(command, **kwargs):
            self.assertEqual(command[0], sys.executable)
            self.assertTrue(kwargs['start_new_session'])
            self.row(command[-1])
            child = unittest.mock.Mock(pid=999)
            child.poll.return_value = 0
            return child
        code, popen = self.run_watch(dict(pids=[], gpu_utilization=0, memory_utilization=0, memory_used_mib=0), launch)
        self.assertEqual(code, 0)
        self.assertEqual(popen.call_count, 1)
        self.summary.assert_called_once_with(self.campaign, self.campaign/'summary')
        self.assertEqual(json.loads((self.state/'status.json').read_text())['status'], 'complete')
    def test_summary_failure_stops_and_retains_raw(self):
        self.summary.side_effect = ValueError('bad summary')
        def launch(command, **kwargs):
            self.row(command[-1])
            child = unittest.mock.Mock(pid=999)
            child.poll.return_value = 0
            return child
        code, popen = self.run_watch(dict(pids=[], gpu_utilization=0, memory_utilization=0, memory_used_mib=0), launch)
        self.assertEqual(code, 2)
        self.assertEqual(popen.call_count, 1)
        self.assertTrue((self.campaign/'rows/a.json').exists())
        self.assertIn('summary refresh failed', json.loads((self.state/'status.json').read_text())['error'])
    def test_uncalibrated_raw_row_stops_before_launch(self):
        self.row(selection_reason='hardware-rate-prior:fingerprint incompatible')
        code, popen = self.run_watch({})
        self.assertEqual(code, 2)
        popen.assert_not_called()
        self.assertTrue((self.campaign/'rows/a.json').exists())
    def test_row_statuses_fail_closed(self):
        for reason in ('hardware-rate-prior', 'fingerprint incompatible', 'missing runtime fingerprint', 'runtime fingerprint mismatch', 'unmeasured fallback'):
            self.assertIsNotNone(watcher.row_error(dict(status='pass', selection_reason=reason)))
        self.assertIsNone(watcher.row_error(dict(status='accuracy-review-required', baselines=[dict(status='unsupported')])))
        self.assertIsNotNone(watcher.row_error(dict(status='pass', baselines=[dict(status='baseline-error')])))
    def test_matching_fingerprint_and_calibrated_fallback_allowed(self):
        for reason in ('matching runtime fingerprint', 'calibrated measured fallback'):
            self.assertIsNone(watcher.row_error(dict(status='pass', selection_reason=reason)))
    def test_missing_baseline_advances_without_masking_own_status(self):
        self.assertIsNotNone(watcher.row_error(dict(status='missing-baseline')))
        self.row(baselines=[dict(status='missing-baseline')])
        def launch(command, **kwargs):
            self.row(command[-1], baselines=[dict(status='missing-baseline')])
            child = unittest.mock.Mock(pid=999)
            child.poll.return_value = 0
            return child
        code, popen = self.run_watch(dict(pids=[], gpu_utilization=0, memory_utilization=0, memory_used_mib=0), launch)
        self.assertEqual(code, 0)
        self.assertEqual(popen.call_count, 1)
        self.summary.assert_called_once_with(self.campaign, self.campaign/'summary')
        self.assertEqual(json.loads((self.state/'status.json').read_text())['status'], 'complete')
    def test_nonbusy_exit_stops(self):
        child = unittest.mock.Mock(pid=999)
        child.poll.return_value = 4
        code, popen = self.run_watch(dict(pids=[], gpu_utilization=0, memory_utilization=0, memory_used_mib=0), lambda *a, **k: child)
        self.assertEqual(code, 4)
        self.assertEqual(popen.call_count, 1)
        self.summary.assert_not_called()
    def test_busy_exit_retries_until_deadline(self):
        child = unittest.mock.Mock(pid=999)
        child.poll.return_value = 76
        code, popen = self.run_watch(dict(pids=[], gpu_utilization=0, memory_utilization=0, memory_used_mib=0), lambda *a, **k: child)
        self.assertEqual(code, 0)
        self.assertGreater(popen.call_count, 1)
        self.summary.assert_not_called()
        self.assertEqual(json.loads((self.state/'status.json').read_text())['status'], 'deadline')
    def test_foreign_process_stops_only_owned_child(self):
        clock = Clock()
        child = unittest.mock.Mock(pid=999)
        child.poll.return_value = None
        idle = dict(pids=[], gpu_utilization=0, memory_utilization=0, memory_used_mib=0)
        busy = dict(pids=[55], gpu_utilization=20, memory_utilization=0, memory_used_mib=10)
        with patch.object(watcher.time, 'time', clock.time), patch.object(watcher.time, 'monotonic', clock.time), patch.object(watcher.time, 'sleep', clock.sleep), patch.object(watcher, 'gpu_snapshot', side_effect=[idle, busy, busy, busy]), patch.object(watcher, 'owned_pid', return_value=False), patch.object(watcher.subprocess, 'Popen', return_value=child), patch.object(watcher, 'stop_child') as stop:
            code = watcher.watch([self.campaign], '1970-01-01T00:01:44+00:00', self.state, 1, 0)
            self.assertEqual(code, 0)
            stop.assert_any_call(child)
            self.assertFalse((self.campaign/'rows/a.json').exists())
    def test_uuid_mismatch(self):
        other = self.root/'other'
        other.mkdir()
        (other/'preparation.json').write_text(json.dumps(dict(config=dict(gpu_uuid='GPU-two'), cases=[])))
        with self.assertRaisesRegex(ValueError, 'same GPU'):
            watcher.watch([self.campaign, other], '2026-11-14T10:00:00+00:00', self.state)


if __name__ == '__main__':
    unittest.main()
