#!/usr/bin/env python3
"""Resume the bounded NTT cohort after an existing trial-1 process exits."""
from __future__ import annotations

import json
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / 'results/paper_targeted_20261012/ntt_extension'
STATE = WORK / 'supervisor_state.json'
WRAPPER = ROOT / 'paper/tools/ntt_extension_20261012.py'
PYTHON = '/home/wt/yes/envs/cubutterfly/bin/python'


def write(value):
    temporary = STATE.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temporary.replace(STATE)


def load():
    return json.loads(STATE.read_text()) if STATE.exists() else {
        'schema': 'cubutterfly-ntt-extension-supervisor-v1', 'status': 'starting',
        'started_unix': time.time(), 'steps': [], 'gpu_uuid': 'GPU-0eae5f07-33de-321b-70a9-ffeae49309e6'}


def run_step(state, number):
    report_path = WORK / f'trial-{number}/results.json'
    if report_path.exists():
        report = json.loads(report_path.read_text())
        if report.get('status') == 'complete' and not report.get('remaining_case_ids'):
            return 0
        state.update(status='stopped-on-incomplete-existing-trial', error=f'trial-{number} requires explicit recovery')
        write(state)
        return 1
    log = WORK / f'supervisor-trial-{number}.log'
    state['status'] = f'trial-{number}-running'
    state['steps'].append({'trial': number, 'status': 'starting', 'log': str(log), 'started_unix': time.time()})
    step = state['steps'][-1]
    write(state)
    with log.open('ab') as stream:
        process = subprocess.Popen([PYTHON, str(WRAPPER), 'trial', '--number', str(number)],
                                   cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        step['pid'] = process.pid
        write(state)
        code = process.wait()
    step.update(status='complete' if code == 0 else 'failed', returncode=code, finished_unix=time.time())
    state['status'] = step['status']
    write(state)
    return code


def main():
    lock = (WORK / 'supervisor.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = load()
    state['supervisor_pid'] = os.getpid()
    write(state)
    trial1 = WORK / 'trial-1/results.json'
    while True:
        report = json.loads(trial1.read_text()) if trial1.exists() else {}
        if report.get('status') == 'complete' and not report.get('remaining_case_ids'):
            break
        if report.get('status', '').startswith('stopped') or report.get('status') in ('runner-error', 'interrupted'):
            state.update(status='stopped-on-trial-1-failure', trial1_status=report['status'])
            write(state)
            return 1
        state['status'] = 'waiting-trial-1'
        state['trial1_rows'] = len(report.get('rows', []))
        state['trial1_remaining'] = len(report.get('remaining_case_ids', []))
        state['updated_unix'] = time.time()
        write(state)
        time.sleep(30)
    state['trial1_complete_unix'] = time.time()
    state['trial1_rows'] = len(report['rows'])
    state['trial1_remaining'] = 0
    write(state)
    for number in (2, 3):
        if run_step(state, number) != 0:
            state['status'] = 'stopped-on-failure'
            write(state)
            return 1
    state['status'] = 'summarizing'
    write(state)
    log = WORK / 'supervisor-summary.log'
    with log.open('ab') as stream:
        process = subprocess.run([PYTHON, str(ROOT / 'paper/tools/summarize_ntt_extension_20261012.py')],
                                 cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
    state['summary_returncode'] = process.returncode
    state['status'] = 'complete' if process.returncode == 0 else 'summary-failed'
    state['finished_unix'] = time.time()
    write(state)
    return process.returncode


if __name__ == '__main__':
    raise SystemExit(main())
