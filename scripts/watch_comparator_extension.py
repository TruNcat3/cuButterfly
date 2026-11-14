#!/usr/bin/env python3
"""Bounded comparator watcher. Deadline terminates only the owned process group.

Existing rows are replay-validated by run_comparator_extension before completion;
only exit 76 or an observed foreign-process interruption is retried.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from watch_idle_campaign import gpu_snapshot, is_idle
import summarize_comparator_extension

GOOD = {'pass', 'accuracy-review-required'}
GAPS = {'unsupported', 'unsupported-semantic', 'missing-lowering', 'runner-gap',
        'validation-unavailable', 'not-applicable', 'dependency-unavailable'}


def deadline_value(value):
    stamp = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if stamp.utcoffset() is None:
        raise ValueError('--until requires an explicit timezone')
    return stamp.timestamp()


def row_error(row):
    """Fail closed on execution errors and any uncalibrated selection evidence."""
    def selections(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key == 'selection_reason':
                    reason = str(item).lower().replace('_', '-').replace(' ', '-')
                    if ('hardware-rate-prior' in reason
                            or ('fingerprint' in reason and any(word in reason for word in
                                ('missing', 'incompatible', 'mismatch', 'not-found', 'unavailable')))
                            or ('unmeasured' in reason and 'fallback' in reason)):
                        return 'uncalibrated selection: ' + str(item)
                error = selections(item)
                if error:
                    return error
        elif isinstance(value, list):
            for item in value:
                error = selections(item)
                if error:
                    return error
        return None
    error = selections(row)
    if error:
        return error
    for index, item in enumerate([row] + row.get('baselines', [])):
        status = item.get('status')
        allowed = GOOD | GAPS | ({'missing-baseline'} if index else set())
        if status not in allowed:
            return 'unacceptable row status: ' + str(status)
    return None


def owned_pid(pid, group):
    try:
        return os.getpgid(int(pid)) == group
    except ProcessLookupError:
        return False


def stop_child(child):
    if child is None:
        return
    # start_new_session makes this group exclusively ours, including descendants.
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        child.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(child.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    child.wait(timeout=5)


def watch(campaigns, until, state_dir, poll_seconds=15, idle_seconds=30):
    if not math.isfinite(poll_seconds) or not 0 < poll_seconds <= 60:
        raise ValueError('poll seconds must be in (0, 60]')
    if not math.isfinite(idle_seconds) or idle_seconds < 0:
        raise ValueError('idle seconds must be finite and nonnegative')
    deadline = deadline_value(until)
    campaigns = list(dict.fromkeys(Path(c).resolve(strict=True) for c in campaigns))
    manifests = [json.loads((c / 'preparation.json').read_text()) for c in campaigns]
    uuids = {m['config']['gpu_uuid'] for m in manifests}
    if len(uuids) != 1:
        raise ValueError('campaigns must target the same GPU UUID')
    gpu_uuid = uuids.pop()
    items = [(c, r['id']) for c, m in zip(campaigns, manifests) for r in m['cases']]
    if not items:
        raise ValueError('no campaign cases')
    state_dir = Path(state_dir).resolve()
    state_dir.mkdir(parents=True, exist_ok=True)
    with (state_dir / 'watcher.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = dict(gpu_uuid=gpu_uuid, campaigns=list(map(str, campaigns)), until=until,
                     deadline_unix=deadline, pid=os.getpid(), attempts=0,
                     cpu_preparation_status='ready', policy='deadline stops owned group; interrupted rows remain pending')
        previous = None
        def record(status, **updates):
            nonlocal previous
            completed = sum((c / 'rows' / (i + '.json')).exists() for c, i in items)
            state.update(status=status, updated_unix=time.time(), total=len(items),
                         completed=completed, remaining=len(items)-completed, **updates)
            temp = state_dir / 'status.json.tmp'
            temp.write_text(json.dumps(state, indent=2) + '\n')
            temp.replace(state_dir / 'status.json')
            event = (status, state['attempts'], updates.get('error'))
            if event != previous:
                line = json.dumps({k: state.get(k) for k in ('status', 'updated_unix', 'completed', 'remaining', 'attempts', 'error')})
                with (state_dir / 'events.jsonl').open('a') as stream:
                    stream.write(line + '\n')
                print(line, flush=True)
                previous = event
        validated = set()
        child = None
        current = None
        idle_since = None
        received = []
        handlers = {s: signal.signal(s, lambda n, f: received.append(n)) for s in (signal.SIGTERM, signal.SIGINT)}
        try:
            record('cpu-preparation-ready')
            while True:
                # Inspect raw rows even when exit code was zero, and on restart.
                for c, i in items:
                    path = c / 'rows' / (i + '.json')
                    if path.exists():
                        error = row_error(json.loads(path.read_text())['row'])
                        if error:
                            record('failed', error=error, case=i, campaign=str(c))
                            return 2
                if received:
                    record('stopped', received_signal=received[0])
                    return 128 + received[0]
                if time.time() >= deadline:
                    record('deadline', child_pid=None)
                    return 0
                if child is not None:
                    code = child.poll()
                    if code is not None:
                        child = None
                        idle_since = None
                        if code == 76:
                            record('waiting', reason='runner-busy', child_pid=None)
                        elif code != 0:
                            record('failed', error='runner exit ' + str(code), child_pid=None)
                            return code or 2
                        elif not (current[0] / 'rows' / (current[1] + '.json')).exists():
                            record('failed', error='runner produced no row', child_pid=None)
                            return 2
                        else:
                            try:
                                summarize_comparator_extension.write_outputs(current[0], current[0] / 'summary')
                            except Exception as error:
                                record('failed', error='summary refresh failed: ' + repr(error), child_pid=None)
                                return 2
                            validated.add(current[0])
                        continue
                    snapshot = gpu_snapshot(gpu_uuid)
                    if any(not owned_pid(pid, child.pid) for pid in snapshot['pids']):
                        stop_child(child)
                        child = None
                        idle_since = None
                        record('waiting', reason='foreign-process-interruption', child_pid=None)
                    else:
                        record('running', child_pid=child.pid)
                else:
                    pending = [(c, i) for c, i in items if not (c / 'rows' / (i + '.json')).exists()]
                    # One runner invocation validates all existing envelopes per campaign.
                    current = pending[0] if pending else next(((c, i) for c, i in items if c not in validated), None)
                    if current is None:
                        record('complete', child_pid=None)
                        return 0
                    snapshot = gpu_snapshot(gpu_uuid)
                    if not is_idle(snapshot, 64):
                        idle_since = None
                        record('waiting', reason='gpu-busy')
                    else:
                        if idle_since is None:
                            idle_since = time.monotonic()
                        record('waiting', reason='confirming-idle')
                        if time.monotonic() - idle_since >= idle_seconds and time.time() < deadline:
                            c, i = current
                            command = [sys.executable, str(Path(__file__).with_name('run_comparator_extension.py')), '--run', '--output', str(c), '--case', i]
                            with (state_dir / 'runner.log').open('a') as log:
                                child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                            state['attempts'] += 1
                            record('running', child_pid=child.pid, case=i, campaign=str(c))
                remaining = min(poll_seconds, max(0, deadline-time.time()))
                end = time.monotonic() + remaining
                while not received and time.monotonic() < end:
                    time.sleep(min(1, end-time.monotonic()))
        except Exception as error:
            record('failed', error=repr(error))
            return 2
        finally:
            stop_child(child)
            for sig, handler in handlers.items():
                signal.signal(sig, handler)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', action='append', type=Path, required=True)
    parser.add_argument('--until', required=True)
    parser.add_argument('--state-dir', type=Path, required=True)
    parser.add_argument('--poll-seconds', type=float, default=15)
    parser.add_argument('--idle-seconds', type=float, default=30)
    args = parser.parse_args(argv)
    return watch(args.campaign, args.until, args.state_dir, args.poll_seconds, args.idle_seconds)


if __name__ == '__main__':
    raise SystemExit(main())
