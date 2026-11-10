#!/usr/bin/env python3
"""Resume a finite, checkpointed campaign whenever its target GPU is idle.

The child driver must guard foreign GPU work, clean up its own descendants on
SIGTERM, and write status.json with explicit retryable interruption states.
This watcher never adopts a build or retries a numerical/compilation failure.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import xml.etree.ElementTree as ET


RETRYABLE = {'waiting-for-gpu', 'interrupted-by-foreign-gpu-work',
             'interrupted-calibration'}


def parse_gpu_snapshot(raw, gpu_uuid):
    devices = [gpu for gpu in ET.fromstring(raw).findall('gpu')
               if gpu.findtext('uuid') == gpu_uuid]
    if len(devices) != 1:
        raise ValueError('target GPU is absent or ambiguous')
    gpu = devices[0]
    processes = gpu.find('processes')
    if processes is None or (processes.text or '').strip() not in ('', 'None'):
        raise ValueError('GPU process observation is unavailable')

    def number(path, unit):
        text = (gpu.findtext(path) or '').split()
        if len(text) != 2 or text[1] != unit:
            raise ValueError('GPU metric unavailable: ' + path)
        value = float(text[0])
        if not 0 <= value < float('inf'):
            raise ValueError('invalid GPU metric: ' + path)
        return value

    process_info = [dict(pid=int(p.findtext('pid')), type=p.findtext('type'),
                         name=p.findtext('process_name'), memory=p.findtext('used_memory'))
                    for p in processes.findall('process_info')]
    return dict(gpu_uuid=gpu_uuid, name=gpu.findtext('product_name'),
                memory_total_mib=number('fb_memory_usage/total', 'MiB'),
                memory_used_mib=number('fb_memory_usage/used', 'MiB'),
                gpu_utilization=number('utilization/gpu_util', '%'),
                memory_utilization=number('utilization/memory_util', '%'),
                pids=[p['pid'] for p in process_info], processes=process_info)


def gpu_snapshot(gpu_uuid):
    raw = subprocess.check_output(['nvidia-smi', '-i', gpu_uuid, '-q', '-x'],
                                  text=True, stderr=subprocess.STDOUT, timeout=15)
    return parse_gpu_snapshot(raw, gpu_uuid)


def is_idle(snapshot, max_idle_memory_mib, idle_graphics_names=()):
    permitted = {p['pid'] for p in snapshot.get('processes', [])
                 if p['type'] == 'G' and p['name'] in idle_graphics_names}
    return (not (set(snapshot['pids']) - permitted) and snapshot['gpu_utilization'] == 0
            and snapshot['memory_utilization'] == 0
            and snapshot['memory_used_mib'] <= max_idle_memory_mib)


def read_campaign(root):
    path = root / 'status.json'
    return json.loads(path.read_text()) if path.exists() else {}


def complete(state):
    return str(state.get('status', '')).startswith('complete') and state.get('remaining') == 0


def exit_action(code, campaign, started_unix):
    if code == 73:  # Another manually started driver owns the lock.
        return 'wait-for-driver'
    if code == 0 and complete(campaign):
        return 'finished'
    if (code != 0 and campaign.get('retryable') is True
            and campaign.get('status') in RETRYABLE
            and campaign.get('updated_unix', 0) >= started_unix):
        return 'retry'
    return 'failed'


def driver_locked(root):
    with (root / 'driver.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        return False


def stop_driver(child):
    # The driver's handler tears down separately owned measurement/producer groups.
    # Do not kill unrelated GPU processes or force-kill before this cleanup finishes.
    if child is not None and child.poll() is None:
        child.terminate()
        try:
            child.wait(timeout=45)
        except subprocess.TimeoutExpired as error:
            raise RuntimeError('driver cleanup did not finish; manual inspection required') from error


def watch(root, gpu_uuid, command, poll_seconds=10, idle_seconds=30,
          max_idle_memory_mib=64, idle_graphics_names=(), cpu_preparation_status=None):
    root = Path(root).resolve(strict=True)
    with (root / 'monitor.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state_path = root / 'monitor_status.json'
        state = dict(schema='cubutterfly-idle-campaign-monitor-v1',
                     monitor_pid=os.getpid(), started_unix=time.time(),
                     campaign_root=str(root), gpu_uuid=gpu_uuid, command=command,
                     poll_seconds=poll_seconds, idle_seconds=idle_seconds,
                     max_idle_memory_mib=max_idle_memory_mib, attempts=0,
                     idle_graphics_names=list(idle_graphics_names),
                     cpu_preparation_status=str(cpu_preparation_status) if cpu_preparation_status else None,
                     policy='resume until finite campaign completes; only GPU interruptions retry')
        previous_transition = None

        def record(status, **updates):
            nonlocal previous_transition
            state.update(updates, status=status, updated_unix=time.time())
            temporary = state_path.with_suffix('.tmp')
            temporary.write_text(json.dumps(state, indent=2) + '\n')
            temporary.replace(state_path)
            transition = (status, state['attempts'])
            if transition != previous_transition:
                summary = {key: state[key] for key in
                           ('status', 'updated_unix', 'attempts', 'monitor_pid')}
                summary.update(child_pid=state.get('child_pid'), error=state.get('error'))
                with (root / 'monitor_events.jsonl').open('a') as events:
                    events.write(json.dumps(summary) + '\n')
                print(json.dumps(summary), flush=True)
                previous_transition = transition

        received = []
        handlers = {sig: signal.signal(sig, lambda number, frame: received.append(number))
                    for sig in (signal.SIGTERM, signal.SIGINT)}
        child = None
        idle_since = None
        try:
            record('starting')
            while not received:
                campaign = read_campaign(root)
                state['campaign'] = {key: campaign.get(key) for key in
                                     ('status', 'phase', 'measured_rows', 'remaining', 'error')}
                preparation = None
                if child is None and cpu_preparation_status is not None:
                    path = Path(cpu_preparation_status)
                    preparation = json.loads(path.read_text()) if path.exists() else {}
                    state['cpu_preparation'] = {key: preparation.get(key) for key in
                                               ('status', 'scoring_complete', 'compile_ready', 'updated_unix', 'error')}
                if child is not None:
                    code = child.poll()
                    if code is None:
                        record('running', child_pid=child.pid)
                    else:
                        action = exit_action(code, campaign, state['attempt_started_unix'])
                        record(action, returncode=code, child_pid=None)
                        child = None
                        idle_since = None
                        if action in ('finished', 'failed'):
                            return 0 if action == 'finished' else (code or 2)
                elif driver_locked(root):
                    idle_since = None
                    record('waiting-for-existing-driver')
                elif complete(campaign):
                    record('finished', child_pid=None)
                    return 0
                elif (campaign.get('status') == 'failed'
                      or str(campaign.get('status', '')).startswith('stopped-')):
                    record('failed', error='existing campaign failure requires inspection')
                    return 2
                elif preparation is not None and (
                        not preparation.get('scoring_complete') or
                        not preparation.get('compile_ready')):
                    idle_since = None
                    if preparation.get('status') == 'failed':
                        record('failed', error='CPU preparation failed; see its status file')
                        return 2
                    record('waiting-for-cpu-preparation')
                else:
                    try:
                        snapshot = gpu_snapshot(gpu_uuid)
                    except (OSError, ValueError, TypeError, ET.ParseError,
                            subprocess.SubprocessError) as error:
                        idle_since = None
                        record('waiting-for-gpu-query', error=str(error), idle_for_seconds=0)
                    else:
                        state.pop('error', None)
                        state['gpu'] = snapshot
                        if not is_idle(snapshot, max_idle_memory_mib, idle_graphics_names):
                            idle_since = None
                            record('waiting-for-idle', idle_for_seconds=0)
                        else:
                            now = time.monotonic()
                            if idle_since is None:
                                idle_since = now
                            elapsed = now - idle_since
                            record('confirming-idle', idle_for_seconds=elapsed)
                            if elapsed >= idle_seconds:
                                state['attempts'] += 1
                                state['attempt_started_unix'] = time.time()
                                with (root / 'monitor_driver.log').open('a') as log:
                                    child = subprocess.Popen(command, cwd=root,
                                        env=dict(os.environ, CUDA_VISIBLE_DEVICES=gpu_uuid),
                                        stdin=subprocess.DEVNULL, stdout=log,
                                        stderr=subprocess.STDOUT, start_new_session=True)
                                record('running', child_pid=child.pid)
                # A received signal wakes this short tick without waiting a full poll.
                until = time.monotonic() + poll_seconds
                while not received and time.monotonic() < until:
                    time.sleep(min(1, max(0, until - time.monotonic())))
            record('stopping', received_signal=received[0])
            stop_driver(child)
            record('stopped', child_pid=None)
            return 128 + received[0]
        except BaseException as error:
            record('monitor-error', error=repr(error))
            stop_driver(child)
            raise
        finally:
            for sig, handler in handlers.items():
                signal.signal(sig, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign-dir', type=Path, required=True)
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--poll-seconds', type=float, default=10)
    parser.add_argument('--idle-seconds', type=float, default=30)
    parser.add_argument('--max-idle-memory-mib', type=float, default=64,
                        help='Allow only the idle driver footprint, not active allocations')
    parser.add_argument('--idle-graphics-name', action='append', default=[],
                        help='Explicit idle display process name from nvidia-smi; only G type, zero utilization and memory limit still required')
    parser.add_argument('--cpu-preparation-status', type=Path,
                        help='Wait for CPU scoring_complete and compile_ready before checking GPU readiness')
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if (not command or not 0 < args.poll_seconds <= 60 or not 0 <= args.idle_seconds < float('inf')
            or not 0 <= args.max_idle_memory_mib < float('inf')):
        parser.error('command, positive poll <=60s, and finite nonnegative idle/memory limits required')
    code = watch(args.campaign_dir, args.gpu_uuid, command, args.poll_seconds,
                 args.idle_seconds, args.max_idle_memory_mib, args.idle_graphics_name,
                 args.cpu_preparation_status)
    return code if code >= 0 else 128 - code


if __name__ == '__main__':
    raise SystemExit(main())
