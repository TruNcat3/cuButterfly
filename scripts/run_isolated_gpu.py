#!/usr/bin/env python3
"""Run one bounded experiment under an exclusive CUDA-process guard.

Set XD_REPORT_DIR to a fresh attempt directory. Only a clean, successful attempt
becomes `release`; interrupted attempts stay quarantined. Never stop foreign jobs.
"""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def gpu_pids(uuid):
    raw = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                   '--format=csv,noheader,nounits'], text=True, timeout=15)
    return [int(row.split(',')[1]) for row in raw.splitlines()
            if row.split(',')[0].strip() == uuid]


def descendant(pid, ancestor):
    while pid > 1:
        if pid == ancestor:
            return True
        try:
            pid = int(Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            return False
    return False


def stop_owned(child):
    if child.poll() is not None:
        return
    if os.getpgid(child.pid) != child.pid:
        raise RuntimeError('refusing to signal an unowned process group')
    os.killpg(child.pid, signal.SIGTERM)
    try:
        child.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGKILL)
        child.wait(timeout=10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--report-dir', type=Path, required=True)
    parser.add_argument('--until', required=True, help='explicit ISO deadline with timezone')
    parser.add_argument('--idle-seconds', type=float, default=30)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error('a command is required after --')
    cutoff = datetime.fromisoformat(args.until)
    if cutoff.tzinfo is None:
        parser.error('--until requires a timezone')
    root = args.report_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if (root/'release').exists():
        raise RuntimeError('release already exists; use a new report directory')

    def status(state, **details):
        record = dict(state=state, gpu_uuid=args.gpu_uuid, command=command,
                      updated=datetime.now().astimezone().isoformat(), deadline=args.until, **details)
        temporary = root/'status.tmp'
        temporary.write_text(json.dumps(record, indent=2)+'\n')
        temporary.replace(root/'status.json')
        print(json.dumps(record), flush=True)

    idle_since = None
    previous = None
    while time.time() < cutoff.timestamp():
        pids = gpu_pids(args.gpu_uuid)
        if pids:
            idle_since = None
            if pids != previous:
                status('waiting', external_pids=pids)
            previous = pids
            time.sleep(15)
            continue
        previous = None
        idle_since = idle_since or time.monotonic()
        if time.monotonic() - idle_since < args.idle_seconds:
            time.sleep(1)
            continue
        attempt = root/'attempts'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        attempt.mkdir(parents=True)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu_uuid, XD_REPORT_DIR=str(attempt))
        status('running', attempt=str(attempt))
        with (attempt/'runner.log').open('w') as log:
            child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True)
            conflict = []
            try:
                while child.poll() is None and time.time() < cutoff.timestamp():
                    conflict = [pid for pid in gpu_pids(args.gpu_uuid)
                                if Path(f'/proc/{pid}').exists() and not descendant(pid, child.pid)]
                    if conflict:
                        break
                    time.sleep(1)
                # Check once more at exit; a conflict invalidates the whole attempt.
                conflict += [pid for pid in gpu_pids(args.gpu_uuid)
                             if Path(f'/proc/{pid}').exists() and not descendant(pid, child.pid)]
                if conflict or time.time() >= cutoff.timestamp():
                    stop_owned(child)
                    status('interrupted', attempt=str(attempt), external_pids=sorted(set(conflict)))
                    idle_since = None
                    continue
            except BaseException:
                stop_owned(child)
                raise
        if child.returncode:
            status('failed', attempt=str(attempt), returncode=child.returncode)
            return child.returncode
        attempt.rename(root/'release')
        status('complete', release=str(root/'release'))
        return 0
    status('expired')
    return 2


if __name__ == '__main__':
    raise SystemExit(main())
