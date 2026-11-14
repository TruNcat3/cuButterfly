#!/usr/bin/env python3
"""Finish native controls, paired calibration and public AUTO validation.

CPU inspection is the default. --execute runs the finite prepared population;
--wait-native-pid can join an already running native calibration on this host.
"""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('prepared', 'build', 'library', 'native-output', 'public-output', 'profile'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--wait-native-pid', type=int)
    args = parser.parse_args()
    native_profile = args.native_output / 'profile.json'
    native_base = [sys.executable, str(ROOT / 'calibrate_hadacore_native.py'),
                   '--prepared', str(args.prepared), '--output', str(args.native_output),
                   '--library', str(args.library)]
    phases = [
        ('native-calibration', native_base + ['--execute']),
        ('native-export', native_base + ['--export-only', '--profile', str(native_profile)]),
        ('paired-export', [sys.executable, str(ROOT / 'calibrate_hadacore_tile.py'),
                          '--export-only', '--output', str(args.prepared), '--build', str(args.build),
                          '--native-profile', str(native_profile), '--profile', str(args.profile)]),
        ('public-validation', [sys.executable, str(ROOT / 'validate_hadacore_public.py'),
                               '--execute', '--prepared', str(args.prepared),
                               '--output', str(args.public_output), '--library', str(args.library),
                               '--profile', str(args.profile)]),
    ]
    state = {'schema': 'hadacore-completion-v1', 'status': 'prepared',
             'phases': [{'name': name, 'command': cmd} for name, cmd in phases]}
    if not args.execute:
        print(json.dumps(state, indent=2))
        return 0
    checkpoint = args.public_output / 'completion.json'
    if args.wait_native_pid:
        proc = Path('/proc') / str(args.wait_native_pid)
        if proc.exists():
            command = (proc / 'cmdline').read_bytes()
            if (b'calibrate_hadacore_native.py' not in command or
                    str(args.native_output).encode() not in command):
                raise ValueError('wait PID does not own this native calibration')
            start = (proc / 'stat').read_text().rsplit(')', 1)[1].split()[19]
            while proc.exists():
                try:
                    current = (proc / 'stat').read_text().rsplit(')', 1)[1].split()
                except FileNotFoundError:
                    break
                if current[19] != start or current[0] == 'Z':
                    break
                state.update(status='waiting-for-native', waited_pid=args.wait_native_pid,
                             updated_unix=time.time())
                write(checkpoint, state)
                time.sleep(30)
        previous = json.loads((args.native_output / 'native-checkpoint.json').read_text())
        if previous.get('status') != 'complete':
            state.update(status='stopped', reason='existing native calibration did not finish',
                         native_status=previous)
            write(checkpoint, state)
            return 1
        phases = phases[1:]
    for name, command in phases:
        state.update(status='running', phase=name, updated_unix=time.time())
        write(checkpoint, state)
        log = args.public_output / (name + '.log')
        with log.open('ab') as stream:
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            state.update(status='stopped', phase=name, exit_code=result.returncode, log=str(log))
            write(checkpoint, state)
            return 1
    public = json.loads((args.public_output / 'checkpoint.json').read_text())
    state.update(status='complete' if public.get('status') == 'complete' else 'stopped',
                 public_result=public, updated_unix=time.time())
    write(checkpoint, state)
    return 0 if state['status'] == 'complete' else 1


if __name__ == '__main__':
    sys.exit(main())
