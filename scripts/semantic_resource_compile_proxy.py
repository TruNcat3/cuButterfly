#!/usr/bin/env python3
"""Guard the GPU parent, brokering compilation and forwarding CPU scripts."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import semantic_compile_broker as broker
from semantic_worker_resources import exceeded, limits, snapshot


def open_event_log(events):
    path = Path(events)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.open('a')


def run_cpu_helper(arguments):
    events = os.environ.get('CUBUTTERFLY_PREPARE_EVENTS')
    if not events:
        return subprocess.run([sys.executable, *arguments], check=False).returncode
    received = []
    previous = {}
    child = None
    started = time.time()
    identity = dict(worker_pid=os.getppid(), proxy_pid=os.getpid(),
                    script=str(Path(arguments[0]).resolve()), started_unix=started,
                    attempt_token=os.environ.get('CUBUTTERFLY_HELPER_ATTEMPT'))

    def cancelled(signum, frame):
        received.append(dict(signal=signum, received_unix=time.time()))
        # The owner signals the shared process group. Forwarding here delivers
        # a second signal and can turn a helper's ordinary error into a kill.

    # Establish the log before spawning: on a fresh campaign the ranking helper
    # can precede the compiler broker that used to create this directory.
    with open_event_log(events) as target:
        def record(kind, **fields):
            target.write(json.dumps(dict(identity, kind=kind, **fields)) + '\n')
            target.flush()

        try:
            for sig in (signal.SIGINT, signal.SIGTERM):
                previous[sig] = signal.signal(sig, cancelled)
            child = subprocess.Popen([sys.executable, *arguments])
            identity['helper_pid'] = child.pid
            record('cpu_helper_started')
            returncode = child.wait()
            record('cpu_helper_finished', ended_unix=time.time(), returncode=returncode,
                   termination_signal=-returncode if returncode < 0 else None,
                   received_signals=received)
            return returncode
        finally:
            # A failed receipt write must not leave a child using request files
            # after C++ has unwound and removed its temporary workspace.
            try:
                if child is not None and child.poll() is None:
                    child.kill()
                    child.wait()
            finally:
                for sig, handler in previous.items():
                    signal.signal(sig, handler)


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    budget = limits(max_vmas=int(os.environ.get('CUBUTTERFLY_WORKER_MAX_VMAS', '45000')),
                    max_rss_bytes=int(os.environ.get('CUBUTTERFLY_WORKER_MAX_RSS_BYTES', str(8 * 1024**3))))
    sample = snapshot(os.getppid())
    reason = exceeded(sample, budget)
    if reason:
        events = os.environ.get('CUBUTTERFLY_PREPARE_EVENTS')
        if not events:
            raise RuntimeError('resource guard requires preparation event evidence')
        with open_event_log(events) as target:
            target.write(json.dumps({'kind': 'worker_resource_limit', 'reason': reason,
                                     'resources': sample}) + '\n')
        print('worker resource limit: ' + reason, file=sys.stderr)
        return 75
    # JIT_PYTHON is also the interpreter binding for ordinary CPU helpers.
    # Keep compiler requests in the broker; it validates their exact root/path.
    if (arguments and not arguments[0].startswith('-')
            and Path(arguments[0]).name != 'compile_module.py'):
        return run_cpu_helper(arguments)
    return broker.main(arguments)


if __name__ == '__main__':
    raise SystemExit(main())
