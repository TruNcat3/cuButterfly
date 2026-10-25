#!/usr/bin/env python3
"""Guard the GPU parent before forwarding unchanged CPU broker queries."""
import json
import os
import sys

import semantic_compile_broker as broker
from semantic_worker_resources import exceeded, limits, snapshot


def main(argv=None):
    budget = limits(max_vmas=int(os.environ.get('CUBUTTERFLY_WORKER_MAX_VMAS', '45000')),
                    max_rss_bytes=int(os.environ.get('CUBUTTERFLY_WORKER_MAX_RSS_BYTES', str(8 * 1024**3))))
    sample = snapshot(os.getppid())
    reason = exceeded(sample, budget)
    if reason:
        events = os.environ.get('CUBUTTERFLY_PREPARE_EVENTS')
        if not events:
            raise RuntimeError('resource guard requires preparation event evidence')
        with open(events, 'a') as target:
            target.write(json.dumps({'kind': 'worker_resource_limit', 'reason': reason,
                                     'resources': sample}) + '\n')
        print('worker resource limit: ' + reason, file=sys.stderr)
        return 75
    return broker.main(argv)


if __name__ == '__main__':
    raise SystemExit(main())
