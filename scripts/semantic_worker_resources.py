"""Linux process resource accounting for bounded measurement workers."""
from pathlib import Path
import os


class WorkerRecycle(RuntimeError):
    def __init__(self, reason, snapshot=None):
        self.reason, self.snapshot = reason, snapshot
        super().__init__(reason)


def snapshot(pid=None, proc=Path('/proc')):
    pid = os.getpid() if pid is None else pid
    base = Path(proc) / str(pid)
    maps = (base / 'maps').read_text().splitlines()
    status = {}
    for line in (base / 'status').read_text().splitlines():
        key, _, value = line.partition(':')
        status[key] = value.strip()
    return {'pid': pid, 'vma_count': len(maps),
            'rss_bytes': int(status.get('VmRSS', '0 kB').split()[0]) * 1024,
            'virtual_bytes': int(status.get('VmSize', '0 kB').split()[0]) * 1024,
            'fd_count': sum(1 for _ in (base / 'fd').iterdir()),
            'module_mappings': sum('/module.so' in line for line in maps),
            'max_map_count': int((Path(proc) / 'sys/vm/max_map_count').read_text())}


def limits(max_vmas=45000, max_rss_bytes=8 * 1024**3, max_cases=128, max_attempts=64):
    if min(max_vmas, max_rss_bytes, max_cases, max_attempts) < 1:
        raise ValueError('worker resource limits must be positive')
    return dict(max_vmas=max_vmas, max_rss_bytes=max_rss_bytes,
                max_cases=max_cases, max_attempts=max_attempts)


def exceeded(sample, budget, completed=0, attempts=0):
    # Keep a quarter of the kernel limit available for a complete search attempt.
    vma_limit = min(budget['max_vmas'], sample['max_map_count'] * 3 // 4)
    for name, value, limit in [('vma-count', sample['vma_count'], vma_limit),
                               ('rss-bytes', sample['rss_bytes'], budget['max_rss_bytes']),
                               ('completed-cases', completed, budget['max_cases']),
                               ('plan-attempts', attempts, budget['max_attempts'])]:
        if value >= limit:
            return f'{name} reached {value} (limit {limit})'
    return None
