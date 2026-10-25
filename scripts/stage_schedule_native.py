"""Optional native execution of the unchanged CPU stage event simulation."""
from __future__ import annotations

import ctypes
from functools import lru_cache
import hashlib
import os
import pathlib
import subprocess
import tempfile


@lru_cache(maxsize=2)
def _load(allow_compile=True):
    if os.environ.get("CUBUTTERFLY_DISABLE_NATIVE_STAGE_SCHEDULE") == "1":
        return None
    source = pathlib.Path(__file__).with_suffix(".cpp")
    directory = pathlib.Path(tempfile.gettempdir()) / f"cubutterfly-stage-schedule-{os.getuid()}"
    try:
        identity = hashlib.sha256(source.read_bytes() + os.uname().machine.encode()).hexdigest()[:24]
        target = directory / f"{identity}.so"
        if not target.exists():
            if not allow_compile:
                return None
            directory.mkdir(mode=0o700, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix="compile-", dir=directory) as temporary:
                output = pathlib.Path(temporary) / "schedule.so"
                subprocess.run(["c++", "-std=c++11", "-O3", "-ffp-contract=off", "-fno-fast-math",
                                "-shared", "-fPIC", str(source), "-o", str(output)],
                               check=True, capture_output=True, timeout=60)
                os.replace(output, target)
        library = ctypes.CDLL(str(target))
        function = library.cubutterfly_compose
        pointer = ctypes.POINTER(ctypes.c_double)
        function.argtypes = [pointer, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                             ctypes.c_double, pointer]
        function.restype = ctypes.c_double
    except (OSError, AttributeError, subprocess.SubprocessError):
        return None
    return function


def _active_function():
    if os.environ.get("CUBUTTERFLY_DISABLE_NATIVE_STAGE_SCHEDULE") == "1":
        return None
    return _load(allow_compile=os.environ.get("CUBUTTERFLY_STAGE_SCHEDULE_LOAD_ONLY") != "1")


def compose(stages, tiles, factor, overlap, last_fraction, tail_stages):
    function = _active_function()
    if function is None:
        return None
    values = (ctypes.c_double * (4 * len(stages)))(*(value for stage in stages for value in stage))
    tail = None if tail_stages is None else (ctypes.c_double * (4 * len(tail_stages)))(
        *(value for stage in tail_stages for value in stage))
    return function(values, len(stages), tiles, factor, overlap, last_fraction, tail)


def backend():
    """Report the active backend; LOAD_ONLY=1 never builds a missing library."""
    return "native-cpp-double" if _active_function() is not None else "python-event-simulation"
