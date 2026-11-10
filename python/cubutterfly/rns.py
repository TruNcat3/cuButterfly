"""Prepared word64 RNS NTT plans over caller-owned CUDA buffers.

Import performs no library loading. Construction and prepare belong outside
CUDA Graph capture and kernel timing. execute enqueues work without copying or
synchronizing. Buffers and streams must outlive their asynchronous work; plans
must also outlive all captured graphs that reference them. Close only after
that work completes. A plan may be used on one stream at a time.
"""

import ctypes as ct
from contextlib import contextmanager
from enum import IntEnum
import json
import operator
import os


class Error(RuntimeError):
    def __init__(self, message, status=None):
        super().__init__(message)
        self.status = status


class Placement(IntEnum):
    SEPARATE = 0
    IN_PLACE = 1


class CachePolicy(IntEnum):
    PREFER = 0
    REQUIRE = 1
    REFRESH = 2


def _integer(value, name, maximum):
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, not bool")
    try:
        value = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc
    if not 0 <= value <= maximum:
        raise ValueError(f"{name} is outside the ABI range")
    return value


def _pointer(value, name, *, nullable=False, aligned=False):
    if isinstance(value, ct.c_void_p):
        value = value.value or 0
    value = _integer(value, name, (1 << (ct.sizeof(ct.c_void_p) * 8)) - 1)
    if not nullable and not value:
        raise ValueError(f"{name} must be non-null")
    if aligned and value % ct.alignment(ct.c_uint64):
        raise ValueError(f"{name} must be uint64 aligned")
    return ct.c_void_p(value)


def _config_json(config):
    if isinstance(config, str):
        config = json.loads(config)
    if not isinstance(config, dict):
        raise TypeError("config must be a dict or JSON object string")
    # The native parser validates the complete request and rejects unknown keys.
    return json.dumps(config, allow_nan=False, separators=(",", ":")).encode("utf-8")


class Library:
    """Load the RNS ABI explicitly from a built cuButterfly shared library."""

    def __init__(self, path=None):
        path = path if path is not None else os.environ.get("CUBUTTERFLY_LIBRARY")
        if not path:
            raise ValueError("provide a library path or set CUBUTTERFLY_LIBRARY")
        self._dll = ct.CDLL(os.fspath(path))
        pointer = ct.c_void_p
        signatures = {
            "Create": [ct.c_char_p, ct.POINTER(pointer)],
            "Prepare": [pointer, ct.c_int, ct.c_char_p, ct.c_int, ct.c_uint, pointer],
            "Execute": [pointer, pointer, pointer, pointer],
            "InfoJson": [pointer, pointer, ct.POINTER(ct.c_size_t)],
            "Destroy": [pointer],
            "LastError": [],
        }
        for name, arguments in signatures.items():
            function = getattr(self._dll, "cubutterflyRns" + name)
            function.argtypes = arguments
            function.restype = ct.c_char_p if name == "LastError" else ct.c_int

    def _call(self, name, *arguments):
        status = getattr(self._dll, "cubutterflyRns" + name)(*arguments)
        if status:
            message = self._dll.cubutterflyRnsLastError() or b"RNS operation failed"
            raise Error(message.decode("utf-8", errors="replace"), status)

    def _info(self, handle):
        size = ct.c_size_t()
        self._call("InfoJson", handle, None, ct.byref(size))
        buffer = ct.create_string_buffer(size.value)
        self._call("InfoJson", handle, buffer, ct.byref(size))
        return json.loads(buffer.value.decode("utf-8"))


class Plan:
    """Owned RNS plan; config is the native request JSON with explicit domains.

    prepare must succeed before execute. Device addresses must be integers or
    c_void_p; no tensor conversion, allocation, device selection, or implicit
    synchronization is performed. Explicit close is required: the wrapper does
    not destroy GPU resources from a garbage-collection finalizer.
    """

    def __init__(self, library, config):
        self.library = library
        self._handle = ct.c_void_p()
        self._placement = None
        payload = _config_json(config)
        library._call("Create", payload, ct.byref(self._handle))
        if not self._handle.value:
            raise Error("RNS Create returned a null plan")

    @property
    def native_handle(self):
        if not self._handle.value:
            raise Error("RNS plan is closed")
        return self._handle

    def prepare(self, *, placement=Placement.SEPARATE, cache_directory=None,
                policy=CachePolicy.PREFER, repetitions=5, stream=0):
        handle = self.native_handle
        if self._placement is not None:
            raise Error("plan is already prepared; create another plan to retune")
        placement = Placement(_integer(placement, "placement", 1))
        policy = CachePolicy(_integer(policy, "policy", 2))
        repetitions = _integer(repetitions, "repetitions", (1 << 32) - 1)
        if not repetitions:
            raise ValueError("repetitions must be positive")
        directory = b"" if cache_directory is None else os.fsencode(cache_directory)
        if b"\0" in directory:
            raise ValueError("cache_directory must not contain NUL")
        self.library._call("Prepare", handle, int(placement), directory, int(policy),
                           repetitions, _pointer(stream, "stream", nullable=True))
        self._placement = placement
        return self

    def execute(self, input, output, *, stream=0):
        handle = self.native_handle
        if self._placement is None:
            raise Error("RNS plan must be prepared before execution")
        source = _pointer(input, "input", aligned=True)
        destination = _pointer(output, "output", aligned=True)
        if (source.value == destination.value) != (self._placement == Placement.IN_PLACE):
            raise ValueError("pointer aliasing does not match prepared placement")
        self.library._call("Execute", handle, source, destination,
                           _pointer(stream, "stream", nullable=True))

    def info(self):
        return self.library._info(self.native_handle)

    def close(self):
        if self._handle.value:
            self.library._call("Destroy", self._handle)
            self._handle = ct.c_void_p()

    def __enter__(self):
        self.native_handle
        return self

    def __exit__(self, *exception):
        self.close()


@contextmanager
def replace_ntt_callback(target, attribute, callback):
    """Temporarily replace an explicit Python callback attribute.

    The caller supplies all argument/result adaptation in callback. This cannot
    intercept statically linked C++ calls. Target must expose an ordinary
    instance attribute dictionary; inherited methods are restored by deleting
    the temporary override. Concurrent patching of the same attribute is not
    supported.
    """
    if isinstance(target, type) or not hasattr(target, "__dict__"):
        raise TypeError("target must be an explicit object with instance attributes")
    if not isinstance(attribute, str) or not attribute:
        raise TypeError("attribute must be a nonempty string")
    if not callable(callback):
        raise TypeError("callback must be callable")
    original = getattr(target, attribute)
    if not callable(original):
        raise TypeError("original attribute must be callable")
    # Descriptors can mutate external state in setattr; support ordinary methods.
    descriptor = next((vars(base)[attribute] for base in type(target).__mro__
                       if attribute in vars(base)), None)
    if hasattr(descriptor, "__set__"):
        raise TypeError("callback attribute must not be a data descriptor")
    owned = attribute in vars(target)
    raw = vars(target).get(attribute)
    setattr(target, attribute, callback)
    try:
        yield callback
    finally:
        if owned:
            setattr(target, attribute, raw)
        else:
            delattr(target, attribute)
