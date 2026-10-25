"""Small ctypes adapter for the optional CUDA VkFFT xD bridge.

The bridge binds one fixed contiguous Torch input/output pair to a prepared
VkFFT application.  Construction performs all VkFFT initialization, including
NVRTC compilation; ``__call__`` only appends the prepared forward or inverse
plan to the configured CUDA stream.
"""

from __future__ import annotations

import ctypes as ct
import hashlib
import operator
from pathlib import Path
from typing import Any, Sequence


_UINTPTR = ct.c_uint64 if ct.sizeof(ct.c_void_p) == 8 else ct.c_uint32


class NativeVkfft:
    """Prepared out-of-place contiguous C2C VkFFT execution.

    ``source`` and ``output`` are deliberately fixed at construction time so
    the callback is safe to put inside the same CUDA graph as other prepared
    plans.  The caller owns both Torch tensors; this class owns only VkFFT's
    application and its generated resources.
    """

    def __init__(
        self,
        torch: Any,
        shape: Sequence[int],
        batch: int,
        fp64: bool,
        inverse: bool,
        stream: Any,
        source: Any,
        output: Any,
        library: str | Path,
    ) -> None:
        self.torch = torch
        self.shape = self._validate_shape(shape)
        self.batch = self._validate_batch(batch)
        self.fp64 = self._validate_flag(fp64, "fp64")
        self.inverse = self._validate_flag(inverse, "inverse")
        self._validate_tensors(torch, source, output)
        stream_value = self._stream_value(stream)

        path = Path(library).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"VkFFT xD bridge library not found: {path}")
        self.library = path
        self.library_sha256 = self._sha256(path)
        self._lib = ct.CDLL(str(path))
        self._bind()
        self.version = int(self._lib.vkfft_xd_version())
        self._handle = ct.c_void_p()

        shape_array = (ct.c_uint64 * len(self.shape))(*self.shape)
        status = self._lib.vkfft_xd_create(
            ct.byref(self._handle),
            ct.c_uint32(len(self.shape)),
            shape_array,
            ct.c_uint64(self.batch),
            ct.c_int(int(self.fp64)),
            ct.c_int(int(self.inverse)),
            _UINTPTR(stream_value),
            _UINTPTR(int(source.data_ptr())),
            _UINTPTR(int(output.data_ptr())),
        )
        if status != 0:
            handle = self._handle
            self._handle = None
            if handle is not None and handle.value:
                self._lib.vkfft_xd_destroy(handle)
            raise RuntimeError(self._format_error(status, "vkfft_xd_create"))

        self.source = source
        self.output = output

    @staticmethod
    def _validate_shape(shape: Sequence[int]) -> tuple[int, ...]:
        try:
            raw_values = tuple(shape)
        except (TypeError, ValueError) as exc:
            raise TypeError("shape must be an integer sequence") from exc
        if any(isinstance(value, bool) for value in raw_values):
            raise TypeError("shape must be an integer sequence, not bool values")
        try:
            values = tuple(operator.index(value) for value in raw_values)
        except (TypeError, ValueError) as exc:
            raise TypeError("shape must be an integer sequence") from exc
        if len(values) not in (1, 2, 3):
            raise ValueError("VkFFT xD bridge supports rank 1..3")
        if any(value <= 0 for value in values):
            raise ValueError("shape dimensions must be positive")
        return values

    @staticmethod
    def _validate_batch(batch: int) -> int:
        if isinstance(batch, bool):
            raise TypeError("batch must be an integer, not bool")
        try:
            value = operator.index(batch)
        except (TypeError, ValueError) as exc:
            raise TypeError("batch must be a positive integer") from exc
        if value <= 0:
            raise ValueError("batch must be a positive integer")
        return value

    @staticmethod
    def _validate_flag(value: bool, name: str) -> bool:
        if value not in (False, True):
            raise ValueError(f"{name} must be bool or 0/1")
        return bool(value)

    @staticmethod
    def _stream_value(stream: Any) -> int:
        if stream is None:
            return 0
        candidate = getattr(stream, "cuda_stream", stream)
        try:
            value = operator.index(candidate)
        except (TypeError, ValueError) as exc:
            raise TypeError("stream must be a CUDA stream or integer handle") from exc
        if value < 0:
            raise ValueError("stream handle must be non-negative")
        return value

    def _validate_tensors(self, torch: Any, source: Any, output: Any) -> None:
        for name, tensor in (("source", source), ("output", output)):
            if not hasattr(tensor, "data_ptr") or not hasattr(tensor, "shape"):
                raise TypeError(f"{name} must be a Torch tensor")
            if getattr(tensor.device, "type", None) != "cuda":
                raise ValueError(f"{name} must be a CUDA tensor")
            if not tensor.is_contiguous():
                raise ValueError(f"{name} must be contiguous")

        expected_shape = (self.batch, *self.shape)
        if tuple(int(value) for value in source.shape) != expected_shape:
            raise ValueError(f"source shape {tuple(source.shape)} != {expected_shape}")
        if tuple(int(value) for value in output.shape) != expected_shape:
            raise ValueError(f"output shape {tuple(output.shape)} != {expected_shape}")
        expected_dtype = torch.complex128 if self.fp64 else torch.complex64
        if source.dtype != expected_dtype or output.dtype != expected_dtype:
            raise ValueError(
                f"source/output dtype must be {expected_dtype}; "
                f"got {source.dtype} and {output.dtype}"
            )
        if source.device != output.device:
            raise ValueError("source and output must be on the same CUDA device")
        source_ptr = int(source.data_ptr())
        output_ptr = int(output.data_ptr())
        if source_ptr == 0 or output_ptr == 0:
            raise ValueError("source/output device pointers must be non-null")
        if source_ptr == output_ptr:
            raise ValueError("source and output must be distinct for out-of-place FFT")

        device_index = getattr(source.device, "index", None)
        if device_index is not None and hasattr(torch.cuda, "current_device"):
            if int(torch.cuda.current_device()) != int(device_index):
                raise ValueError("source device must be the current CUDA device")

    def _bind(self) -> None:
        self._lib.vkfft_xd_create.restype = ct.c_int
        self._lib.vkfft_xd_create.argtypes = [
            ct.POINTER(ct.c_void_p),
            ct.c_uint32,
            ct.POINTER(ct.c_uint64),
            ct.c_uint64,
            ct.c_int,
            ct.c_int,
            _UINTPTR,
            _UINTPTR,
            _UINTPTR,
        ]
        self._lib.vkfft_xd_execute.restype = ct.c_int
        self._lib.vkfft_xd_execute.argtypes = [ct.c_void_p]
        self._lib.vkfft_xd_destroy.restype = ct.c_int
        self._lib.vkfft_xd_destroy.argtypes = [ct.c_void_p]
        self._lib.vkfft_xd_version.restype = ct.c_int
        self._lib.vkfft_xd_version.argtypes = []
        self._lib.vkfft_xd_last_error.restype = ct.c_char_p
        self._lib.vkfft_xd_last_error.argtypes = []

    def _error_text(self) -> str:
        raw = self._lib.vkfft_xd_last_error()
        if not raw:
            return "unknown VkFFT xD bridge error"
        return raw.decode("utf-8", errors="replace")

    def _format_error(self, status: int, operation: str) -> str:
        return f"{operation} failed with status {status}: {self._error_text()}"

    def __call__(self) -> None:
        self.execute()

    def execute(self) -> None:
        if self._handle is None or not self._handle.value:
            raise RuntimeError("VkFFT xD plan is closed")
        status = self._lib.vkfft_xd_execute(self._handle)
        if status != 0:
            raise RuntimeError(self._format_error(status, "vkfft_xd_execute"))

    def metadata(self) -> dict[str, Any]:
        return {
            "library": str(self.library),
            "library_sha256": self.library_sha256,
            "vkfft_version": self.version,
            "shape": list(self.shape),
            "batch": self.batch,
            "precision": "fp64" if self.fp64 else "fp32",
            "direction": "inverse" if self.inverse else "forward",
            "placement": "out-of-place",
            "normalize_inverse": self.inverse,
        }

    def close(self) -> None:
        handle = self._handle
        if handle is None or not handle.value:
            self._handle = None
            return
        status = self._lib.vkfft_xd_destroy(handle)
        if status != 0:
            raise RuntimeError(self._format_error(status, "vkfft_xd_destroy"))
        self._handle = None

    def __enter__(self) -> "NativeVkfft":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1 << 20), b""):
                digest.update(block)
        return digest.hexdigest()


__all__ = ["NativeVkfft"]
