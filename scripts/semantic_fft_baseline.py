"""Prepared external FFT axis composition for logical ranks above three."""
from __future__ import annotations

import math
import hashlib
from pathlib import Path


def support(name, s, vkfft_library=None, max_vendor_rank=3):
    """Validate only the composed core; physical layouts belong to the boundary."""
    def reject(reason, status='unsupported-semantic'):
        return dict(supported=False, status=status, reason=reason, metadata={})

    shape = s.get('shape', [])
    if type(max_vendor_rank) is not int or not 1 <= max_vendor_rank <= 3:
        return reject('max_vendor_rank must be an integer in [1,3]')
    if name not in ('cufft', 'vkfft') or s.get('operator') != 0 or s.get('fft_type') != 0:
        return reject('external axis composition requires C2C FFT')
    if not isinstance(shape, (list, tuple)) or len(shape) <= 3:
        return reject('external axis composition requires logical rank above three')
    if any(type(n) is not int or n <= 0 or n >= 2**31 for n in shape):
        return reject('axis lengths must fit positive signed 32-bit integers')
    batch = s.get('batch')
    if type(batch) is not int or batch <= 0:
        return reject('batch must be a positive integer')
    groups = [shape[start:start + max_vendor_rank] for start in range(0, len(shape), max_vendor_rank)]
    if any(math.prod(group) >= 2**31 for group in groups):
        return reject('vendor group point count must fit positive signed 32-bit integers')
    if any(batch * math.prod(shape) // math.prod(group) >= 2**31 for group in groups):
        return reject('complete per-group line batch must fit signed 32-bit integers')
    storage = s.get('storage')
    if storage not in (6, 7) or s.get('compute') not in ((0, 1) if storage == 6 else (0, 2)):
        return reject('C2C storage and requested compute precision must match')
    if s.get('length_mode') != 0:
        return reject('axis composition does not implement length embedding')
    if type(s.get('inverse')) is not int or s['inverse'] not in (0, 1) or type(s.get('normalize_inverse')) is not bool:
        return reject('canonical inverse and boolean normalization required')
    if name == 'vkfft' and vkfft_library is None:
        return reject('--vkfft-library not supplied', 'missing-external-dependency')
    return dict(supported=True, status='supported', reason='complete logical FFT via grouped vendor axes',
                metadata=dict(implementation='external-axis-composition', logical_shape=list(shape),
                              batch=batch, axis_order=list(range(len(shape))),
                              max_vendor_rank=max_vendor_rank,
                              grouping_policy='consecutive logical axes, largest allowed group first'))


def _vendor_core(name, torch, shape, lines, fp64, inverse, stream, source, output, library):
    if name == 'cufft':
        from diagnose_xd_fft import NativeCufft
        return NativeCufft(torch, shape, lines, fp64, inverse, stream, source, output)
    from vkfft_xd import NativeVkfft
    return NativeVkfft(torch, shape, lines, fp64, inverse, stream, source, output, library)


class AxisComposition:
    """Compose normalized vendor plans over consecutive groups of up to 3 axes.

    The two packed buffers are shared by sequential axis plans on one stream.
    Tensor views and vendor workspaces are prepared before graph capture.
    """

    def __init__(self, name, torch, s, source, output, stream, vkfft_library=None,
                 max_vendor_rank=3):
        contract = support(name, s, vkfft_library, max_vendor_rank)
        if not contract['supported']:
            raise ValueError(contract['reason'])
        expected_shape = (s['batch'], *s['shape'])
        dtype = torch.complex128 if s['storage'] == 7 else torch.complex64
        for tensor in (source, output):
            if tuple(tensor.shape) != expected_shape or tensor.dtype != dtype or not tensor.is_contiguous():
                raise ValueError('axis composition requires matching packed logical tensors')
        if source.device != output.device or source.data_ptr() == output.data_ptr():
            raise ValueError('axis composition requires separate source/output on the same device')
        self.source, self.output = source, output
        self.steps = []
        self.name, self.inverse = name, bool(s['inverse'])
        self.shape, self.batch = list(s['shape']), s['batch']
        self.stream = int(stream.cuda_stream)
        self.packed_source = torch.empty_like(source)
        self.packed_output = torch.empty_like(output)
        self._metadata = dict(contract['metadata'], provider=name, caller_stream=self.stream,
                              adapter_source=str(Path(__file__).resolve()),
                              adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                              precision='fp64' if s['storage'] == 7 else 'fp32',
                              direction='inverse' if self.inverse else 'forward',
                              normalize_inverse=s['normalize_inverse'],
                              core_normalize_inverse=self.inverse,
                              fft_output_scale=s['fft_output_scale'],
                              scratch_tensor_bytes=2 * source.numel() * source.element_size(),
                              workspace_policy='vendor plans and two shared packed buffers allocated before capture',
                              included_operations=[], groups=[])
        current = source
        try:
            for start in range(0, len(self.shape), max_vendor_rank):
                logical_axes = list(range(start, min(start + max_vendor_rank, len(self.shape))))
                tensor_axes = [axis + 1 for axis in logical_axes]
                vendor_shape = [self.shape[axis] for axis in logical_axes]
                points = math.prod(vendor_shape)
                order = [i for i in range(len(expected_shape)) if i not in tensor_axes] + tensor_axes
                permuted_shape = tuple(expected_shape[i] for i in order)
                lines = source.numel() // points
                packed_source = self.packed_source.view(lines, *vendor_shape)
                packed_output = self.packed_output.view(lines, *vendor_shape)
                core = _vendor_core(name, torch, vendor_shape, lines, s['storage'] == 7,
                                    self.inverse, stream, packed_source, packed_output, vkfft_library)
                self.steps.append((self.packed_source.view(permuted_shape), current.permute(order),
                                   core, output.permute(order), self.packed_output.view(permuted_shape)))
                group_metadata = dict(logical_axes=logical_axes, vendor_shape=vendor_shape,
                                      complete_line_batch=lines, input_permutation=order,
                                      vendor_rank=len(vendor_shape), points=points)
                if hasattr(core, 'metadata'):
                    group_metadata['vendor'] = core.metadata()
                self._metadata['groups'].append(group_metadata)
                label = f'logical axes {logical_axes}'
                self._metadata['included_operations'].extend([
                    f'{label}: GPU copy logical group-last view into preallocated packed input',
                    f'{label}: {name} native rank{len(vendor_shape)} C2C shape {vendor_shape} complete line batch {lines}',
                ])
                if self.inverse:
                    self._metadata['included_operations'].append(
                        f'{label}: inverse normalization by 1/{points}' +
                        (' via GPU multiply in NativeCufft' if name == 'cufft' else ' inside VkFFT plan'))
                self._metadata['included_operations'].append(
                    f'{label}: GPU copy packed output back to logical group-last view')
                current = output
        except BaseException:
            self.close()
            raise
        self.closed = False

    def __call__(self):
        if self.closed:
            raise RuntimeError('axis composition is closed')
        for packed, logical, core, logical_output, packed_output in self.steps:
            packed.copy_(logical)
            core()
            logical_output.copy_(packed_output)

    def metadata(self):
        return self._metadata

    def close(self):
        self.closed = True
        steps, self.steps = self.steps, []
        first_error = None
        for _, _, core, _, _ in steps:
            try:
                core.close()
            except Exception as error:
                first_error = first_error or error
        if first_error is not None:
            raise first_error
