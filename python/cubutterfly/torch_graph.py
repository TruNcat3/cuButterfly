"""Optional PyTorch whole-function CUDA Graph integration; torch is imported lazily."""


class InferenceGraph:
    """Capture a Tensor -> Tensor inference function for one fixed input contract.

    Construction warms up and captures outside application timing. Calls must
    be serialized on ``stream`` and use the captured shape, dtype and device.
    The callable and its parameters must remain alive and unchanged; all
    modules must be in eval mode. Autograd and training are unsupported.

    Every call copies the supplied input, replays, and clones the result. The
    returned Tensor owns independent storage and survives later calls. All
    three operations belong in application timing. Results are asynchronous
    on the declared stream; consumers on other streams must synchronize.
    This wrapper does not choose or alter the callable's operator plans.
    """

    def __init__(self, function, example, *, stream=None, warmup=3):
        import torch
        if not isinstance(example, torch.Tensor) or not example.is_cuda:
            raise ValueError("a CUDA Tensor example is required")
        if example.requires_grad:
            raise ValueError("inference input must not require gradients")
        if warmup < 1:
            raise ValueError("warmup must be positive")
        self.torch = torch
        self.function = function
        self.shape, self.dtype, self.device = tuple(example.shape), example.dtype, example.device
        self.stream = stream if stream is not None else torch.cuda.Stream(device=self.device)
        if torch.device(self.stream.device) != self.device:
            raise ValueError("capture stream device differs from input device")
        if int(self.stream.cuda_stream) == 0:
            raise ValueError("capture requires a non-default CUDA stream")
        self._check_module()
        self.stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(self.stream), torch.inference_mode():
            self._input = example.clone()
            for _ in range(warmup):
                output = function(self._input)
            if not isinstance(output, torch.Tensor):
                raise ValueError("inference function must return one Tensor")
        self.stream.synchronize()
        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.stream(self.stream), torch.inference_mode():
            with torch.cuda.graph(self._graph, stream=self.stream):
                self._output = function(self._input)
        self.stream.synchronize()

    def _check_module(self):
        modules = getattr(self.function, "modules", None)
        if modules is not None and any(module.training for module in modules()):
            raise ValueError("all captured modules must be in eval mode")

    def _validate(self, value):
        torch = self.torch
        if not isinstance(value, torch.Tensor):
            raise ValueError("input must be a Tensor")
        if tuple(value.shape) != self.shape or value.dtype != self.dtype or value.device != self.device:
            raise ValueError("input shape/dtype/device differs from capture")
        if value.requires_grad:
            raise ValueError("inference input must not require gradients")
        if int(torch.cuda.current_stream(self.device).cuda_stream) != int(self.stream.cuda_stream):
            raise ValueError("call must run on the declared capture stream")
        self._check_module()

    def __call__(self, value):
        self._validate(value)
        with self.torch.inference_mode():
            self._input.copy_(value)
            self._graph.replay()
            return self._output.clone()

    def contract(self):
        return dict(shape=list(self.shape), dtype=str(self.dtype), device=str(self.device),
                    stream=int(self.stream.cuda_stream), inference_only=True,
                    serialized_calls=True, input_copy_included=True,
                    output_ownership="independent clone per call; clone included in timing")
