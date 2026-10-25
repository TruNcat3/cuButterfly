"""CPU-only tests for the installed public frontend's ctypes boundary."""
import ctypes as ct
import importlib
import pathlib
import sys
import io
import json
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "python"))
import cubutterfly as cb


class Function:
    def __init__(self, dll, name):
        self.dll, self.name = dll, name

    def __call__(self, *args):
        self.dll.calls.append((self.name, args))
        if self.name == "GetLastErrorString":
            return b"specific semantic failure"
        if self.name == "GetStatusString":
            return b"INVALID_VALUE"
        if self.name == self.dll.fail:
            return 1
        if self.name in ("Create", "CreateDescriptor", "CreatePlan"):
            ct.cast(args[-1], ct.POINTER(ct.c_void_p))[0] = 123
        if self.name.endswith("Size"):
            ct.cast(args[-1], ct.POINTER(ct.c_size_t))[0] = 64
        if self.name == "PlanGetSelectionSource":
            ct.cast(args[-1], ct.POINTER(ct.c_int))[0] = 1
        if self.name in ("DescriptorGetJson", "DescriptorGetCapabilities", "PlanGetDescriptorJson",
                         "PlanGetMappingJson", "PlanGetAlgorithmName", "PlanGetSelectionReason"):
            value = b'{"from":"C library"}\0'
            ct.cast(args[-1], ct.POINTER(ct.c_size_t))[0] = len(value)
            if args[-2] is not None:
                ct.memmove(args[-2], value, len(value))
        return 0


class Dll:
    def __init__(self):
        self.calls, self.fail = [], None

    def __getattr__(self, name):
        function = Function(self, name[len("cubutterfly"):])
        setattr(self, name, function)
        return function


class FrontendTests(unittest.TestCase):
    def setUp(self):
        self.dll = Dll()
        with patch.object(ct, "CDLL", return_value=self.dll):
            self.library = cb.Library("mock.so")

    def calls(self, name):
        return [args for function, args in self.dll.calls if function == name]

    def test_import_does_not_load_library(self):
        with patch.object(ct, "CDLL", side_effect=AssertionError("unexpected library load")):
            importlib.reload(cb)

    def test_explicit_final_storage_epilogue(self):
        transform = cb.Transform(op=cb.Operator.FWHT, storage=cb.DataType.FP16,
                                 compute=cb.ComputeType.FP32, extents=[64],
                                 output_scale=-0.125, output_rounding=cb.OutputRounding.FINAL_STORAGE)
        with cb.Descriptor(self.library, transform):
            self.assertEqual(self.calls("SetOutputScale")[0][1], -0.125)
            self.assertEqual(self.calls("SetOutputRounding")[0][1], 1)

    def test_old_library_defaults_and_explicit_epilogue_rejection(self):
        class OldDll(Dll):
            def __getattr__(self, name):
                if name in ("cubutterflySetOutputScale", "cubutterflySetOutputRounding"):
                    raise AttributeError(name)
                return super().__getattr__(name)
        old = OldDll()
        with patch.object(ct, "CDLL", return_value=old):
            library = cb.Library("old.so")
        with cb.Descriptor(library, cb.Transform()):
            pass
        for transform in (cb.Transform(output_scale=0.5),
                          cb.Transform(output_rounding=cb.OutputRounding.FINAL_STORAGE)):
            with self.assertRaisesRegex(cb.Error, "does not support explicit output epilogues"):
                cb.Descriptor(library, transform)

    def test_all_semantic_fields_and_independent_strides(self):
        t = cb.Transform(op=cb.Operator.STRUCTURED_2X2, storage=cb.DataType.FP64,
                         compute=cb.ComputeType.FP64, extents=[8, 8, 8, 8], batch=7,
                         inverse=True, normalize_inverse=False, in_place=True,
                         fft_type=2, fft_output_scale=2.5, length_mode=1,
                         output_strides=[512, 64, 8, 1], input_batch_stride=9000,
                         output_batch_stride=10000, input_orders=[1, 0, 1, 0],
                         stage_matrices=[[[1, 2, 3, 4]]], execution_mode=2)
        with cb.Descriptor(self.library, t, cb.PlanOptions(mapping="explicit", execution_mode=1)) as d:
            d.validate()
            self.assertEqual(d.json(), '{"from":"C library"}')
            self.assertEqual(d.capabilities_json(), d.json())
            strides = self.calls("SetStrides")[0]
            self.assertIsNone(strides[1])
            self.assertEqual(strides[2], 9000)
            self.assertEqual(list(strides[3]), [512, 64, 8, 1])
            self.assertEqual(strides[4], 10000)
            self.assertEqual(self.calls("SetShape")[0][1:2], (4,))
            self.assertEqual(self.calls("SetExecutionMode")[0][1], 1)
            self.assertEqual(self.calls("SetFftOutputScale")[0][1], 2.5)
            self.assertEqual(self.calls("SetDirection")[0][1:], (1, 0))
            self.assertEqual(self.calls("SetPlacement")[0][1], 1)
            self.assertEqual(self.calls("SetLengthMode")[0][1], 1)
            self.assertEqual(self.calls("SetAlgorithmPolicy")[0][1:], (2, b"explicit"))
            self.assertEqual(list(self.calls("SetLogicalOrders")[0][1]), [1, 0, 1, 0])
            self.assertIsNone(self.calls("SetLogicalOrders")[0][2])
            matrix = self.calls("SetStageMatrices")[0][2][0]
            self.assertEqual((matrix.m00, matrix.m01, matrix.m10, matrix.m11), (1, 2, 3, 4))
        self.assertEqual(len(self.calls("DestroyDescriptor")), 1)

    def test_ntt_fields(self):
        with cb.Descriptor(self.library, cb.Transform(op=cb.Operator.NTT, storage=cb.DataType.UINT32,
                           modulus=17, word_bits=32, output_order=cb.NttLayout.BIT_REVERSED)):
            self.assertEqual(self.calls("SetModulus")[0][1:], (17, 32))
            self.assertEqual(self.calls("SetNttLayouts")[0][1:], (0, 2))
            self.assertFalse(self.calls("SetLogicalOrders"))

    def test_legacy_order_rejections_cleanup(self):
        for t in (cb.Transform(output_order=1), cb.Transform(op=cb.Operator.NTT,
                  output_order=2, input_orders=[0])):
            with self.assertRaises(ValueError):
                cb.Descriptor(self.library, t)
        self.assertEqual(len(self.calls("DestroyDescriptor")), 2)

    def test_json_roundtrip_delegates_to_library(self):
        with cb.Descriptor.from_json(self.library, '{"every":"field"}') as d:
            self.assertEqual(self.calls("DescriptorSetJson")[0][1], b'{"every":"field"}')
            self.assertEqual(d.json(), '{"from":"C library"}')

    def test_detailed_error_captured_before_cleanup(self):
        self.dll.fail = "SetShape"
        with self.assertRaisesRegex(cb.Error, "INVALID_VALUE: specific semantic failure"):
            cb.Descriptor(self.library, cb.Transform())
        names = [name for name, _ in self.dll.calls]
        self.assertLess(names.index("GetLastErrorString"), names.index("GetStatusString"))
        self.assertLess(names.index("GetStatusString"), names.index("DestroyDescriptor"))

    def test_plan_pointer_workspace_queries_cleanup(self):
        class Tensor:
            def data_ptr(self):
                return 456
        with cb.Plan(self.library, cb.Transform(), stream=9) as p:
            with self.assertRaisesRegex(cb.Error, "pass stream="):
                p.set_stream(10)
            self.assertEqual(len(self.calls("SetStream")), 1)
            self.assertEqual((p.workspace_size(), p.input_size(), p.output_size()), (64, 64, 64))
            p.set_workspace(Tensor(), 64)
            p.execute(Tensor(), ct.c_void_p(789))
            self.assertEqual(self.calls("Execute")[0][2].value, 456)
            self.assertEqual(self.calls("Execute")[0][3].value, 789)
            self.assertEqual(p.mapping_json(), p.descriptor_json())
            source = ct.c_int(-1)
            self.library._call("PlanGetSelectionSource", p.native_handle, ct.byref(source))
            self.assertEqual(source.value, 1)
        self.assertEqual(len(self.calls("DestroyPlan")), 1)
        self.assertEqual(len(self.calls("Destroy")), 1)

    def test_failed_plan_releases_descriptor_and_context(self):
        self.dll.fail = "CreatePlan"
        with self.assertRaisesRegex(cb.Error, "specific semantic failure"):
            cb.Plan(self.library, cb.Transform())
        self.assertEqual(len(self.calls("DestroyDescriptor")), 1)
        self.assertEqual(len(self.calls("Destroy")), 1)

    def test_non_ntt_custom_modulus_rejected(self):
        with self.assertRaisesRegex(ValueError, "require an NTT"):
            cb.Descriptor(self.library, cb.Transform(modulus=17))
        self.assertEqual(len(self.calls("DestroyDescriptor")), 1)

    def test_integer_inputs_do_not_wrap_or_coerce(self):
        size_overflow = 1 << (ct.sizeof(ct.c_size_t) * 8)
        cases = [
            {"extents": [-1]}, {"extents": [size_overflow]}, {"extents": [8.0]},
            {"extents": [True]}, {"batch": "2"}, {"batch": -1}, {"batch": size_overflow},
            {"input_strides": [1.5]}, {"output_strides": [-1]},
            {"input_batch_stride": -1}, {"output_batch_stride": size_overflow},
            {"input_orders": [True]}, {"output_orders": [1 << 31]},
            {"op": -1}, {"storage": 1.0}, {"compute": "0"}, {"fft_type": True},
            {"length_mode": 1 << 31}, {"execution_mode": -1}, {"input_order": -1},
            {"output_order": "0"}, {"word_bits": 1 << 32}, {"modulus": 1 << 64},
            {"inverse": 2}, {"normalize_inverse": 1}, {"in_place": 0},
        ]
        for changes in cases:
            with self.subTest(changes=changes):
                count = len(self.calls("SetShape"))
                with self.assertRaises((TypeError, ValueError)):
                    cb.Descriptor(self.library, cb.Transform(**changes))
                self.assertEqual(len(self.calls("SetShape")), count)
        for options in (cb.PlanOptions(policy=1.5), cb.PlanOptions(execution_mode=1 << 31)):
            with self.assertRaises((TypeError, ValueError)):
                cb.Descriptor(self.library, cb.Transform(), options)

    def test_integer_index_protocol_and_empty_matrices(self):
        class Index:
            def __index__(self):
                return 8
        with cb.Descriptor(self.library, cb.Transform(extents=[Index()], stage_matrices=[[]])):
            self.assertEqual(list(self.calls("SetShape")[0][2]), [8])
            self.assertFalse(self.calls("SetStageMatrices"))

    def test_cli_only_uses_cpu_descriptor_apis(self):
        from cubutterfly.__main__ import main
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "descriptor.json"
            path.write_text('{"every":"field"}', encoding="utf-8")
            output = io.StringIO()
            with patch.object(ct, "CDLL", return_value=self.dll), patch("sys.stdout", output):
                self.assertEqual(main([str(path), "--library", "mock.so"]), 0)
            self.assertEqual(json.loads(output.getvalue()), {
                "descriptor": {"from": "C library"}, "capabilities": {"from": "C library"}})
            self.assertFalse(self.calls("Create"))
            self.assertFalse(self.calls("CreatePlan"))

    def test_check_xd_bridge_forwards_semantics_and_workspace(self):
        import check_xd
        class Torch:
            uint8 = "uint8"
            def empty(self, size, **kwargs):
                self.request = (size, kwargs)
                return self
            def data_ptr(self):
                return 456
        torch = Torch()
        with patch.object(ct, "CDLL", return_value=self.dll):
            api = check_xd.Api(pathlib.Path("mock.so"))
            with check_xd.Handle(api, 8) as handle:
                with check_xd.make_plan(api, handle, torch, op=2, storage=2, compute=1,
                         shape=[8, 8], batch=3, input_batch_stride=512, output_batch_stride=640,
                         output_strides=[8, 1], input_orders=[1, 0], fft_output_scale=2.0,
                         mapping="saved-mapping") as plan:
                    self.assertIs(plan.workspace, torch)
                    self.assertEqual(plan.metadata["workspace_bytes"], 64)
                    self.assertEqual(torch.request, (64, {"dtype": "uint8", "device": "cuda"}))
                    args = self.calls("SetStrides")[0]
                    self.assertIsNone(args[1])
                    self.assertEqual((args[2], args[4]), (512, 640))
                    self.assertEqual(list(self.calls("SetLogicalOrders")[0][1]), [1, 0])
                    self.assertEqual(self.calls("SetAlgorithmPolicy")[0][1:], (2, b"saved-mapping"))
                    self.assertEqual(self.calls("SetFftOutputScale")[0][1], 2.0)
        self.assertEqual(len(self.calls("DestroyDescriptor")), 1)
        self.assertEqual(len(self.calls("DestroyPlan")), 1)

    def test_check_xd_old_library_is_explicitly_unsupported(self):
        import check_xd
        from types import SimpleNamespace
        with patch.object(cb, "Library", side_effect=AttributeError("cubutterflyDescriptorGetJson")):
            with self.assertRaisesRegex(check_xd.UnsupportedCase, "public descriptor API"):
                check_xd.make_plan(SimpleNamespace(path="old.so"), None, None,
                                   op=0, storage=6, compute=0, shape=[8], batch=1)


if __name__ == "__main__":
    unittest.main()
