# Direct GPU-NTT control

`gpuntt_native_compare.cu` is a timing-only control. It calls the pinned
GPU-NTT `GPU_NTT_Inplace` and `GPU_INTT_Inplace` dispatch directly, so it must
be built against the pinned vendor static library. It is intentionally kept
outside the cuButterfly provider so a vendor timing cannot be mistaken for the
current `RnsCoreFamily::GpuNttMerge` template.

Example on the A100 build used by the investigation:

```bash
nvcc -std=c++17 -O3 -arch=sm_80 \
  -Iinclude \
  -Iresults/e13_assets/sources/HEonGPU/thirdparty/GPU-NTT/src/include \
  benchmarks/gpuntt_native_compare.cu \
  -Lbuild-contract-20261017 -lcuntt \
  -L/tmp/gpuntt-build/src -lntt-1.0 \
  -Xlinker -rpath -Xlinker "$PWD/build-contract-20261017" \
  -o /tmp/cuntt_gpuntt_native_compare

CUDA_VISIBLE_DEVICES=0 /tmp/cuntt_gpuntt_native_compare 14 2 80
```

The modulus and roots are shared with the cuButterfly single-domain RNS
fixture. The output is kernel timing only; correctness remains covered by the
separate adapter regression.
