# FFT v0.6/v0.8/cuFFT Three-Way Comparison

All rows use the same binary, FP32 forward in-place transform, and timing protocol.

| logN | Batch | v0.6 ms | v0.8 ms | cuFFT ms | v0.6/cuFFT | v0.8/cuFFT | v0.8 mapping |
|--:|--:|--:|--:|--:|--:|--:|:--|
| 18 | 2 | 0.029778 | 0.030351 | 0.031724 | 1.065x | 1.045x | fft18-fp32_b2_d4_3x3x3x9_t512x512x512x512_e8x8x8x16_rrr_ffg_gt256x256_ge8x8 |
| 18 | 16 | 0.201257 | 0.202199 | 0.226447 | 1.125x | 1.120x | fft18-fp32_b16_d4_3x6x3x6_t512x512x512x512_e8x16x8x16_rrr_fgf_gt256x256_ge8x8 |
| 18 | 64 | 0.812708 | 0.793375 | 0.707052 | 0.870x | 0.891x | fft18-fp32_b64_p8s10_t128x128_e16x16_rec |
| 20 | 2 | 0.119460 | 0.120381 | 0.121364 | 1.016x | 1.008x | fft20-fp32_b2_p10s10_t512x512_e8x8_rec |
| 20 | 8 | 0.416051 | 0.405811 | 0.383181 | 0.921x | 0.944x | fft20-fp32_b8_d4_10x3x3x4_t512x512x512x512_e16x8x8x16_rrr_gff_gt256x128_ge16x16 |
| 20 | 16 | 0.827249 | 0.778383 | 0.724132 | 0.875x | 0.930x | fft20-fp32_b16_d4_3x3x4x10_t512x512x512x512_e8x8x16x16_rrr_ffg_gt256x128_ge16x16 |
