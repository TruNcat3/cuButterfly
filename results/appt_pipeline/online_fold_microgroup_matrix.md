# APPT Online Per-Fold Microgroup Screen

Tesla V100, `logN=20`, three trials and 50 event-timed repetitions per point.
`publish` is the number of adjacent A dependency groups combined by fold 1;
`output` is the number of adjacent C groups combined by fold 2. The fine
`1/1` implementation is the reference.

| word bits | batch | fine 1/1 ms | best publish/output | best ms | speedup |
|---:|---:|---:|:--|---:|---:|
| 32 | 1 | 0.2574 | 4/1 | 0.2440 | 1.055x |
| 32 | 4 | 0.7659 | 2/1 | 0.7494 | 1.022x |
| 64 | 1 | 0.6094 | 1/1 | 0.6094 | 1.000x |
| 64 | 4 | 1.7790 | 1/1 | 1.7790 | 1.000x |

The response is not monotonic. Wider stores reduce transaction fragmentation,
but a fold-1 CTA then waits for multiple upstream A groups and can block an
otherwise runnable task. Fold-2 warp-local output grouping loses in every
measured shape. The current default therefore remains `1/1` for uint64, while
uint32 can select a publication-only microgroup.

Raw corrected samples are in `online_fold_microgroup_matrix_fixed.csv`. The
earlier matrix is retained as an invalidated artifact: its template dispatch
crossed the publication and output group conditions, so independent variants
did not activate the requested path.

A separate CTA-role screen confirms that the best role weights depend on both
numeric width and batch. The selected `(fold0, fold1, fold2)` weights are
`6/6/8`, `7/7/6`, `8/8/4`, and `5/7/8` for 32-bit batch 1, 32-bit batch 4,
64-bit batch 1, and 64-bit batch 4 respectively. The 64-bit batch-4 change from
`7/7/6` to `5/7/8` improves the median from 1.8267 to 1.8008 ms (`1.014x`).
This modest gain identifies fold 2 as the large-load bottleneck but also shows
that static weighting alone cannot close the remaining gap.
