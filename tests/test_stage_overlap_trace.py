import pathlib
import sqlite3
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
from analyze_stage_overlap import overlap, analyze_factors
from benchmark_stage_overlap import summarize_paired
import pytest


def test_kernel_overlap_counts_intersection_not_stream_presence():
    assert overlap([(0, 10), (20, 30)], [(10, 20), (30, 40)]) == 0
    assert overlap([(0, 10), (20, 30)], [(5, 25)]) == 10
    assert overlap([], [(0, 10)]) == 0


@pytest.mark.parametrize("bad_join",[False,True])
def test_factor_trace_checks_slice_dependencies_and_final_join(tmp_path,bad_join):
    path=tmp_path/"trace.sqlite"
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE StringIds (id INTEGER,value TEXT)")
        db.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start INTEGER,end INTEGER,streamId INTEGER,demangledName INTEGER)")
        for group in range(3):
            name=f"void cuntt::detail::factor_streamed::kernel<FFT<float, 64>, 18u, 6u, {group*6}u, 4u, 1u, 0u, false, cuntt::Complex32, true>()"
            db.execute("INSERT INTO StringIds VALUES (?,?)",(group,name))
        db.executemany("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,?,?)",[
            (0,10,11,0),(10,20,11,0),
            (10,25,12,1),(25,40,12,1),
            (39 if bad_join else 40,50,13,2)])
    result=analyze_factors(path,2)
    assert result["factor_partition"]==[6,6,6] and result["invocations"]==1
    assert result["early_consumer_launches"]==1
    assert result["dependency_order_verified"] is not bad_join
    assert result["actual_overlap"] is not bad_join


def test_paired_summary_compares_matching_mappings_and_rounds():
    rows = [dict(logN=20, batch=1, round=r, mode=m, kernel_ms=t)
            for r in (0, 1) for m, t in (("linear-p256-table", 4), ("xor-p256-table", 2), ("cufft", 1))]
    result = summarize_paired(rows)[0]
    assert result["best"] == "xor-p256-table"
    assert result["best_over_cufft_latency"] == 2
    assert result["xor_vs_same_mapping"]["xor-p256-table"]["median_latency_reduction_pct"] == 50
    aggregate = summarize_paired(rows)[-1]
    assert aggregate["aggregate"] == "mean_best_throughput_fraction"
    assert aggregate["percent_of_cufft"] == 50
    assert aggregate["target_met"] is False
    with pytest.raises(ValueError):
        summarize_paired(rows[:-1])
    with pytest.raises(ValueError):
        summarize_paired(rows + rows[:1])
