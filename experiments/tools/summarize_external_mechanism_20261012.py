#!/usr/bin/env python3
"""CPU-only summary for the bounded external profile extension."""
import json
from pathlib import Path
import statistics
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/tools'))
from summarize_mechanism_cohort_20261012 import csv_rows
from external_mechanism_20261012 import OUT, save


def main():
    timing, profiles, numerics = {}, {}, {}
    for implementation in ['cubutterfly','external']:
        trials=[json.loads((OUT/f'ntt.{implementation}.trial{i}.json').read_text()) for i in range(1,4)]
        assert all(t['status']=='pass' and t['correctness']['exact'] for t in trials)
        timing[implementation]=dict(median_us=1000*statistics.median(t['timing']['median_ms'] for t in trials),
            independent_process_trials=3,process_median_us=[1000*t['timing']['median_ms'] for t in trials],
            protocol=trials[0]['timing']['protocol'],mapping=trials[0]['metadata'])
        stem=OUT/f'ntt.{implementation}'
        rows=csv_rows(Path(str(stem)+'.ncu.csv').read_text(),'"ID"')
        kernels={}
        for row in rows:
            kernel=kernels.setdefault(row['ID'],dict(name=row['Kernel Name'],block=row['Block Size'],grid=row['Grid Size'],metrics={}))
            kernel['metrics'][row['Metric Name']]=float(row['Metric Value'].replace(',',''))
        sums={}
        for kernel in kernels.values():
            for key,value in kernel['metrics'].items():
                if key.endswith('.sum'):sums[key]=sums.get(key,0)+value
        result=subprocess.run(['nsys','stats','--report','cuda_gpu_trace','--format','csv',str(stem)+'.nsys.nsys-rep'],text=True,capture_output=True,check=True)
        Path(str(stem)+'.nsys.trace.csv').write_text(result.stdout)
        trace=csv_rows(result.stdout,'Start (ns)')
        intervals=sorted((int(row['Start (ns)']),int(row['Start (ns)'])+int(row['Duration (ns)'])) for row in trace if row['Name'])
        merged=[]
        for a,b in intervals:
            if merged and a<=merged[-1][1]:merged[-1][1]=max(b,merged[-1][1])
            else:merged.append([a,b])
        profiles[implementation]=dict(logical_calls=1,warmup=100,kernel_count=len(kernels),nsys_kernel_count=len(intervals),
            kernels=list(kernels.values()),per_call_counter_sums=sums,
            l2_read_write_bytes_from_32byte_sectors=32*(sums['lts__t_sectors_op_read.sum']+sums['lts__t_sectors_op_write.sum']),
            nsys_interval_union_us=sum(b-a for a,b in merged)/1000,
            nsys_first_to_last_kernel_us=(max(b for a,b in intervals)-min(a for a,b in intervals))/1000,
            occupancy='Not collected; register/shared allocations are resource evidence, not measured occupancy.',
            percentage_aggregation='Per-kernel only; no naive average or L2 hit percentage.',normal_timing_accepted=False)
        fwht=json.loads((OUT/f'fwht.{implementation}.trial1.json').read_text())
        numerics[implementation]=dict(status='numerical_tolerance_unassessed',legacy_guard_passed=False,
            timing_permitted=False,error=fwht['error'],distribution=fwht['correctness'],
            historical_comparison_only=True,full_elements=524288)
    save(OUT/'summary.json',dict(schema='cubutterfly-external-mechanism-summary-v1',timing=timing,
        gpu_ntt_over_fixed_resident_control=timing['external']['median_us']/timing['cubutterfly']['median_us'],
        profiles=profiles,fwht=numerics,dao_n65536_gap=json.loads((OUT/'cohort.json').read_text())['dao_n65536_gap'],
        interpretation='Compare GPU-NTT Merge against fixed shared-iterative resident8 diagnostic, not public-selector fastest cuButterfly; does not replace E1.',
        stagewise_control_proposal='Existing NTT baseline backend packs one useful butterfly per thread across full batch (src/ntt.cu stage_kernel); same radix2 Shoup primitive, but includes required bit-reversal copy and uses in-place stage execution. Measure complete natural-order call and state those boundary differences; do not label as single-axis same-lowering control. No kernel implementation required.'))
    print(json.dumps({'ratio':timing['external']['median_us']/timing['cubutterfly']['median_us'],'profiles':{key:value['kernel_count'] for key,value in profiles.items()}}))


if __name__=='__main__':main()
