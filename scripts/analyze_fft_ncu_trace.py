#!/usr/bin/env python3
"""Summarize per-kernel FFT Nsight Compute CSV metrics."""
import argparse, csv, json, pathlib

def main():
    p=argparse.ArgumentParser(); p.add_argument('csv',type=pathlib.Path); p.add_argument('--output',type=pathlib.Path)
    a=p.parse_args(); rows=list(csv.DictReader(a.csv.read_text().splitlines()[2:])); out={}
    for r in rows:
        if 'Metric Value' not in r: continue
        out.setdefault(r['Kernel Name'],{})[r['Metric Name']]=r['Metric Value']
    result=[]
    for name,m in out.items():
        result.append({'kernel':name,'dram_pct':m.get('dram__throughput.avg.pct_of_peak_sustained_elapsed'),
                       'sm_pct':m.get('sm__throughput.avg.pct_of_peak_sustained_elapsed'),
                       'shared_load_conflicts':m.get('l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum'),
                       'shared_store_conflicts':m.get('l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum'),
                       'barrier_stall_pct':m.get('smsp__warp_issue_stalled_barrier_per_warp_active.pct'),
                       'long_scoreboard_pct':m.get('smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct'),
                       'registers_per_thread':m.get('launch__registers_per_thread')})
    text=json.dumps(result,indent=2)+'\n'
    if a.output: a.output.write_text(text)
    else: print(text,end='')
if __name__=='__main__': main()
