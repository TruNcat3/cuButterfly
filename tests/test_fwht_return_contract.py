"""CPU-only checks for the bounded main66 return-contract cohort."""
import importlib.util
from pathlib import Path
import sys

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
spec=importlib.util.spec_from_file_location('fwht_return',ROOT/'scripts/benchmark_fwht_return_contract.py')
bench=importlib.util.module_from_spec(spec);spec.loader.exec_module(bench)


def test_frozen_ten_requests_preserve_layout_precision_and_direction():
    config=bench.load_config()
    assert len(config['cases'])==10
    assert sum(c['descriptor']['semantics']['inverse'] for c in config['cases'])==5
    assert config['protocol']['graph_calls']==20 and config['protocol']['replays']==7
    import copy
    for key,value in [('storage',0),('placement',1),('output_orders',[1])]:
        s=copy.deepcopy(config['cases'][0]['descriptor']['semantics']);s[key]=value
        with pytest.raises(ValueError):bench.validate_semantics(s)


def test_returned_contract_allocates_distinct_outputs_without_copy():
    outputs=[]
    def allocate():
        value={};outputs.append(value);return value
    def execute(source,output):output['data']=source['data']
    source={'data':1}
    first=bench.returned_call(allocate,execute,source)
    source['data']=2
    second=bench.returned_call(allocate,execute,source)
    assert first is not second and first['data']==1 and second['data']==2
    assert len(outputs)==2


def test_order_alternates():
    assert bench.backend_order(0,1)==list(reversed(bench.backend_order(0,2)))
    assert bench.backend_order(0,1)==list(reversed(bench.backend_order(1,1)))


def test_historical_payloads_are_bound_when_raw_evidence_is_available():
    import json
    cases=bench.load_config()['cases']
    if not all((ROOT/case['historical_source']).is_file() for case in cases):
        pytest.skip('historical raw payloads are not included in the source release')
    for case in cases:
        path=ROOT/case['historical_source']
        assert bench.sha(path)==case['historical_source_sha256']
        row=json.loads(path.read_text())['row']
        assert row['descriptor']==case['descriptor']
        assert bench.semantic.digest(row)==case['historical_row_sha256']


def test_portable_frozen_summary_recomputes_without_raw_results():
    import json
    import math
    import statistics
    path=ROOT/'paper/generated/fwht_return_contract_20261107.json'
    data=json.loads(path.read_text())
    assert len(data['records'])==120 and len(data['pairs'])==60
    assert sum(len(row['timing']['samples_ms']) for row in data['records'])==840
    cases={case['id']:case for case in data['config']['cases']}
    for row in data['records']:
        assert row['timing']['median_ms']==statistics.median(row['timing']['samples_ms'])
        assert row['semantics']==cases[row['case_id']]['descriptor']['semantics']
        if row['interface']=='native-return':
            assert row['lifecycle']['independent_return_addresses']
            assert row['lifecycle']['previous_output_preserved']
    for pair in data['pairs']:
        rows={row['backend']:row for row in data['records'] if row['trial']==pair['trial']
              and row['case_id']==pair['case_id'] and row['interface']==pair['interface']}
        assert pair['speedup']==rows['dao']['timing']['median_ms']/rows['cubutterfly']['timing']['median_ms']
    for group in data['groups']:
        for trial in group['trials']:
            values=[pair['speedup'] for pair in data['pairs'] if pair['interface']==group['interface'] and pair['trial']==trial['trial']]
            assert trial['pairs']==10
            assert trial['gm']==pytest.approx(math.exp(statistics.mean(map(math.log,values))),rel=1e-14)
            assert trial['wins']==sum(value>1 for value in values)
    assert all(status['state']=='complete' for status in data['guard_records'].values())
    assert data['reference_continuity']['all_ten_reference_value_equal']
    for row in data['absolute_latency_and_result_working_set']:
        assert row['caller_result_live_bytes']==row['result_bytes_per_call']*(20 if row['interface']=='native-return' else 1)
