import importlib.util
import json
import math
from pathlib import Path
import statistics

import pytest
ROOT=Path(__file__).resolve().parents[1]
ARTIFACT=ROOT/'paper/generated/fwht_native_repair_20261107.json'
spec=importlib.util.spec_from_file_location('fwht_native_repair',ROOT/'paper/tools/export_fwht_native_repair.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

def test_portable_frozen_samples_statistics_semantics_lifecycle():
    d=json.loads(ARTIFACT.read_text())
    assert d['schema']=='fwht-native-repair-confirmation-v1'
    assert 'not current new AUTO or DEFAULT' in d['interpretation']
    assert d['counts']==dict(requests=10,processes=3,records=240,errors=240,failures=0)
    assert len(set(d['process_pids']))==3
    assert d['measurement_gpu_uuid']=='GPU-0eae5f07-33de-321b-70a9-ffeae49309e6'
    assert d['old_source_provenance_gpu_uuid']!=d['measurement_gpu_uuid']
    assert d['new_library_sha256']==d['build_receipt']['library_sha256']
    for row in d['rows']:
        s=row['semantics'];assert s['storage']==2 and s['compute']==1 and s['fft_output_scale']==1
        assert len(row['measurements'])==24
        for r in row['measurements']:
            assert len(r['samples_ms'])==7 and statistics.median(r['samples_ms'])==r['median_ms']
            assert r['semantics']==s and r['correctness_fp32_reference']['verdict']=='accuracy-review-required'
            life=r['output_lifecycle'];assert life['all_7_replay_outputs_equal_direct'] and life['input_unchanged']
            if r['interface']=='native-return':
                assert len(set(life['capture_output_pointers']))==life['distinct_capture_outputs']==20
                assert life['direct_pointer']!=life['second_pointer']
            else:assert life['distinct_capture_outputs']==1
        for interface in ['native-return','supplied-output']:
            t=row['table'][interface];ratios=[];values={s:[] for s in ['new_same_mapping','dao']}
            for trial in [1,2,3]:
                records={r['side']:r for r in row['measurements'] if r['trial']==trial and r['interface']==interface}
                ratios.append(records['dao']['median_ms']/records['new_same_mapping']['median_ms'])
                for side in values:values[side].append(records[side]['median_ms']*1000)
            assert ratios==t['process_paired_dao_over_new_same']
            assert statistics.median(ratios)==t['median_paired_dao_over_new_same']
            assert statistics.median(values['new_same_mapping'])==pytest.approx(t['new_same_us'])
            assert statistics.median(values['dao'])==pytest.approx(t['dao_us'])
    for interface,group in d['aggregate'].items():
        for process in group['processes']:
            trial=process['trial'];oldnew=[]
            for side in ['old_original','new_same_mapping']:
                ratios=[]
                for row in d['rows']:
                    records={r['side']:r for r in row['measurements'] if r['trial']==trial and r['interface']==interface}
                    ratios.append(records['dao']['median_ms']/records[side]['median_ms'])
                    if side=='old_original':oldnew.append(records['old_original']['median_ms']/records['new_same_mapping']['median_ms'])
                g=process['by_side'][side]
                assert g['dao_over_cu_gm']==pytest.approx(math.prod(ratios)**0.1)
                assert (g['wins'],g['losses'],g['ties'])==(sum(x>1 for x in ratios),sum(x<1 for x in ratios),sum(x==1 for x in ratios))
            assert process['old_over_new_same_gm']==pytest.approx(math.prod(oldnew)**0.1)
            for metric,comparison in process['numeric_comparison'].items():
                pairs=[]
                for row in d['rows']:
                    records={r['side']:r for r in row['measurements'] if r['trial']==trial and r['interface']==interface}
                    pairs.append([records['old_original']['correctness_fp32_reference'][metric],records['new_same_mapping']['correctness_fp32_reference'][metric]])
                assert comparison['pairs']==pairs
                assert [comparison[k] for k in ['improved','equal','worsened']]==[sum(b<a for a,b in pairs),sum(a==b for a,b in pairs),sum(b>a for a,b in pairs)]


def test_raw_export_matches_frozen_when_available():
    frozen=json.loads(ARTIFACT.read_text())
    paths=[r['path'] for r in frozen['sources']]
    paths += [r['error_artifact']['path'] for row in frozen['rows'] for r in row['measurements']]
    paths += [r['path'] for r in frozen['fp32_reference_audit']['reference_records']]
    paths += [r['error_artifact']['path'] for r in frozen['fp32_reference_audit']['records']]
    missing=[p for p in paths if not (ROOT/p).is_file()]
    if missing:
        pytest.skip('Raw-evidence recomputation unavailable: '+missing[0]+'; portable checks remain active; skip is not measurement completion')
    assert module.export()==frozen


def test_float32_reference_dense_small_and_portable_precision_labels():
    import numpy as np
    x=np.arange(16,dtype=np.float32).reshape(1,4,4)
    h=np.array([[1,1,1,1],[1,-1,1,-1],[1,1,-1,-1],[1,-1,-1,1]],dtype=np.float32)
    semantics=dict(operator=2,batch=1,shape=[4,4],inverse=False,normalize_inverse=False)
    actual=module.float32_reference(x,semantics)
    assert actual.dtype==np.float32 and np.array_equal(actual,(h@x[0]@h.T)[None])
    semantics.update(inverse=True,normalize_inverse=True)
    assert np.array_equal(module.float32_reference(actual,semantics),x)
    d=json.loads(ARTIFACT.read_text());audit=d['fp32_reference_audit']
    assert d['reference_precision']['primary']=='float32'
    assert d['reference_precision']['higher_precision_diagnostic']=='float64'
    assert 'Earlier FWHT return-contract' in audit['correction']
    assert len(audit['records'])==240 and len(audit['old_new_actual_comparisons'])==60
    assert audit['all_old_new_actual_bytewise_equal']
    for pair in audit['old_new_actual_comparisons']:
        assert pair['elementwise_equal'] and pair['bytewise_equal']
        assert pair['different_elements']==pair['different_bytes']==0
        assert pair['old_actual_sha256']==pair['new_actual_sha256']
    for record in audit['reference_records']:
        assert record['arrays']['source']['dtype']=='float32'
        assert record['arrays']['reference_fp32']['dtype']=='float32'
        assert record['arrays']['reference_fp64_diagnostic']['dtype']=='float64'
    for row in d['rows']:
        for record in row['measurements']:
            assert 'correctness' not in record
            assert record['correctness_fp32_reference']['verdict']=='accuracy-review-required'
            assert record['correctness_fp64_reference_diagnostic']['verdict']=='accuracy-review-required'
            assert record['fp32_reference_error_artifact']['arrays']['reference']['dtype']=='float32'


def test_public_default_mapping_statistics_and_installed_identity():
    support=json.loads((ROOT/'paper/generated/fwht_native_repair_support_20261107.json').read_text())
    d=support['public_default']; summary=d['summary']
    assert summary['single_process_sanity'] and summary['guard_status']['state']=='complete'
    assert len(d['records'])==40 and len(d['plan_checks'])==10

    def normalize(value):
        if isinstance(value,dict):
            result={key:normalize(item) for key,item in value.items()}
            if isinstance(result.get('target'),dict) and 'runtime_fingerprint' in result['target']:
                result['target']['runtime_fingerprint']='<runtime identity>'
            return result
        if isinstance(value,list):return [normalize(item) for item in value]
        return value

    for check in d['plan_checks'].values():
        assert normalize(check['actual_mapping'])==normalize(check['expected_original_mapping'])
        assert check['matches_original_except_target_fingerprint'] and not check['mapping_differences']
    for interface in module.INTERFACES:
        pairs={}
        for record in d['records']:
            if record['interface']!=interface:continue
            timing=record['timing']
            assert len(timing['samples_ms'])==7
            assert statistics.median(timing['samples_ms'])==timing['median_ms']
            assert record['correctness_fp32_reference']['verdict']=='accuracy-review-required'
            assert 'correctness' not in record
            pairs.setdefault(record['case_id'],{})[record['backend']]=record
        assert len(pairs)==10
        ratios=[p['dao']['timing']['median_ms']/p['cubutterfly']['timing']['median_ms'] for p in pairs.values()]
        assert module.gm(ratios)==pytest.approx(summary['geomean_by_interface'][interface])
    assert sum(summary['setup_seconds_by_case'].values())==pytest.approx(summary['setup_seconds_total'])
    main=json.loads(ARTIFACT.read_text())
    assert support['installation']['library_sha256']==main['new_library_sha256']
    assert support['installation']['previous_library']['sha256']==main['old_library_sha256']
    assert support['installation']['runtime_fingerprint']==main['build_receipt']['runtime_fingerprint']
    assert len(d['fp32_reference_audit']['records'])==40
