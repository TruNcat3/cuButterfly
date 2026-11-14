import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import trace_fwht_comparator as trace


@pytest.fixture
def prepared(tmp_path):
    source = Path('/home/wt/git/cuButterfly/results/comparator_extension_20261111/hadacore11')
    if not source.exists():
        pytest.skip('recorded campaign unavailable')
    trace.prepare(source,tmp_path)
    return tmp_path


def test_prepare_fixed_population_and_commands(prepared):
    commands = trace.read(prepared/'commands.json')['commands']
    assert len(commands) == 10
    assert {c['case'] for c in commands} == set(trace.CASES)
    for command in commands:
        assert '--mode timing' in command['timing']
        assert '--clock-control none' in command['ncu']
        assert '--profile-from-start off' in command['ncu']
        assert '--profile-tag ncu' in command['ncu']
        assert trace.METRICS in command['ncu']
        assert '--capture-range cudaProfilerApi' in command['nsys']
        assert '--profile-tag nsys' in command['nsys']


def test_identity_and_payload_binding(prepared):
    binding = trace.read(prepared/'bindings'/f'{trace.CASES[0]}.json')
    trace.validate_binding(binding,prepared)
    changed = copy.deepcopy(binding)
    changed['recorded']['identity']['gpu_name'] = 'tampered'
    with pytest.raises(ValueError,match='checksum'):
        trace.validate_binding(changed,prepared)
    with (prepared/binding['payload']).open('ab') as handle:
        handle.write(b'tamper')
    with pytest.raises(ValueError,match='payload checksum'):
        trace.validate_binding(binding,prepared)


def test_semantic_and_mapping_rejection(prepared):
    binding = trace.read(prepared/'bindings'/f'{trace.CASES[0]}.json')
    altered = copy.deepcopy(binding['case'])
    altered['descriptor']['semantics']['compute'] = 2
    with pytest.raises(ValueError,match='FP16'):
        trace.validate_case(altered)
    binding['recorded']['row']['mapping']['contract']['compute'] = 2
    binding['recorded']['row_sha256'] = trace.campaign.digest(binding['recorded']['row'])
    with pytest.raises(ValueError,match='mapping contract'):
        trace.validate_binding(binding,prepared)


def test_cli_modes():
    p = trace.parser()
    for extra in (['--mode','unknown'],['--case','fwht-fp16-n256-b1']):
        with pytest.raises(SystemExit):
            p.parse_args(['--run','--output','/tmp/unused']+extra)
    args = p.parse_args(['--run','--output','/tmp/unused','--case',trace.CASES[0],'--backend','hadacore','--mode','profile'])
    assert args.mode == 'profile'


def test_thread_override_metadata_and_other_fields_rejected():
    selected = dict(backend='temporal-tile',tile_threads=128,compute_unit='radix4',contract={'compute':0})
    expected, diff = trace.replay_mapping(selected,'cubutterfly',512)
    assert selected['tile_threads'] == 128
    assert diff == {'tile_threads':{'selected':128,'expected':512}}
    metadata = trace.verify_mapping(selected,expected,copy.deepcopy(expected),512)
    assert metadata == dict(selected=selected,expected=expected,actual=expected,diff=diff,
                            threads_override=512,actual_matches_expected=True)
    invalid = copy.deepcopy(expected)
    invalid['compute_unit'] = 'radix2'
    with pytest.raises(AssertionError,match='beyond tile_threads'):
        trace.verify_mapping(selected,invalid,invalid,512)
    with pytest.raises(AssertionError,match='expected mapping'):
        trace.verify_mapping(selected,expected,invalid,512)
    with pytest.raises(ValueError,match='requires cubutterfly'):
        trace.replay_mapping(selected,'hadacore',512)
    assert trace.replay_mapping(selected,'cubutterfly') == (selected,{})


def test_profiler_api_wrapper_without_cuda():
    from types import SimpleNamespace
    calls = []
    cudart = SimpleNamespace(cudaProfilerStart=lambda:calls.append('start'),
                             cudaProfilerStop=lambda:calls.append('stop'))
    torch = SimpleNamespace(cuda=SimpleNamespace(cudart=lambda:cudart))
    trace.profiler_call(torch,'cudaProfilerStart')
    trace.profiler_call(torch,'cudaProfilerStop')
    assert calls == ['start','stop']


def test_target_uuid_and_after_guard_failure(monkeypatch):
    identity = {'gpu_uuid':'0eae5f07-33de-321b-70a9-ffeae49309e6'}
    assert trace.target_uuid(identity) == 'GPU-'+identity['gpu_uuid']
    with pytest.raises(ValueError,match='must match'):
        trace.target_uuid(identity,'GPU-00000000-0000-0000-0000-000000000000')
    def fail(uuid):
        raise RuntimeError('foreign PID')
    monkeypatch.setattr(trace.campaign,'exclusive_gpu',fail)
    report = {'status':'accuracy-review-required','guards':{'before':True}}
    with pytest.raises(RuntimeError):
        trace.after_guard(report,trace.target_uuid(identity))
    assert report['status'] == 'interrupted'
    assert report['guards']['after'] is False


def test_recorded_guard_rejected(prepared):
    binding = trace.read(prepared/'bindings'/f'{trace.CASES[0]}.json')
    binding['recorded']['guards']['after'] = False
    with pytest.raises(ValueError,match='guards'):
        trace.validate_binding(binding,prepared)
