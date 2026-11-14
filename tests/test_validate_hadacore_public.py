from pathlib import Path
import sys
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import validate_hadacore_public as p


def setup(tmp_path,native_ms=.02,mma_ms=.01):
    lib=tmp_path/'lib.so';lib.write_bytes(b'main')
    error=tmp_path/'error.npz';error.write_bytes(b'errors')
    rows=[]
    for name,ms in [('native',native_ms),('mma',mma_ms)]:
        evidence=tmp_path/(name+'.json')
        p.write(evidence,dict(gpu_uuid='GPU-test',guards={'before':True,'after':True},
            timing={'median_ms':ms,'protocol':'shared'},error_artifact=str(error),error_artifact_sha256=p.sha(error)))
        rows.append(dict(log_n=8,batch=1,median_ms=ms,timing_protocol='shared',
            evidence_path=str(evidence),evidence_sha256=p.sha(evidence),library_sha256=p.sha(lib),
            status='validated-exact-upstream',mapping={'tile_threads':128},chunks=1,groups_per_block=1))
    profile=tmp_path/'profile.json';p.write(profile,dict(gpu_uuid='GPU-test',rows=[rows[1]],native_rows=[rows[0]]))
    return lib,profile,rows


def test_selector_prefers_best_same_protocol_and_native_ties(tmp_path):
    lib,profile,rows=setup(tmp_path)
    selected=p.winner(profile,256,1,lib);assert selected['kind']=='mma'
    p.check_mapping({'local_core':{'name':'hadacore-local-tile','chunks':1,'groups_per_block':1}},selected,'')
    with pytest.raises(AssertionError,match='geometry'):
        p.check_mapping({'local_core':{'name':'hadacore-local-tile','chunks':1,'groups_per_block':4}},selected,'')
    lib,profile,rows=setup(tmp_path,.01,.01)
    assert p.winner(profile,256,1,lib)['kind']=='native'


def test_missing_pair_is_explicit_fallback(tmp_path):
    lib,profile,rows=setup(tmp_path)
    r=p.read(profile);r['native_rows']=[];p.write(profile,r)
    selected=p.winner(profile,256,1,lib);assert selected['kind']=='fallback'
    with pytest.raises(AssertionError,match='reason'):p.check_mapping({},selected,'')
    p.check_mapping({},selected,'no matching calibrated native/MMA pair; retained native selection')


def test_changed_evidence_rejected(tmp_path):
    lib,profile,rows=setup(tmp_path)
    Path(rows[1]['evidence_path']).write_text('{}')
    with pytest.raises(ValueError,match='evidence changed'):p.winner(profile,256,1,lib)


def test_native_mapping_only_permits_arithmetic_policy_change():
    selected=dict(kind='native',row={'mapping':{'tile_threads':128}})
    p.check_mapping({'tile_threads':128,'arithmetic_policy':'allow-fp16-mma'},selected,'')
    with pytest.raises(AssertionError):p.check_mapping({'tile_threads':256},selected,'')
