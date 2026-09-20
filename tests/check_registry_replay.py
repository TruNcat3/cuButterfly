#!/usr/bin/env python3
"""GPU correctness contract for temporary registry fixtures; no timing evidence."""
import argparse
import csv
import io
import json
import os
import pathlib
import subprocess
import sys
import tempfile

sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/"scripts"))
from hardware_registry import promote


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--build-dir",type=pathlib.Path,required=True)
    parser.add_argument("--output",type=pathlib.Path,required=True)
    args=parser.parse_args()
    rows=[]
    with tempfile.TemporaryDirectory(prefix="cubutterfly-registry-test-") as temporary:
        registry=pathlib.Path(temporary)/"registry.json"
        env={**os.environ,"CUBUTTERFLY_REGISTRY":str(registry),"CUBUTTERFLY_COMPILE_MODE":"precompiled"}
        def run(command):
            result=subprocess.run(list(map(str,command)),env=env,text=True,capture_output=True,check=True)
            return next(csv.DictReader(io.StringIO(result.stdout)))
        butterfly=args.build_dir/"cubutterfly_bench"
        profile=run([butterfly,"--device-identity"])
        for op, precision in (("fft","fp32"),("fwht","fp64"),("structured-2x2","fp32"),
                              ("subset-zeta","uint32"),("superset-zeta","uint32"),("ntt","word32"),("ntt","word64")):
            ntt=op=="ntt"
            command=[args.build_dir/("cuntt_bench" if ntt else "cubutterfly_bench"),"--logN","8","--batch","2",
                     "--verify","--warmup","1","--repeat","2","--csv"]
            if ntt: command += ["--word-bits",precision[4:],"--modulus","2013265921"]
            else:
                command += ["--operator",op,"--precision",precision,"--placement","out-of-place"]
                if op=="structured-2x2": command += ["--stage-matrix","1,0.25,-0.5,1"]
            mapping=["--backend","shared-iterative","--stage-partition","3,5","--compute-unit","radix2"]
            for inverse in (False,True):
                operation=command+(["--inverse"] if inverse else [])
                expected=run(operation+mapping)
                assert expected["correct"]=="1"
                if ntt: expected.update(operator="ntt",precision=precision,placement=expected["output_order"])
                name=f"test-fixture-{op}-{precision}-{inverse}"
                # Synthetic ranking value only, isolated from the real registry.
                candidate=dict(name=name,status="measured",correct=True,median_kernel_ms=1.0,
                               samples=[expected],test_fixture=True)
                promote(registry,profile,{"candidates":[candidate]})
                actual=run(operation+["--auto-select"])
                assert actual["selected_implementation"]==name,(name,actual)
                assert actual["correct"]=="1" and json.loads(actual["mapping_json"])==json.loads(expected["mapping_json"])
                rows.append(dict(operator=op,precision=precision,inverse=inverse,correct=True,full_mapping_matches=True))
        document=json.loads(registry.read_text())
        for target in document["targets"]:
            for record in target["records"]: record["runtime_fingerprint"]="obsolete-fixture"
        registry.write_text(json.dumps(document))
        assert not run(operation+["--auto-select"])["selected_implementation"].startswith("test-fixture-")
    args.output.write_text(json.dumps(dict(kind="correctness-only",cases=rows,obsolete_fingerprint_rejected=True),indent=2)+"\n")
    print(f"{len(rows)} complete registry replays and obsolete-code rejection passed")


if __name__=="__main__": main()
