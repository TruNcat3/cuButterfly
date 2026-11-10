#!/usr/bin/env python3
"""Build extra cuFFTDx online points on demand without changing defaults."""
import argparse, json, pathlib, subprocess

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--build-dir',required=True,type=pathlib.Path)
    p.add_argument('--log-n',required=True,nargs='+',type=int)
    p.add_argument('--pair',required=True,nargs='+',metavar='THREADSxEPT')
    p.add_argument('--source',type=pathlib.Path,default=pathlib.Path('config/v100_fft_codegen.json'))
    a=p.parse_args(); root=pathlib.Path(__file__).resolve().parents[1]
    spec=json.loads(a.source.read_text()); fam=spec.setdefault('families',{}).setdefault('cufftdx-online',{})
    fam['dimension_log_n']=sorted(set(fam.get('dimension_log_n',[]))|set(a.log_n))
    pairs={tuple(x) for x in fam.get('thread_ept_pairs',[])}
    pairs|={tuple(map(int,x.lower().split('x'))) for x in a.pair}
    fam['thread_ept_pairs']=[list(x) for x in sorted(pairs)]
    out=a.build_dir/'on_demand_fft_codegen.json'; out.write_text(json.dumps(spec,indent=2)+'\n')
    subprocess.run(['cmake','-S',str(root),'-B',str(a.build_dir),f'-DCUBUTTERFLY_FFT_CODEGEN_SPEC={out}'],check=True)
    subprocess.run(['cmake','--build',str(a.build_dir),'-j2'],check=True)
    print(f'compiled on-demand FFT variant spec: {out}')
if __name__=='__main__': main()
