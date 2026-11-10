import hashlib, json, platform, subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'experiments/provenance'
R=ROOT/'results/external_cpu_baselines_20261012'
rows=[json.loads(x) for x in (R/'fxt_cpu_rows.jsonl').read_text().splitlines()]
def digest(p):
    p=Path(p); return {'path':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size}
source_files=[R/'sources/fxt-2026.05.19.tar.gz',R/'sources/fxt_home.html',R/'sources/fxt_walsh_doc.txt',
 R/'sources/hexl_README.md',R/'sources/hexl_ntt.hpp',R/'sources/butterfly_README.md',R/'sources/butterfly_cpu.cpp',
 R/'sources/butterfly_cpu_tree.json',R/'sources/spiral_home.html',R/'sources/spiral_generator.html',
 R/'sources/ntl_fft.h',R/'sources/ntl_zzp_doc.html',R/'cpu_identity.json',R/'fxt_cpu_rows.jsonl',
 ROOT/'experiments/provenance/benchmark_fxt_cpu_20261012.cpp']
sources={str(p.relative_to(ROOT)):digest(p)['sha256'] for p in source_files if p.exists()}
audit={
 'schema':'external-baseline-audit-20261012-v1','human_verification':'not recorded',
 'scope':'CPU-only supplemental baselines; never pooled into GPU speedup or operator GM',
 'sources':[
  {'name':'FXT','url':'https://www.jjj.de/fxt/','release':'fxt-2026.05.19','local':digest(R/'sources/fxt-2026.05.19.tar.gz'),'claims':['Walsh transforms','arithmetic subset/superset transforms','templated Type supports double and integer']},
  {'name':'Intel HEXL','url':'https://github.com/intel/hexl','local':digest(R/'sources/hexl_README.md'),'claims':['uint64 API','negacyclic ring Z_q[X]/(X^N+1)','q up to approximately 62 bits','single-threaded and thread-safe']},
  {'name':'HazyResearch butterfly','url':'https://github.com/HazyResearch/butterfly','release':'master snapshot','local':digest(R/'sources/butterfly_README.md'),'claims':['official C++ CPU extension','floating/complex scalar dispatch','2x2 butterfly factors; CUDA optional']},
  {'name':'SPIRAL','url':'https://www.spiral.net/','local':digest(R/'sources/spiral_home.html'),'claims':['transform code generation and platform-tuned implementations; generator context, no directly runnable contract here']},
  {'name':'NTL','url':'https://libntl.org/','local':digest(R/'sources/ntl_fft.h'),'claims':['FFT-backed polynomial arithmetic; UserFFTInit and FFT prime APIs; not a direct cyclic NTT wrapper']},
 ],
 'contract_matrix':[
  {'operator':'fwht','precision':'fp64','external':'FXT walsh_wak (Walsh-Kronecker)','status':'measured','semantic_notes':'in-place FXT call adapted to out-of-place; self-inverse unnormalized; natural input/output for this call; exact basis must be checked against requested FWHT order'},
  {'operator':'subset-zeta','precision':'uint32','external':'FXT arith_transform_plus','status':'measured','semantic_notes':'arithmetic subset transform; integer addition is unbounded in this wrapper, so it is only exact for non-overflow test values and not a modular uint32 baseline'},
  {'operator':'superset-zeta','precision':'uint32','external':'FXT rev_arith_transform_plus','status':'measured','semantic_notes':'reverse arithmetic transform; same non-modular uint32 caveat'},
  {'operator':'structured-2x2','precision':'fp32','external':'HazyResearch butterfly CPU','status':'path identified; not built this round','semantic_notes':'supports arbitrary 2x2 butterfly factors, but API packaging and exact stage order/stride adapter remain to be implemented'},
  {'operator':'ntt','precision':'word32','external':'Intel HEXL / NTL','status':'no valid direct match','semantic_notes':'HEXL is negacyclic and uint64 API; NTL FFT is polynomial multiplication machinery, not a matching cyclic natural-order word32 transform'},
 ],
 'fxt_protocol':{'compiler':'g++','flags':'-O3 -march=native -std=c++17 -fopenmp','host':'AMD EPYC 7542 32-Core Processor','threads':'OpenMP outer batch parallelism; FXT transform itself serial','warmup_calls':3,'timing':'7 samples; each sample 5 calls; median; includes required out-of-place copy','correctness':'FWHT long-double reference full output; zeta direct subset/superset enumeration at N=256 plus full uint32 transform equality; no GPU'},
 'seed_contract':{'seed':'20773043 + log2(N) + batch','distribution':'FWHT input = (mt19937_64 % 2000001 - 1000000)/1000000 as double; zeta input = mt19937_64 uint32 cast','order':'logical index 0..N-1; batch contiguous; FXT output retained as returned','source':'experiments/provenance/benchmark_fxt_cpu_20261012.cpp'},
 'rows':rows,'source_hashes':sources,
 'limitations':['CPU timings are not GPU comparisons','FXT subset/superset values are uint32 additions without modulus and can overflow for large random values; direct check observed zero mismatch under this run but this does not establish modular semantics','FXT Walsh basis/order needs explicit matched-contract adapter before claiming FWHT equivalence','No current-build external library binary was substituted for a missing GPU baseline'],
}
(OUT/'EXTERNAL_BASELINE_AUDIT_20261012.json').write_text(json.dumps(audit,indent=2)+'\n')

L=['# External CPU baseline audit (2026-10-12)','', 'This is a supplemental CPU-only audit. CPU measurements are not pooled into GPU speedups or operator geometric means. Sources are downloaded from official project/author endpoints and hash-bound in the JSON ledger.', '', '## Coverage', '', '| Missing GPU comparator | Real external implementation | Status | Contract boundary |', '| --- | --- | --- | --- |']
for c in audit['contract_matrix']: L.append(f"| {c['operator']} / {c['precision']} | {c['external']} | {c['status']} | {c['semantic_notes']} |")
L += ['', '## FXT measurement', '', 'FXT release `2026.05.19` was compiled from the downloaded author release. The wrapper calls the library transforms, uses an out-of-place copy adapter, and checks all output values. It adds N=2^20, B=4 at one and four OpenMP outer-batch threads to the earlier N=2^12/2^16 rows (30 rows total).', '', '| Operator | Precision | N | B | Threads | Median us/batch | us/transform | Mismatches | Error check |', '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |']
for r in rows:
    if r['N']>=1048576:
        err=f"relative L2 {r['relative_l2']:.3e}" if 'relative_l2' in r else 'exact integer'
        L.append(f"| {r['operator']} | {r['precision']} | {r['N']} | {r['B']} | {r['threads']} | {r['latency_us']:.3f} | {r['per_transform_us']:.3f} | {r['mismatches']} | {err} |")
L += ['', 'The N=2^20/B=4 results are host measurements on AMD EPYC 7542. FP64 FWHT full-output relative L2 is 2.274e-16 for both thread settings with zero mismatches against a long-double reference. Zeta runs have zero mismatches against the direct subset/superset enumeration check at N=256 and full integer transform reference; the wrapper uses uint32 addition without a modulus, so these are not modular NTT or uint32-wrap claims.', '', '## Reuse and blockers', '', '- FXT is a real, templated transform library and is usable as a CPU algorithm witness for Walsh and arithmetic subset/superset transforms. Basis/order and normalization must be explicitly reconciled before a same-contract comparison; no GPU ratio is inferred.', '- Intel HEXL is a real optimized CPU NTT library, but its documented contract is negacyclic `Z_q[X]/(X^N+1)`, `uint64_t`, and `q == 1 (mod 2N)`. It is therefore not a direct baseline for the current cyclic natural-order word32 request. Its output is bit-reversed and would require a declared adapter.', '- NTL exposes user FFT-prime initialization and FFT polynomial arithmetic. It is a construction component, not a drop-in current word32 cyclic NTT baseline.', '- HazyResearch butterfly has an official CPU C++ implementation for floating/complex 2x2 factors. It is a credible structured-butterfly path, but exact stage matrices, order, strides and packaging were not built in this bounded round.', '- SPIRAL provides generation methodology and platform-tuned transform code generation; no directly executable matching contract was obtained in this round.', '', 'Reproduction command:', '', '```bash', 'g++ -O3 -march=native -std=c++17 -fopenmp -I results/external_cpu_baselines_20261012/fxt/src experiments/provenance/benchmark_fxt_cpu_20261012.cpp -o results/external_cpu_baselines_20261012/benchmark_fxt_cpu', 'OMP_PROC_BIND=close OMP_PLACES=cores CUDA_VISIBLE_DEVICES="" results/external_cpu_baselines_20261012/benchmark_fxt_cpu > results/external_cpu_baselines_20261012/fxt_cpu_rows.jsonl', '```', '', 'See `EXTERNAL_BASELINE_AUDIT_20261012.json` for URLs, raw source hashes, seed contract and every measured row.']
(OUT/'EXTERNAL_BASELINE_AUDIT_20261012.md').write_text('\n'.join(L)+'\n')
print('wrote',len(rows),'rows',len(sources),'hashes')
