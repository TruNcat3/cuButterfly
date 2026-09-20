#!/usr/bin/env python3
"""Compile one execution mapping into a standalone module with a versioned C ABI.

This never invokes CMake or relinks cuButterfly. Cache identity covers the
template source, mapping, compiler, GPU architecture and MathDx headers.
"""
import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import pathlib
import subprocess
import tempfile
import time

try:
    import resident_mapping
except ImportError:  # pragma: no cover - package import fallback
    from . import resident_mapping


@contextlib.contextmanager
def compilation_budget(timeout):
    """One installation shares an independent NVCC budget across subprocesses."""
    path = os.environ.get("CUBUTTERFLY_COMPILE_BUDGET_FILE")
    limit = float(os.environ.get("CUBUTTERFLY_COMPILE_BUDGET_SECONDS", "0"))
    if not path or not limit:
        yield None if timeout == 0 else timeout
        return
    with open(path, "a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        handle.seek(0)
        spent = float(handle.read() or "0")
        if spent >= limit:
            raise RuntimeError("installation specialization compilation budget exhausted")
        remaining = limit-spent
        started = time.monotonic()
        try:
            yield min(timeout, remaining) if timeout else remaining
        finally:
            handle.seek(0); handle.truncate()
            handle.write(str(spent + time.monotonic()-started)); handle.flush()


def register_prefix_geometry(point):
    """Resolve physical CTA shape before rendering or accepting a cache hit."""
    geometry = resident_mapping.register_geometry(point)
    return (geometry["local"], geometry["total"], geometry["suffix"],
            geometry["prefix_columns"], geometry["prefix_lanes"])


def render_register_fft(point, sm, identity):
    geometry = resident_mapping.register_geometry(point)
    local, total, suffix = geometry["local"], geometry["total"], geometry["suffix"]
    pc, lanes = geometry["prefix_columns"], geometry["prefix_lanes"]
    ept, sc = geometry["suffix_ept"], geometry["suffix_columns"]
    suffix_chunk = geometry["suffix_chunk"]
    fp64 = point.get("precision") == "fp64"
    real, complex_type, width = ("double", "Complex64", 16) if fp64 else ("float", "Complex32", 8)
    prefix_codelet = point.get("prefix_codelet", "native")
    prefix_layout = point.get("prefix_shared_layout", "linear")
    if prefix_codelet not in ("native", "cufftdx-thread"):
        raise ValueError("register prefix_codelet must be native or cufftdx-thread")
    if prefix_layout not in ("linear", "xor"):
        raise ValueError("register prefix_shared_layout must be linear or xor")
    if fp64 and prefix_codelet == "cufftdx-thread":
        raise ValueError("cufftdx-thread register prefix is only supported for FP32")
    if any(x <= 0 or x & (x-1) for x in (pc, sc, ept)) or sc > 16:
        raise ValueError("grouped register FFT requires power-of-two column counts and EPT; suffix columns <=16")
    prefix_header = '#include "fft_thread_codelet.cuh"\n' if prefix_codelet == "cufftdx-thread" else ""
    prefix_codelet_type = "register_tile::DxCodelet" if prefix_codelet == "cufftdx-thread" else "register_tile::NativeCodelet"
    prefix_xor = "true" if prefix_layout == "xor" else "false"
    # A function launch can deduce Complex from its pointer argument, but a
    # cudaFuncGetAttributes expression cannot. Keep the query explicit so an
    # FP64 native prefix never aliases the default Complex32 kernel.
    prefix_resource_args = f",{prefix_codelet_type},Complex,{prefix_xor}"
    if lanes != 1:
        prefix_resource_args += f",{lanes}"
    prefix_launch_args = "" if prefix_codelet == "native" and prefix_layout == "linear" and lanes == 1 else prefix_resource_args
    parts = geometry["local_stage_partitions"]
    rectangular = bool(parts and parts[0])
    if rectangular:
        log_a, log_b = geometry["log_a"], geometry["log_b"]
        rectangular_resource_args = f",{prefix_codelet_type},Complex,{prefix_xor}"
        rectangular_launch_args = "" if prefix_codelet == "native" and prefix_layout == "linear" else rectangular_resource_args
        prefix_resource = f"register_tile::rectangular_prefix<{log_a},{log_b},{geometry['prefix_ept']},{pc},Inverse{rectangular_resource_args}>"
        prefix_launch = f"register_tile::launch_rectangular_prefix<{log_a},{log_b},{geometry['prefix_ept']},{pc},Inverse{rectangular_launch_args}>"
    else:
        prefix_resource = f"register_tile::prefix<{local},{pc},Inverse{prefix_resource_args}>"
        prefix_launch = f"register_tile::launch_prefix<{local},{pc},Inverse{prefix_launch_args}>"
    suffix_resource = f"grouped_suffix<Suffix<Inverse>,{sc},Complex"
    suffix_launch = f"launch_grouped_suffix<{suffix},{ept},{sc},Inverse"
    if suffix_chunk:
        # Specialize the JIT-known local extent only for chunked writeback.
        # Whole-tile writeback retains its measured incumbent lowering.
        suffix_resource += f",{suffix_chunk},{local}"
        suffix_launch += f",Complex,{suffix_chunk},{local}"
    suffix_resource += ">"
    suffix_launch += ">"
    suffix_tile = (f"(1U<<{suffix})*{sc}*{width}"
                   if not suffix_chunk else
                   f"(1U<<{suffix})*{sc}*{width}*{suffix_chunk}/{ept}")
    return f'''#include <cubutterfly/module.h>
#include "fft_register_tile.cuh"
{prefix_header}#include "fft_grouped_suffix.cuh"
using namespace cuntt;
using namespace cuntt::detail;
using Complex = {complex_type};
template<bool Inverse> using Suffix = decltype(cufftdx::Block()+cufftdx::Size<(1U<<{suffix})>()+
 cufftdx::Type<cufftdx::fft_type::c2c>()+cufftdx::Direction<Inverse ? cufftdx::fft_direction::inverse : cufftdx::fft_direction::forward>()+
 cufftdx::Precision<{real}>()+cufftdx::ElementsPerThread<{ept}>()+cufftdx::FFTsPerBlock<{sc}>()+cufftdx::SM<{sm*10}>());
template<bool Inverse> int resources_impl(uint32_t group, cubutterflyModuleResourcesV1* out, uint64_t* local=nullptr) {{
 if (!out || group>1) return int(cudaErrorInvalidValue);
 cudaFuncAttributes a{{}}; cudaError_t status;
 if (group==0) {{
  status=cudaFuncGetAttributes(&a,{prefix_resource});
  out->threads={point['prefix_threads']}; out->dynamic_shared_bytes=(1U<<{local})*{pc}*{width};
 }} else {{
  status=cudaFuncGetAttributes(&a,{suffix_resource});
  out->threads=Suffix<Inverse>::max_threads_per_block;
  constexpr unsigned tile={suffix_tile};
  out->dynamic_shared_bytes={sc}==1 ? Suffix<Inverse>::shared_memory_size :
   (tile>Suffix<Inverse>::shared_memory_size ? tile : Suffix<Inverse>::shared_memory_size);
 }}
 out->registers_per_thread=a.numRegs; out->static_shared_bytes=a.sharedSizeBytes;
 if(local) *local=a.localSizeBytes;
 return int(status);
}}
static int resources(int inverse,uint32_t group,cubutterflyModuleResourcesV1* out) {{
 return inverse ? resources_impl<true>(group,out) : resources_impl<false>(group,out);
}}
extern "C" int cubutterfly_module_local_bytes_v1(int inverse,uint32_t group,uint64_t* local) {{
 if(!local) return int(cudaErrorInvalidValue);
 cubutterflyModuleResourcesV1 r{{}};
 return inverse ? resources_impl<true>(group,&r,local) : resources_impl<false>(group,&r,local);
}}
template<bool Inverse> int execute_group(uint32_t group,const cubutterflyModuleInvocationV1& a) {{
 if(group==0) {{
  {prefix_launch}(static_cast<const Complex*>(a.input),static_cast<Complex*>(a.output),
   {total},a.batch,a.batch_stride,a.element_stride,a.stream);
 }} else {{
  {suffix_launch}(static_cast<const Complex*>(a.input),static_cast<Complex*>(a.output),
   {total},a.batch,a.stream,a.batch_stride,a.element_stride,a.normalize_inverse);
 }}
 return int(cudaGetLastError());
}}
extern "C" int cubutterfly_module_launch_group_v1(uint32_t group,const cubutterflyModuleInvocationV1* a) {{
 if(group>1 || !a || !a->input || !a->output || !a->batch) return int(cudaErrorInvalidValue);
 return a->inverse ? execute_group<true>(group,*a) : execute_group<false>(group,*a);
}}
static int launch(const cubutterflyModuleInvocationV1* a) {{
 if(!a || !a->input || !a->output || !a->workspace) return int(cudaErrorInvalidValue);
 auto args=*a; args.output=a->workspace;
 auto status=cubutterfly_module_launch_group_v1(0,&args); if(status!=cudaSuccess) return status;
 args.input=a->workspace; args.output=a->output;
 return cubutterfly_module_launch_group_v1(1,&args);
}}
extern "C" const cubutterflyModuleV1* cubutterfly_module_v1() {{
 static const cubutterflyModuleV1 module{{1,sizeof(cubutterflyModuleV1),"{identity}",{sm},2,resources,launch}};
 return &module;
}}
'''


def normalize_factor_io_policies(raw_policies, factor_count):
    """Validate and expand the per-factor I/O lowering policy."""
    if raw_policies is None or not isinstance(raw_policies, (list, tuple)):
        raise ValueError("factor_io_policies must be a list with one entry per factor")
    if raw_policies and len(raw_policies) != factor_count:
        raise ValueError("factor_io_policies must be empty or match factor_partition length")
    policies = list(raw_policies) if raw_policies else ["dynamic"] * factor_count
    if any(policy not in ("dynamic", "static-unrolled") for policy in policies):
        raise ValueError("factor_io_policies entries must be dynamic or static-unrolled")
    return policies


def render_factor_fft(point, sm, identity):
    total = int(point["logN"])
    macro = point.get("stage_partition") or [total]
    factors = point.get("factor_partition") or macro
    ept, columns = int(point.get("factor_ept", 16)), int(point.get("factor_columns", 8))
    tiles, depth = int(point.get("data_tiles_per_cta", 1)), int(point.get("prefetch_depth", 0))
    slices = int(point.get("factor_slices", 1))
    io_policies = normalize_factor_io_policies(point.get("factor_io_policies", []), len(factors))
    fp64 = point.get("precision") == "fp64"
    core=point.get("fft_core","cufftdx-block")
    if core not in ("cufftdx-block","register-tile"): raise ValueError("unsupported factor core")
    if point.get("precision") not in ("fp32", "fp64") or not 1 <= total <= 30:
        raise ValueError("factor FFT requires FP32/64 logN 1..30")
    if sum(macro) != total or sum(factors) != total or any(s < 1 for s in macro) or any(s < 1 or s > 14 for s in factors):
        raise ValueError("positive macro/factor partitions must cover logN; local factor <=14")
    ends, done = set(), 0
    for s in factors:
        done += s; ends.add(done)
    done = 0
    for s in macro:
        done += s
        if done not in ends: raise ValueError("factor crosses macro boundary")
    if tiles < 1 or depth not in range(5) or (depth and sm < 80):
        raise ValueError("positive data tiles and prefetch depth 0..4 required; async needs SM80+")
    if any(x < 1 or x & (x-1) for x in (ept, columns)) or ept > 32 or columns > (8 if fp64 else 16):
        raise ValueError("invalid factor EPT or columns")
    if slices<1 or slices&(slices-1) or (slices>1 and (len(factors)<2 or slices>(1<<factors[-1])//columns)):
        raise ValueError("factor slices must cover complete tiles within the last unprocessed digit")
    real, complex_type = ("double", "Complex64") if fp64 else ("float", "Complex32")
    aliases, resources, launches, range_limits = [], [], [], []
    done = 0
    for group, log in enumerate(factors):
        if core=="register-tile" and (log%2 or ept!=(1<<(log//2))):
            raise ValueError("native factor core requires even-log factors and EPT=sqrt(factor size)")
        size = 1 << log
        threads = size // ept * columns
        if size < ept or threads < 32 or threads > 1024 or threads % 32 or columns > (1 << (total-log)) or (done and columns > (1 << done)):
            raise ValueError("invalid factor shape, EPT or column grouping")
        static_io = io_policies[group] == "static-unrolled"
        if static_io and (size % ept or threads * ept != size * columns):
            raise ValueError("static-unrolled factor I/O requires an integral EPT/block shape")
        aliases.append(f'''template<bool Inverse> using F{group}=decltype(cufftdx::Block()+cufftdx::Size<{size}>()+
 cufftdx::Type<cufftdx::fft_type::c2c>()+cufftdx::Direction<Inverse ? cufftdx::fft_direction::inverse : cufftdx::fft_direction::forward>()+
 cufftdx::Precision<{real}>()+cufftdx::ElementsPerThread<{ept}>()+cufftdx::FFTsPerBlock<{columns}>()+cufftdx::SM<{sm*10}>());''')
        if core=="register-tile":
            aliases[-1]=f"template<bool Inverse> using F{group}=factor_streamed::NativeFactor<{log},{ept},{columns},Inverse,Complex>;"
        partial=slices>1 and group+1<len(factors)
        if static_io:
            policy_args = f",{str(partial).lower()},true"
        else:
            policy_args = f",{str(partial).lower()}" if partial else ""
        kernel = f"factor_streamed::kernel<F{group}<Inverse>,{total},{log},{done},{columns},{tiles},{depth},Inverse,Complex{policy_args}>"
        shared = f"factor_streamed::shared_bytes<F{group}<Inverse>,{columns},{depth},Complex>()"
        resources.append(f'''case {group}: {{
 cudaFuncAttributes a{{}}; auto status=cudaFuncGetAttributes(&a,{kernel}); if(status) return int(status);
 constexpr unsigned bytes={shared};
 if constexpr(bytes>48*1024) {{ status=cudaFuncSetAttribute({kernel},cudaFuncAttributeMaxDynamicSharedMemorySize,bytes); if(status) return int(status); }}
 out->threads={threads}; out->registers_per_thread=a.numRegs;
 out->dynamic_shared_bytes=bytes; out->static_shared_bytes=a.sharedSizeBytes;
 if(local) *local=a.localSizeBytes; return 0;
 }}''')
        launches.append(f'''case {group}: {{
 const uint64_t count=r ? r->tile_count : (a.batch<<{total-log})/{columns};
 {kernel}<<<static_cast<unsigned>((count+{tiles}-1)/{tiles}),F{group}<Inverse>::block_dim,{shared},a.stream>>>(
 static_cast<const Complex*>(a.input),static_cast<Complex*>(a.output),a.batch,a.batch_stride,a.element_stride,a.normalize_inverse,
 r ? r->tile_first : 0,count,r ? r->tile_period : count,r ? r->tile_span : count);
 return int(cudaGetLastError());
 }}''')
        range_limits.append(f'''case {group}: full=(r->io.batch<<{total-log})/{columns}; partial={str(partial).lower()}; break;''')
        done += log
    return f'''#include <cubutterfly/module.h>
#include "{'fft_factor_native.cuh' if core=='register-tile' else 'fft_factor_streamed.cuh'}"
using namespace cuntt; using namespace cuntt::detail;
using Complex={complex_type};
{chr(10).join(aliases)}
template<bool Inverse> int query(uint32_t group,cubutterflyModuleResourcesV1* out,uint64_t* local=nullptr) {{
 if(!out) return int(cudaErrorInvalidValue);
 switch(group) {{ {chr(10).join(resources)} default: return int(cudaErrorInvalidValue); }}
}}
static int resources(int inverse,uint32_t group,cubutterflyModuleResourcesV1* out) {{
 return inverse ? query<true>(group,out) : query<false>(group,out);
}}
extern "C" int cubutterfly_module_local_bytes_v1(int inverse,uint32_t group,uint64_t* local) {{
 if(!local) return int(cudaErrorInvalidValue); cubutterflyModuleResourcesV1 r{{}};
 return inverse ? query<true>(group,&r,local) : query<false>(group,&r,local);
}}
template<bool Inverse> int execute(uint32_t group,const cubutterflyModuleInvocationV1& a,const cubutterflyModuleTileRangeV1* r=nullptr) {{
 switch(group) {{ {chr(10).join(launches)} default: return int(cudaErrorInvalidValue); }}
}}
extern "C" int cubutterfly_module_launch_group_v1(uint32_t group,const cubutterflyModuleInvocationV1* a) {{
 if(!a || !a->input || !a->output || !a->batch || !a->element_stride || !a->batch_stride) return int(cudaErrorInvalidValue);
 return a->inverse ? execute<true>(group,*a) : execute<false>(group,*a);
}}
extern "C" int cubutterfly_module_launch_range_v1(uint32_t group,const cubutterflyModuleTileRangeV1* r) {{
 if(!r || !r->io.input || !r->io.output || !r->io.batch || !r->io.element_stride || !r->io.batch_stride ||
    !r->tile_count || !r->tile_span || r->tile_period<r->tile_span) return int(cudaErrorInvalidValue);
 uint64_t full=0; bool partial=false;
 switch(group) {{ {chr(10).join(range_limits)} default: return int(cudaErrorInvalidValue); }}
 if(r->tile_first>=full) return int(cudaErrorInvalidValue);
 const auto quotient=(r->tile_count-1)/r->tile_span, remainder=(r->tile_count-1)%r->tile_span;
 if(quotient>(full-1-r->tile_first)/r->tile_period ||
    remainder>full-1-r->tile_first-quotient*r->tile_period) return int(cudaErrorInvalidValue);
 if(!partial && (r->tile_first || r->tile_count!=full || r->tile_span!=full || r->tile_period!=full))
    return int(cudaErrorInvalidValue);
 return r->io.inverse ? execute<true>(group,r->io,r) : execute<false>(group,r->io,r);
}}
static int launch(const cubutterflyModuleInvocationV1* a) {{
 if(!a || ({len(factors)}>1 && !a->workspace)) return int(cudaErrorInvalidValue);
 const uint64_t extent=(a->batch-1)*a->batch_stride+((uint64_t(1)<<{total})-1)*a->element_stride+1;
 auto args=*a;
 for(unsigned group=0;group<{len(factors)};++group) {{
  args.output=group+1=={len(factors)} ? a->output : static_cast<Complex*>(a->workspace)+(group%2)*extent;
  auto status=cubutterfly_module_launch_group_v1(group,&args); if(status) return status;
  args.input=args.output;
 }} return 0;
}}
extern "C" const cubutterflyModuleV1* cubutterfly_module_v1() {{
 static const cubutterflyModuleV1 api{{1,sizeof(cubutterflyModuleV1),"{identity}",{sm},{len(factors)},resources,launch}};
 return &api;
}}
'''


def render_shared(point, sm, identity):
    """Specialize the same trait and layout kernel used by precompiled plans."""
    log_n = int(point["logN"])
    partition = [int(s) for s in point["stage_partition"]]
    threads = int(point["threads"])
    if not partition or sum(partition) != log_n or any(s <= 0 or s > 15 for s in partition):
        raise ValueError("shared template partition must cover logN with positive capacity-bounded groups")
    if not 1 <= log_n <= 30 or threads < 32 or threads > 1024 or threads % 32:
        raise ValueError("invalid shared template shape or launch")
    operator, precision = point["operator"], point["precision"]
    resident_parts, resident_chunks = resident_mapping.validate_resident_axes(
        point, partition, backend="shared-iterative", operator=operator)
    output_order = point.get("output_order", "natural")
    if output_order not in ("natural", "bit-reversed"):
        raise ValueError("shared JIT output_order must be natural or bit-reversed")
    if operator != "ntt" and output_order != "natural":
        raise ValueError("bit-reversed output_order is only supported for NTT shared JIT")
    static_output_order = {"natural": 0, "bit-reversed": 1}[output_order]
    native = "true" if point.get("accumulation", "native") == "native" else "false"
    low = precision in ("fp16", "bf16")
    real = {"fp64":"double", "fp32":"float", "fp16-fp32":"float", "fp16":"Fp16", "bf16":"Bf16",
            "word32":"std::uint32_t", "word64":"std::uint64_t", "uint32":"std::uint32_t"}[precision]
    multiply = "ComplexMultiply::Gauss3" if point.get("complex_multiply") == "gauss3" else "ComplexMultiply::FourMul"
    if operator in ("subset-zeta", "superset-zeta", "xor-zeta"):
        if precision != "uint32": raise ValueError("zeta traits require uint32")
        trait, init = f"BooleanZetaOperator<{'true' if operator == 'superset-zeta' else 'false'}>", "bool(a.io.inverse)"
    elif operator == "fwht":
        trait = f"LowFwhtOperator<{real},{native}>" if low else f"FwhtOperator<{real}>"
        init = "bool(a.io.inverse),bool(a.io.normalize_inverse)"
    elif operator == "structured-2x2":
        trait = f"LowStructured2x2Operator<{real},{native}>" if low else f"Structured2x2Operator<{real}>"
        init = f"static_cast<const DeviceMatrix2x2<{real}>*>(a.coefficients),bool(a.io.inverse),bool(a.io.normalize_inverse)"
    elif operator == "fft":
        complex_type = {"fp64":"Complex64", "fp32":"Complex32", "fp16-fp32":"Complex32", "fp16":"Complex16", "bf16":"ComplexBf16"}[precision]
        trait = f"LowFftOperator<{real},{native}>" if low else f"FftOperator<{complex_type},{real}>"
        init = f"static_cast<const {complex_type}*>(a.coefficients),bool(a.io.inverse),bool(a.io.normalize_inverse),{multiply}"
    elif operator == "ntt":
        if precision not in ("word32", "word64"): raise ValueError("NTT traits require word32/64")
        trait = f"NttStageOperatorT<{real}>"
        init = f"static_cast<const {real}*>(a.coefficients),static_cast<const {real}*>(a.coefficients_aux),static_cast<{real}>(a.modulus),static_cast<{real}>(a.scale),static_cast<{real}>(a.scale_shoup)"
    else:
        raise ValueError("unsupported mathematical operator")
    aligned = int(point.get("writer_aligned", True))
    batch_tile = int(point.get("batch_tile", 1))
    if not 1 <= batch_tile <= 4: raise ValueError("shared batch tile must be in [1,4]")
    queries, launches, group_launches = [], [], []
    first = 0
    for group, stages in enumerate(partition):
        local_parts = resident_parts[group] if resident_parts else []
        if local_parts:
            kernel = (f"shared_resident_kernel<Operator,{log_n},{first},{stages},"
                      f"{aligned},{static_output_order},{','.join(map(str, local_parts))}>")
        else:
            kernel = f"shared_iterative_kernel<Operator,{log_n},{first},{stages},{aligned},{static_output_order}>"
        queries.append(f'''case {group}: {{
 constexpr size_t bytes=2ULL*(1ULL<<{stages})*sizeof(Value);
 auto status=cudaFuncGetAttributes(&attr,{kernel});
 if(status==cudaSuccess && bytes>48*1024) status=cudaFuncSetAttribute({kernel},cudaFuncAttributeMaxDynamicSharedMemorySize,bytes);
 out->dynamic_shared_bytes=bytes; out->threads={threads};
 out->registers_per_thread=attr.numRegs; out->static_shared_bytes=attr.sharedSizeBytes;
 return int(status); }}''')
        destination = "static_cast<Value*>(a.io.output)" if group+1 == len(partition) else f"static_cast<Value*>(a.io.workspace)+{group%2}*extent"
        group_launches.append(f'''case {group}: {{
 const auto blocks=((a.io.batch+{batch_tile}-1)/{batch_tile})*(1ULL<<{log_n-stages});
 {kernel}<<<static_cast<unsigned>(blocks),{threads},2ULL*(1ULL<<{stages})*sizeof(Value),a.io.stream>>>(
 static_cast<const Value*>(a.io.input),static_cast<Value*>(a.io.output),{log_n},{first},{stages},
 a.io.batch_stride,a.io.element_stride,{aligned},a.io.batch,{batch_tile},op);
 return int(cudaGetLastError()); }}''')
        launches.append(f'''{{ auto* destination={destination};
 auto group_args=a; group_args.io.input=source; group_args.io.output=destination;
 auto status=cubutterfly_module_launch_group_v2({group},&group_args);
 if(status!=cudaSuccess) return status; source=destination; }}''')
        first += stages
    needs_coefficients = operator in ("fft", "ntt", "structured-2x2")
    resident_include = '#include "shared_resident.cuh"\n' if any(resident_parts) else ""
    return f'''#include <cubutterfly/module.h>
#include "operator_traits.cuh"
#include "ntt_traits.cuh"
#include "shared_iterative.cuh"
{resident_include}
using namespace cuntt;
using namespace cuntt::traits;
using namespace cuntt::detail;
using Operator={trait};
using Value=Operator::Value;
static int resources(int,uint32_t group,cubutterflyModuleResourcesV1* out) {{
 if(!out) return int(cudaErrorInvalidValue); cudaFuncAttributes attr{{}};
 switch(group) {{ {''.join(queries)} default: return int(cudaErrorInvalidValue); }}
}}
extern "C" int cubutterfly_module_launch_group_v2(uint32_t group,const cubutterflyModuleInvocationV2* invocation) {{
 if(!invocation) return int(cudaErrorInvalidValue); const auto& a=*invocation;
 if(!a.io.input || !a.io.output || !a.io.batch ||
    ({int(needs_coefficients)} && !a.coefficients) || ({int(operator=='ntt')} && !a.coefficients_aux)) return int(cudaErrorInvalidValue);
 const Operator op{{{init}}};
 switch(group) {{ {''.join(group_launches)} default: return int(cudaErrorInvalidValue); }}
}}
static int launch(const cubutterflyModuleInvocationV2* invocation) {{
 if(!invocation) return int(cudaErrorInvalidValue); const auto& a=*invocation;
 if(!a.io.batch || ({int(len(partition)>1)} && !a.io.workspace)) return int(cudaErrorInvalidValue);
 const size_t extent=(a.io.batch-1)*a.io.batch_stride+((1ULL<<{log_n})-1)*a.io.element_stride+1;
 const auto* source=static_cast<const Value*>(a.io.input);
 {''.join(launches)}
 return int(cudaSuccess);
}}
extern "C" const cubutterflyModuleV2* cubutterfly_module_v2() {{
 static const cubutterflyModuleV2 module{{2,sizeof(cubutterflyModuleV2),"{identity}",{sm},{len(partition)},resources,launch}};
 return &module;
}}
'''


def compile_mapping(point, *, root, mathdx, nvcc, sm, cache, timeout=600):
    shared = point.get("backend") == "shared-iterative"
    factor = point.get("backend") == "factor-streamed"
    if not shared and not factor and (point.get("fft_core") != "register-tile" or point.get("precision") not in ("fp32", "fp64")):
        raise ValueError("this module template implements FP32/64 register-tile FFT; other templates are separate lowerings")
    lanes = point.get("prefix_codelet_lanes", 1)
    if type(lanes) is not int or lanes < 1 or lanes & (lanes-1):
        raise ValueError("prefix_codelet_lanes must be an integer power of two")
    if shared or factor:
        if lanes != 1:
            raise ValueError("prefix_codelet_lanes only applies to online-reorder register-tile FFT")
        if shared:
            resident_mapping.validate_resident_axes(
                point, point.get("stage_partition", []),
                backend="shared-iterative", operator=point.get("operator", "fft"))
        elif resident_mapping.resident_requested(point):
            raise ValueError("resident axes are unsupported for factor-streamed")
    else:
        register_prefix_geometry(point)
    factor_io_policies = None
    if factor:
        factor_shape = point.get("factor_partition") or point.get("stage_partition") or [point.get("logN")]
        factor_io_policies = normalize_factor_io_policies(
            point.get("factor_io_policies", []), len(factor_shape))
    compiler = subprocess.run([str(nvcc), "--version"], text=True, capture_output=True, check=True).stdout
    # Keep effective lowering choices in the cache key even when callers rely
    # on their defaults.  This prevents a selector replay from aliasing a
    # module built with a different physical codelet, layout, or I/O policy.
    cache_point = dict(point)
    # The C++ JIT serializer always emits both public resident axes, including
    # empty arrays for the historical lowering.  Canonicalize the Python-side
    # cache point to keep precompiled and runtime module identities identical.
    if shared or not factor:
        cache_point.setdefault("local_stage_partitions", [])
        cache_point.setdefault("exchange_chunks", [])
    if not shared and not factor:
        cache_point.setdefault("prefix_codelet", "native")
        cache_point.setdefault("prefix_shared_layout", "linear")
        cache_point.setdefault("prefix_codelet_lanes", 1)
    if factor:
        cache_point["factor_io_policies"] = factor_io_policies
    digest = hashlib.sha256(json.dumps(dict(point=cache_point, sm=sm, compiler=compiler, abi=2 if shared else 1), sort_keys=True).encode())
    files = list((root / "include").rglob("*.h")) + list((root / "include").rglob("*.hpp"))
    resident = bool(point.get("local_stage_partitions")) and any(point.get("local_stage_partitions"))
    templates = (("operator_traits.cuh", "ntt_traits.cuh", "shared_iterative.cuh", "shared_resident.cuh")
                 if resident else ("operator_traits.cuh", "ntt_traits.cuh", "shared_iterative.cuh")) if shared else ("fft_register_tile.cuh", "fft_grouped_suffix.cuh")
    if factor: templates = ("fft_register_tile.cuh", "fft_factor_streamed.cuh")
    if factor and point.get("fft_core")=="register-tile": templates += ("fft_factor_native.cuh",)
    if not shared and not factor and point.get("prefix_codelet", "native") == "cufftdx-thread":
        templates += ("fft_thread_codelet.cuh",)
    files += [root / "src" / name for name in templates]
    files += [pathlib.Path(__file__), pathlib.Path(__file__).with_name("resident_mapping.py")]
    if not shared: files += list((mathdx / "include").rglob("*.hpp"))
    for path in sorted(files):
        try:
            label = str(path.relative_to(root))
        except ValueError:
            label = path.name
        digest.update(label.encode())
        digest.update(path.read_bytes())
    identity = digest.hexdigest()
    directory = cache / identity
    directory.mkdir(parents=True, exist_ok=True)
    module = directory / "module.so"
    with (directory / "compile.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if module.exists():
            return module
        source = render_shared(point, sm, identity) if shared else (render_factor_fft(point, sm, identity) if factor else render_register_fft(point, sm, identity))
        with tempfile.TemporaryDirectory(prefix="compile-", dir=directory) as staging:
            staging = pathlib.Path(staging)
            unit = staging / "module.cu"
            unit.write_text(source)
            command = [str(nvcc), "-std=c++17", "-O3", "--shared", "-Xcompiler=-fPIC", f"-arch=sm_{sm}",
                       f"-DCUBUTTERFLY_CUFFTDX_SM={sm*10}", "-I"+str(root/"include"), "-I"+str(root/"src"),
                       "-I"+str(mathdx/"include"), str(unit), "-o", str(staging/"module.so")]
            started = time.monotonic()
            with compilation_budget(timeout) as remaining:
                completed = subprocess.run(command, text=True, capture_output=True, timeout=remaining)
            (directory/"compiler.log").write_text(completed.stdout+completed.stderr)
            if completed.returncode:
                raise RuntimeError(f"specialization compilation failed; see {directory/'compiler.log'}")
            manifest = dict(schema="cubutterfly-module-v1", identity=identity, mapping=point, sm=sm,
                            compiler=compiler, command=command, compile_seconds=time.monotonic()-started,
                            module_sha256=hashlib.sha256((staging/"module.so").read_bytes()).hexdigest())
            (directory/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
            os.replace(staging/"module.so", module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mapping", required=True, help="JSON execution mapping")
    parser.add_argument("--root", type=pathlib.Path, required=True)
    parser.add_argument("--mathdx", type=pathlib.Path, required=True)
    parser.add_argument("--nvcc", type=pathlib.Path, required=True)
    parser.add_argument("--sm", type=int, required=True)
    parser.add_argument("--cache", type=pathlib.Path, required=True)
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()
    print(compile_mapping(json.loads(args.mapping), root=args.root, mathdx=args.mathdx, nvcc=args.nvcc,
                          sm=args.sm, cache=args.cache, timeout=args.timeout))


if __name__ == "__main__":
    main()
