#pragma once
#include <cuntt/butterfly.hpp>
#include <nlohmann/json.hpp>
#include <cuda_runtime_api.h>
#include <dlfcn.h>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <memory>
#include <cmath>
#include <cstdlib>
namespace cuntt::detail {
using HadaJson=nlohmann::json;
inline std::string hada_hash(const std::string& path) {
    std::string quoted="'";
    for(char c:path) quoted += c=='\'' ? "'\\''" : std::string(1,c);
    quoted += "'";
    auto command="sha256sum -- "+quoted;
    FILE* f=popen(command.c_str(),"r");
    if(!f) throw std::invalid_argument("cannot hash HadaCore artifact");
    char result[66]={}; const auto n=fread(result,1,65,f); const int status=pclose(f);
    if(status || n<64) throw std::invalid_argument("cannot hash HadaCore artifact: "+path);
    return std::string(result,64);
}
inline HadaJson hada_read(const std::string& path) {
    std::ifstream f(path); if(!f) throw std::invalid_argument("missing HadaCore artifact: "+path);
    HadaJson j;f>>j;return j;
}
inline std::string hada_uuid(const cudaUUID_t& u) {
    std::ostringstream out;out<<"GPU-"<<std::hex<<std::setfill('0');
    for(int i=0;i<16;++i) {if(i==4||i==6||i==8||i==10)out<<'-';out<<std::setw(2)<<unsigned(static_cast<unsigned char>(u.bytes[i]));}
    return out.str();
}
inline bool hada_domain(const ButterflyConfig& c) {
    return c.allow_fp16_mma && c.op==ButterflyOperator::Fwht && c.precision==ButterflyPrecision::Fp16 &&
       c.accumulation==ButterflyAccumulation::Native && c.log_n>=1 && c.log_n<=15 && !c.inverse &&
       c.placement==ButterflyPlacement::OutOfPlace && c.element_stride==1 &&
       (!c.batch_stride || c.batch_stride==(std::size_t{1}<<c.log_n)) &&
       c.output_scale==1 && !c.final_storage_rounding;
}
inline HadaJson hada_profile(const ButterflyConfig& c) {
    if(!hada_domain(c))return {};
    const char* path=std::getenv("CUBUTTERFLY_HADACORE_PROFILE");if(!path||!*path)return {};
    auto j=hada_read(path);
    if(j.at("schema")!="cubutterfly-hadacore-profile-v1" ||
       j.at("arithmetic_contract")!="fp16-mma-orthonormal-final-scale")
        throw std::invalid_argument("invalid HadaCore profile schema/contract");
    int dev=0;cudaDeviceProp p{};
    if(cudaGetDevice(&dev)!=cudaSuccess||cudaGetDeviceProperties(&p,dev)!=cudaSuccess)
        throw std::invalid_argument("cannot identify HadaCore profile GPU");
    if(j.at("gpu_uuid")!=hada_uuid(p.uuid)||j.at("sm")!=p.major*10+p.minor)return {};
    if(hada_hash(j.at("library"))!=j.at("library_sha256"))
        throw std::invalid_argument("HadaCore library hash mismatch");
    j["profile_sha256"]=hada_hash(path);return j;
}
inline bool hada_evidence(const HadaJson& row) {
    if(!row.contains("evidence_path")||!row.contains("evidence_sha256"))return false;
    return hada_hash(row.at("evidence_path"))==row.at("evidence_sha256");
}
inline void hada_validate_evidence(const HadaJson& r,const HadaJson& j,const HadaJson& e) {
    const auto& m=e.at("mapping");
    bool good=e.at("status")=="validated-exact-upstream" && e.at("gpu_uuid")==j.at("gpu_uuid") &&
      e.at("library_sha256")==j.at("library_sha256") && e.at("contract")==j.at("arithmetic_contract") &&
      m.at("log_n")==r.at("log_n") && m.at("tile_count")==r.at("batch") &&
      m.at("chunks_per_warp")==r.at("chunks") && m.at("groups_per_block")==r.at("groups_per_block") &&
      m.at("arithmetic_contract")==j.at("arithmetic_contract") &&
      e.at("guards").at("before")==true && e.at("guards").at("after")==true;
    for(const char* k:{"exact_repeat","input_immutable","buffer_guards","graph_reproduced",
                      "upstream_graph_reproduced","exact_upstream_match","finite"}) good=good && e.at(k)==true;
    good=good && e.at("timing").at("median_ms")==r.at("median_ms") &&
      e.at("timing").at("protocol")==r.at("timing_protocol") && e.at("resources")==r.at("resources");
    if(!good)throw std::invalid_argument("HadaCore evidence does not match calibrated request/geometry/contract");
}
inline void hada_validate_native(const HadaJson& r,const HadaJson& j,const HadaJson& e) {
    bool good=e.at("status")=="measured-error-report" && e.at("gpu_uuid")==j.at("gpu_uuid") &&
      e.at("library_sha256")==r.at("library_sha256") && e.at("mapping")==r.at("mapping") &&
      e.at("arithmetic_policy")=="STRICT" && e.at("guards").at("before")==true && e.at("guards").at("after")==true;
    for(const char* k:{"exact_repeat","input_immutable","buffer_guards","graph_reproduced","finite"})good=good && e.at(k)==true;
    const auto& d=e.at("descriptor").at("semantics");
    good=good && d.at("operator")==2 && d.at("storage")==0 && d.at("compute")==0 &&
      d.at("shape")==HadaJson::array({uint64_t(1)<<r.at("log_n").get<unsigned>()}) &&
      d.at("batch")==r.at("batch") && d.at("inverse")==0 && d.at("placement")==0 &&
      d.value("arithmetic_policy",0)==0 && d.at("input_strides")==HadaJson::array({1}) &&
      d.at("output_strides")==HadaJson::array({1}) && d.at("input_batch_stride")==d.at("shape").at(0) &&
      d.at("output_batch_stride")==d.at("shape").at(0) && d.at("input_orders")==HadaJson::array({0}) &&
      d.at("output_orders")==HadaJson::array({0}) &&
      e.at("timing").at("median_ms")==r.at("median_ms") && e.at("timing").at("protocol")==r.at("timing_protocol");
    if(!good)throw std::invalid_argument("native calibration evidence request/mapping/protocol mismatch");
}
inline std::vector<ButterflyConfig> hada_candidates(const ButterflyConfig& base,const HadaJson& j) {
    std::vector<ButterflyConfig> out;if(j.is_null()||j.empty())return out;
    for(const auto& r:j.at("rows")) {
      if(r.at("log_n")!=base.log_n||r.at("batch")!=base.batch||r.at("status")!="validated-exact-upstream")continue;
      if(!hada_evidence(r))throw std::invalid_argument("HadaCore validation evidence hash mismatch");
      const double ms=r.at("median_ms");if(!(ms>0&&std::isfinite(ms)))continue;
      const auto evidence=hada_read(r.at("evidence_path"));
      hada_validate_evidence(r,j,evidence);
      auto c=base;c.auto_select=false;c.local_core="hadacore-local-tile";
      c.hadacore_chunks=r.at("chunks");c.hadacore_groups_per_block=r.at("groups_per_block");
      c.hadacore_library_sha256=j.at("library_sha256");c.hadacore_profile_sha256=j.at("profile_sha256");
      c.backend=ButterflyBackend::TemporalTile;c.compute_unit=ComputeUnit::Radix2;
      c.local_exchange=LocalExchange::SharedMemory;c.stage_partition={base.log_n};
      c.stage_overlap=false;c.tile_threads=r.at("resources").at("threads");
      out.push_back(std::move(c));
    }return out;
}
inline HadaJson hada_row(const ButterflyConfig& c,const HadaJson& j) {
 for(const auto& r:j.at("rows"))if(r.at("log_n")==c.log_n&&r.at("batch")==c.batch&&
 r.at("chunks")==c.hadacore_chunks&&r.at("groups_per_block")==c.hadacore_groups_per_block&&
 r.at("status")=="validated-exact-upstream")return r;
 throw std::invalid_argument("missing-executable: HadaCore calibrated geometry missing");
}
class HadaCoreProvider {
 void* library_=nullptr;
 using Execute=int(*)(void*,void*,uint64_t,int,int,uint32_t,float,cudaStream_t);
 Execute execute_=nullptr;
 public:
 struct Resources {int registers_per_thread,static_shared_bytes,local_bytes_per_thread,max_threads_per_block,active_blocks_per_sm,binary_version,ptx_version;};
 Resources resources{}; HadaJson row;
 explicit HadaCoreProvider(const ButterflyConfig& c) {
    auto j=hada_profile(c);if(j.empty()||c.hadacore_library_sha256!=j.at("library_sha256")||
      c.hadacore_profile_sha256!=j.at("profile_sha256"))throw std::invalid_argument("missing-executable: HadaCore profile identity mismatch");
    auto candidates=hada_candidates(c,j);bool found=false;
    for(const auto& v:candidates)if(v.hadacore_chunks==c.hadacore_chunks&&v.hadacore_groups_per_block==c.hadacore_groups_per_block)found=true;
    if(!found)throw std::invalid_argument("missing-executable: HadaCore geometry unvalidated");
    row=hada_row(c,j);
    library_=dlopen(j.at("library").get<std::string>().c_str(),RTLD_NOW|RTLD_LOCAL);
    if(!library_)throw std::invalid_argument("cannot load HadaCore provider");
    try {
      auto prepare=reinterpret_cast<int(*)(int,int)>(dlsym(library_,"cubutterfly_hadacore_tile_prepare"));
      auto query=reinterpret_cast<int(*)(int,int,Resources*)>(dlsym(library_,"cubutterfly_hadacore_tile_resources"));
      execute_=reinterpret_cast<Execute>(dlsym(library_,"cubutterfly_hadacore_tile_execute"));
      if(!prepare||!query||!execute_||prepare(c.log_n,c.hadacore_chunks)||query(c.log_n,c.hadacore_chunks,&resources))
        throw std::invalid_argument("HadaCore preparation/resource query failed");
      struct Info {int log_n,chunks_per_warp,warps_per_block,blocks_per_sm,threads,group_elements,shared_bytes,stages;};
      auto info_query=reinterpret_cast<int(*)(int,int,Info*)>(dlsym(library_,"cubutterfly_hadacore_tile_info"));
      Info info{};
      if(!info_query || info_query(c.log_n,c.hadacore_chunks,&info) || info.threads!=int(c.tile_threads) ||
         info.group_elements!=256*info.chunks_per_warp*info.warps_per_block ||
         info.shared_bytes!=row.at("resources").at("dynamic_shared_bytes") ||
         (c.batch*(uint64_t(1)<<c.log_n))%info.group_elements)
        throw std::invalid_argument("HadaCore compiled group geometry mismatch");
      const char* keys[]={"registers_per_thread","static_shared_bytes","local_bytes_per_thread",
        "max_threads_per_block","active_blocks_per_sm","binary_version","ptx_version"};
      const int actual[]={resources.registers_per_thread,resources.static_shared_bytes,resources.local_bytes_per_thread,
        resources.max_threads_per_block,resources.active_blocks_per_sm,resources.binary_version,resources.ptx_version};
      for(int i=0;i<7;++i)if(row.at("resources").at(keys[i])!=actual[i])
        throw std::invalid_argument("HadaCore compiler resources differ from calibration");
      if(c.tile_threads>unsigned(resources.max_threads_per_block)||resources.active_blocks_per_sm<=0)
        throw std::invalid_argument("HadaCore launch exceeds device resources");
    }catch(...){dlclose(library_);library_=nullptr;throw;}
 }
 ~HadaCoreProvider(){if(library_)dlclose(library_);}
 void execute(const ButterflyConfig& c,const void* a,void* b,cudaStream_t stream) const {
   if(execute_(const_cast<void*>(a),b,c.batch,c.log_n,c.hadacore_chunks,c.hadacore_groups_per_block,
               std::sqrt(float(std::size_t{1}<<c.log_n)),stream))throw std::runtime_error("HadaCore launch failed");
 }
};
}
