#include "fft_factor_plan.hpp"
#include <stdexcept>
#include <iostream>
using namespace cuntt;
using namespace cuntt::detail;
void require(bool b) { if (!b) throw std::runtime_error("FP16 factor contract mismatch"); }
template<class F> void rejects(F f) { try { f(); } catch(const std::invalid_argument&) { return; } throw std::runtime_error("invalid factor accepted"); }
ButterflyConfig point() {
 ButterflyConfig c; c.op=ButterflyOperator::Fft; c.precision=ButterflyPrecision::Fp16;
 c.backend=ButterflyBackend::FactorStreamed; c.fft_core=FftCore::CufftDxBlock;
 c.shared_layout=SharedLayout::WriterAligned; c.cross_twiddle=CrossTwiddleMode::Recurrence;
 c.log_n=12; c.batch=5; c.stage_partition={12}; c.factor_partition={6,6};
 c.factor_columns=8; c.factor_ept=8; c.data_tiles_per_cta=4; c.prefetch_depth=2;
 return c;
}
int main() try {
 auto c=point(); normalize_factor_mapping(c);
 require(c.execution_group_mappings.size()==2 && c.tile_threads==32);
 auto p=make_factor_dataflow_plan(c); require(p.execution_groups.size()==2);
 for(const auto& g:p.execution_groups) {
  require(g.threads==32 && g.units_per_cta==8 && g.grid_ctas==10);
  require(g.data_space*g.data_time==64*8/2*4);
  require(g.live_shared_bytes==64*8*4*4);
 }
 require(p.inter_stage_global_bytes==5*4096*4);
 c=point(); c.inverse=true; c.normalize_inverse=false; normalize_factor_mapping(c);
 c=point(); c.inverse=true; c.normalize_inverse=true; rejects([&]{ normalize_factor_mapping(c); });
 c=point(); c.accumulation=ButterflyAccumulation::Fp32; rejects([&]{ normalize_factor_mapping(c); });
 c=point(); c.precision=ButterflyPrecision::Bf16; rejects([&]{ normalize_factor_mapping(c); });
 c=point(); c.fft_core=FftCore::RegisterTile; rejects([&]{ normalize_factor_mapping(c); });
 for(auto cols:{1U,3U,64U}) { c=point(); c.factor_columns=cols; rejects([&]{ normalize_factor_mapping(c); }); }
 c=point(); c.log_n=18; c.stage_partition={18}; c.factor_partition={6,6,6};
 c.factor_slices=2; c.factor_overlap=true; c.factor_io_policies={"dynamic","static-unrolled","dynamic"};
 normalize_factor_mapping(c); p=make_factor_dataflow_plan(c);
 require(p.execution_groups[0].launch_count==2 && p.execution_groups[1].partial_dependency_ready);
 const auto r=factor_tile_range(c,0,1); require(r.first==4 && r.period==8 && r.span==4);
 c=point(); c.factor_columns=32; c.factor_ept=8; normalize_factor_mapping(c); require(c.tile_threads==128);
 c=point(); c.precision=ButterflyPrecision::Fp32; normalize_factor_mapping(c); require(c.tile_threads==64);
 std::cout<<"FP16 factor contract passed\n"; return 0;
} catch(const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
