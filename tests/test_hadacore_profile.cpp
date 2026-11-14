#include "hadacore_provider.hpp"
#include <iostream>
#include <cassert>
using namespace cuntt::detail;
int main() {
 HadaJson r={{"log_n",8},{"batch",1},{"chunks",1},{"groups_per_block",1},
             {"median_ms",0.01},{"timing_protocol","test-protocol"},{"resources",{{"threads",32}}}};
 HadaJson j={{"gpu_uuid","GPU-test"},{"library_sha256","binary"},
             {"arithmetic_contract","fp16-mma-orthonormal-final-scale"}};
 HadaJson e={{"status","validated-exact-upstream"},{"gpu_uuid","GPU-test"},
   {"library_sha256","binary"},{"contract",j.at("arithmetic_contract")},
   {"mapping",{{"log_n",8},{"tile_count",1},{"chunks_per_warp",1},{"groups_per_block",1},
               {"arithmetic_contract",j.at("arithmetic_contract")}}},
   {"guards",{{"before",true},{"after",true}}},
   {"timing",{{"median_ms",0.01},{"protocol","test-protocol"}}},{"resources",r.at("resources")}};
 for(const char* k:{"exact_repeat","input_immutable","buffer_guards","graph_reproduced",
                   "upstream_graph_reproduced","exact_upstream_match","finite"})e[k]=true;
 hada_validate_evidence(r,j,e);
 auto reject=[&](const HadaJson& changed){bool threw=false;try{hada_validate_evidence(r,j,changed);}catch(const std::exception&){threw=true;}assert(threw);};
 for(const char* k:{"exact_repeat","input_immutable","buffer_guards","graph_reproduced",
                   "upstream_graph_reproduced","exact_upstream_match","finite"}){auto c=e;c[k]=false;reject(c);}
 for(const char* k:{"gpu_uuid","library_sha256","contract","status"}){auto c=e;c[k]="wrong";reject(c);}
 for(const char* k:{"log_n","tile_count","chunks_per_warp","groups_per_block"}){auto c=e;c["mapping"][k]=99;reject(c);}
 {auto c=e;c["guards"]["after"]=false;reject(c);}
 {auto c=e;c["timing"]["median_ms"]=0.02;reject(c);}
 {auto c=e;c["timing"]["protocol"]="other";reject(c);}
 {auto c=e;c["resources"]["threads"]=64;reject(c);}
 std::cout<<"HadaCore evidence binding CPU tests passed\n";
}
