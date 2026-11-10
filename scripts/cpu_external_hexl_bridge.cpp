#include <hexl/ntt/ntt.hpp>
#include <cstdint>
#include <vector>
struct Plan {
  intel::hexl::NTT ntt;
  uint64_t n, q;
  std::vector<uint64_t> twist, untwist, scratch, rev;
  Plan(uint64_t n_, uint64_t q_, uint64_t psi):ntt(n_,q_,psi),n(n_),q(q_),twist(n),untwist(n),scratch(n),rev(n) {
    auto mul=[&](uint64_t a,uint64_t b){return uint64_t((__uint128_t(a)*b)%q);};
    auto power=[&](uint64_t a,uint64_t e){uint64_t r=1;while(e){if(e&1)r=mul(r,a);a=mul(a,a);e>>=1;}return r;};
    uint64_t inv=power(psi,q-2); twist[0]=untwist[0]=1;
    unsigned bits=0;while((1ULL<<bits)<n)++bits;
    for(uint64_t i=0;i<n;++i){uint64_t x=i,r=0;for(unsigned b=0;b<bits;++b){r=(r<<1)|(x&1);x>>=1;}rev[i]=r;if(i){twist[i]=mul(twist[i-1],inv);untwist[i]=mul(untwist[i-1],psi);}}
  }
};
extern "C" void* hexl_library_address(){return reinterpret_cast<void*>(&intel::hexl::NTT::CheckArguments);}
extern "C" void* hexl_create(uint64_t n,uint64_t q,uint64_t psi){try{return new Plan(n,q,psi);}catch(...){return nullptr;}}
extern "C" void hexl_close(void* p){delete static_cast<Plan*>(p);}
extern "C" void hexl_execute(void* p,const uint64_t* input,uint64_t* output,uint64_t rows,int inverse,int normalized){
  auto& a=*static_cast<Plan*>(p);
  auto mul=[&](uint64_t x,uint64_t y){return uint64_t((__uint128_t(x)*y)%a.q);};
  for(uint64_t row=0;row<rows;++row){const auto* src=input+row*a.n;auto* dst=output+row*a.n;
    if(!inverse){for(uint64_t j=0;j<a.n;++j)a.scratch[j]=mul(src[j],a.twist[j]);a.ntt.ComputeForward(dst,a.scratch.data(),1,1);for(uint64_t j=0;j<a.n;++j)a.scratch[j]=dst[a.rev[j]];for(uint64_t j=0;j<a.n;++j)dst[j]=a.scratch[j];}
    else{for(uint64_t j=0;j<a.n;++j)a.scratch[a.rev[j]]=src[j];a.ntt.ComputeInverse(dst,a.scratch.data(),1,1);for(uint64_t j=0;j<a.n;++j){dst[j]=mul(dst[j],a.untwist[j]);if(!normalized)dst[j]=mul(dst[j],a.n);}}
  }
}
