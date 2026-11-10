// Standalone module validation. Compile ahead of time; run only on an idle GPU.
// Random errors are reported against CPU FP64; no acceptance threshold is invented.
#include <cubutterfly/module.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <dlfcn.h>
#include <complex>
#include <vector>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <stdexcept>
struct C {__half r,i;};
void check(cudaError_t x){if(x!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(x));}
void fft(std::vector<std::complex<double>>& x,bool inverse,bool norm){
 size_t n=x.size();
 for(size_t i=1,j=0;i<n;++i){size_t k=n>>1;for(;j&k;k>>=1)j^=k;j^=k;if(i<j)std::swap(x[i],x[j]);}
 for(size_t m=2;m<=n;m*=2){auto w=std::polar(1.0,(inverse?2:-2)*acos(-1.0)/m);for(size_t a=0;a<n;a+=m){std::complex<double> z=1;for(size_t k=0;k<m/2;++k){auto u=x[a+k],v=x[a+k+m/2]*z;x[a+k]=u+v;x[a+k+m/2]=u-v;z*=w;}}}
 if(inverse&&norm)for(auto& z:x)z/=n;
}
int main(int argc,char** argv){try{
 if(argc!=3)throw std::runtime_error("usage: test module.so N");
 auto h=dlopen(argv[1],RTLD_NOW|RTLD_LOCAL);if(!h)throw std::runtime_error(dlerror());
 auto entry=reinterpret_cast<cubutterflyModuleEntryV1>(dlsym(h,"cubutterfly_module_v1"));
 if(!entry)throw std::runtime_error("missing entry");auto api=entry();
 if(api->group_count!=1)throw std::runtime_error("group count");
 cubutterflyModuleResourcesV1 resource{};
 check(static_cast<cudaError_t>(api->resources(0,0,&resource)));
 check(static_cast<cudaError_t>(api->resources(1,0,&resource)));
 const size_t n=std::stoul(argv[2]);unsigned cases=0;
 for(size_t b:{1U,2U,3U,5U})for(size_t stride:{1U,2U})for(bool inplace:{false,true})
 for(bool inv:{false,true})for(bool norm:{false,true})for(bool impulse:{false,true}){
  const size_t dist=n*stride+7,total=b*dist;std::vector<C> input(total),output(total);
  std::vector<std::vector<std::complex<double>>> refs(b,std::vector<std::complex<double>>(n));
  for(size_t t=0;t<b;++t)for(size_t j=0;j<n;++j){double re=impulse?(j==0?1:0):std::sin(j*.177+t)*.03125;
   double im=impulse?0:std::cos(j*.137+t)*.03125;C c{__float2half(re),__float2half(im)};
   input[t*dist+j*stride]=c;refs[t][j]={__half2float(c.r),__half2float(c.i)};}
  for(auto& x:refs)fft(x,inv,norm);
  C *din=nullptr,*dout=nullptr;check(cudaMalloc(&din,total*sizeof(C)));if(inplace)dout=din;else check(cudaMalloc(&dout,total*sizeof(C)));
  check(cudaMemcpy(din,input.data(),total*sizeof(C),cudaMemcpyHostToDevice));
  cubutterflyModuleInvocationV1 io{din,dout,nullptr,b,dist,stride,inv,norm,nullptr};
  check(static_cast<cudaError_t>(api->launch(&io)));check(cudaDeviceSynchronize());
  check(cudaMemcpy(output.data(),dout,total*sizeof(C),cudaMemcpyDeviceToHost));
  double maxerr=0,err2=0,ref2=0;
  for(size_t t=0;t<b;++t)for(size_t j=0;j<n;++j){auto c=output[t*dist+j*stride];std::complex<double> z{__half2float(c.r),__half2float(c.i)};
   if(!std::isfinite(z.real())||!std::isfinite(z.imag()))throw std::runtime_error("nonfinite output");
   double e=std::abs(z-refs[t][j]);maxerr=std::max(maxerr,e);err2+=e*e;ref2+=std::norm(refs[t][j]);
   if(impulse&&e!=0)throw std::runtime_error("impulse pair/stride/direction mismatch");}
  std::cout<<"B="<<b<<" stride="<<stride<<" inplace="<<inplace<<" inverse="<<inv<<" norm="<<norm<<" impulse="<<impulse<<" max_abs="<<maxerr<<" rel_l2="<<sqrt(err2/std::max(ref2,1e-300))<<'\n';
  if(!inplace)check(cudaFree(dout));check(cudaFree(din));++cases;
 }
 dlclose(h);std::cout<<"completed="<<cases<<'\n';return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
