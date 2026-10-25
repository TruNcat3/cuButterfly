#include "csrc/cpu/butterfly_cpu.h"
#include <cstring>
extern "C" void butterfly_execute(const void* weights,const void* input,void* output,int64_t rows,int64_t n,int64_t levels,int bits){
  auto options=torch::TensorOptions().dtype(bits==32?torch::kFloat32:torch::kFloat64);
  auto w=torch::from_blob(const_cast<void*>(weights),{1,1,levels,n/2,2,2},options);
  auto x=torch::from_blob(const_cast<void*>(input),{rows,1,n},options);
  auto y=butterfly_multiply_fw_cpu(w,x,true,n);
  std::memcpy(output,y.data_ptr(),rows*n*(bits/8));
}
