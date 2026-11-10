#include "walsh/walshwak.h"
#include "walsh/arithtransform.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>
#include <omp.h>

using Clock = std::chrono::steady_clock;

template<class T> void reference(std::vector<T>& a, size_t n, int kind) {
    for (size_t base=0; base<a.size(); base+=n)
        for (size_t bit=1; bit<n; bit*=2)
            for (size_t j=0; j<n; ++j) {
                if (j&bit) continue;
                const T u=a[base+j],v=a[base+j+bit];
                if (kind==0) { a[base+j]=u+v; a[base+j+bit]=u-v; }
                else if (kind==1) a[base+j+bit]=u+v;
                else a[base+j]=u+v;
            }
}

template<class T> void run(int logn, int batch, int threads, int kind) {
    const size_t n=size_t(1)<<logn, count=n*batch;
    std::mt19937_64 gen(20773043+logn+batch);
    std::vector<T> input(count), output(count);
    for (auto& x:input) {
        if (kind==0) x=T((int64_t)(gen()%2000001)-1000000)/T(1000000);
        else x=T(gen());
    }
    omp_set_dynamic(0); omp_set_num_threads(threads);
    auto call=[&]() {
        // FXT is in-place; the copy is the required out-of-place adapter.
        #pragma omp parallel for schedule(static)
        for (int b=0; b<batch; ++b) {
            T* dst=output.data()+b*n;
            std::copy_n(input.data()+b*n,n,dst);
            if (kind==0) walsh_wak(dst,logn);
            else if (kind==1) arith_transform_plus(dst,logn);
            else rev_arith_transform_plus(dst,logn);
        }
    };
    call();
    uint64_t mismatches=0;
    long double l2=0,ref2=0;
    std::vector<double> absolute,relative;
    if (kind==0) {
        std::vector<long double> ref(input.begin(),input.end());
        reference(ref,n,kind);
        for (size_t i=0;i<count;++i) {
            long double error=std::abs((long double)output[i]-ref[i]);
            absolute.push_back(double(error));
            relative.push_back(ref[i]==0 ? (error==0?0:INFINITY) : double(error/std::abs(ref[i])));
            l2+=error*error;ref2+=ref[i]*ref[i];
        }
    } else {
        auto ref=input;reference(ref,n,kind);
        for(size_t i=0;i<count;++i) mismatches+=ref[i]!=output[i];
        // Independent direct set enumeration confirms natural subset/superset order.
        std::vector<uint32_t> small(256);
        for(size_t j=0;j<256;++j)small[j]=uint32_t(gen());
        auto got=small;
        if(kind==1)arith_transform_plus(got.data(),8);else rev_arith_transform_plus(got.data(),8);
        for(size_t mask=0;mask<256;++mask) {
            uint32_t total=0;
            for(size_t j=0;j<256;++j)
                if(kind==1 ? (j&mask)==j : (j&mask)==mask)total+=small[j];
            mismatches+=total!=got[mask];
        }
        if(mismatches)throw std::runtime_error("integer reference mismatch");
    }
    for(int i=0;i<3;++i)call();
    std::vector<double> samples;
    for(int s=0;s<7;++s) {
        auto start=Clock::now();for(int r=0;r<5;++r)call();
        samples.push_back(std::chrono::duration<double,std::micro>(Clock::now()-start).count()/5);
    }
    auto sorted=samples;std::sort(sorted.begin(),sorted.end());
    const double time=sorted[3];
    std::cout<<"{\"operator\":\""<<(kind==0?"fwht":kind==1?"subset-zeta":"superset-zeta")
        <<"\",\"precision\":\""<<(kind==0?"fp64":"uint32")<<"\",\"N\":"<<n
        <<",\"B\":"<<batch<<",\"threads\":"<<threads<<",\"latency_us\":"<<time
        <<",\"transforms_per_s\":"<<batch*1e6/time<<",\"per_transform_us\":"<<time/batch
        <<",\"mismatches\":"<<mismatches<<",\"checked_elements\":"<<count<<",\"samples_us\":[";
    for(size_t i=0;i<samples.size();++i)std::cout<<(i?",":"")<<samples[i];
    std::cout<<"]";
    if(kind==0) {
        std::sort(absolute.begin(),absolute.end());std::sort(relative.begin(),relative.end());
        std::cout<<",\"relative_l2\":"<<double(std::sqrt(l2/ref2))<<",\"absolute_quantiles\":[";
        for(double q:{.5,.9,.99,1.})std::cout<<(q==.5?"":",")<<absolute[size_t(q*(count-1))];
        std::cout<<"],\"relative_quantiles\":[";
        for(double q:{.5,.9,.99,1.})std::cout<<(q==.5?"":",")<<relative[size_t(q*(count-1))];
        std::cout<<"]";
    }
    std::cout<<"}\n";
}

int main() {
    std::cout<<std::setprecision(17);
    for(int kind:{0,1,2})for(int logn:{12,16,20})for(int batch:{1,16,4})for(int threads:{1,4}) {
        if (logn==20 && batch!=4) continue;
        if (logn!=20 && batch==4) continue;
        if(kind==0)run<double>(logn,batch,threads,kind);
        else run<uint32_t>(logn,batch,threads,kind);
    }
}
