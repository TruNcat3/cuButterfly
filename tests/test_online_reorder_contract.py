"""CPU checks for generic online-reorder grouping capacity and candidate axes."""
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def test_grouped_online_reorder_contract(tmpdir):
    tmp_path = Path(str(tmpdir))
    source = tmp_path / "contract.cpp"
    source.write_text(r'''
#include "online_reorder_contract.hpp"
#include <cassert>
enum class Op { Fft, Fwht, SubsetZeta, SupersetZeta, XorZeta };
enum class Precision { Fp16, Bf16, Fp32, Fp64, Uint32, Uint64 };
struct Config { Op op; Precision precision; };
int main() {
    using namespace cuntt::detail;
    for (auto op : {Op::SubsetZeta, Op::SupersetZeta, Op::XorZeta}) {
        for (auto precision : {Precision::Fp16, Precision::Bf16, Precision::Fp32, Precision::Fp64, Precision::Uint32}) {
            Config config{op, precision};
            assert(online_reorder_value_bytes(config)==4);
            assert(!online_reorder_shape(16,8,64,1,online_reorder_value_bytes(config)).legal);
        }
        assert(online_reorder_value_bytes(Config{op,Precision::Uint64})==8);
    }
    assert(online_reorder_value_bytes(Config{Op::Fft,Precision::Bf16})==4);
    assert(online_reorder_value_bytes(Config{Op::Fwht,Precision::Fp16})==2);
    for (unsigned bytes : {2U,4U,8U,16U}) {
        auto shape=online_reorder_shape(16,8,4,4,bytes);
        assert(shape.legal && shape.prefix_values==1024 && shape.suffix_values==1024);
        assert(shape.max_shared_bytes==1024*bytes);
        assert(online_reorder_shape(16,8,8,1,bytes).legal);
        assert(online_reorder_shape(16,8,1,8,bytes).legal);
        auto groups=online_reorder_group_candidates(16,8,bytes);
        assert(groups.front()==std::make_pair(1U,1U));
        bool prefix=false,suffix=false,coupled=false;
        for (auto [g,c] : groups) {
            assert(online_reorder_shape(16,8,g,c,bytes).legal);
            prefix |= g>1 && c==1; suffix |= g==1 && c>1; coupled |= g>1 && c>1;
        }
        assert(prefix && suffix && coupled && groups.size()<=31);
    }
    assert(online_reorder_shape(16,8,16,16,4).legal); // 4096 values: 16 KiB
    assert(!online_reorder_shape(16,8,64,64,4).legal); // 16384 values: 64 KiB
    assert(!online_reorder_shape(16,8,512,1,4).legal); // exceeds prefix tile axis
    assert(!online_reorder_shape(16,8,3,1,4).legal);
    assert(!online_reorder_shape(16,8,1,0,4).legal);
    assert(!online_reorder_shape(6,5,4,1,4).legal);
    assert(!online_reorder_shape(6,5,1,64,4).legal);
    assert(!online_reorder_shape(21,10,1,1,4).legal);
    assert(!online_reorder_shape(10,4,1,1,4).legal);
    assert(!online_reorder_shape(16,8,4,4,16,8192).legal);
    assert(online_reorder_shape(16,8,2,2,16,8192).legal);
    assert(!online_reorder_shape(16,8,1,1,256,1024*1024).legal);
}
''')
    binary = tmp_path / "contract"
    subprocess.run(["g++", "-std=c++17", "-Wall", "-Wextra", "-I"+str(ROOT/"src"),
                    str(source), "-o", str(binary)], check=True, capture_output=True, text=True)
    subprocess.run([str(binary)], check=True)
