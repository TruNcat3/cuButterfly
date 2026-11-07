from pathlib import Path
import subprocess


def test_shared_columns_match_capacity_and_independent_work(tmp_path):
    root = Path(__file__).resolve().parents[1]
    source = tmp_path / "check.cpp"
    source.write_text(r'''
#include "shared_tile_contract.hpp"
#include <cassert>
int main() {
    using namespace cuntt::detail;
    assert((shared_tile_candidates(16,{8,8},4,16384)==std::vector<unsigned>{1,2,4,8}));
    assert((shared_tile_candidates(16,{8,8},16,16384)==std::vector<unsigned>{1,2}));
    assert((shared_tile_candidates(12,{12},4,65536)==std::vector<unsigned>{1}));
    assert(shared_tile_candidates(16,{8,8},4,1024).empty());
    assert(shared_tile_bytes(8,8,4)==16384);
    validate_shared_tiles(16,{8,8},8);
    for (auto tiles:{0U,3U,512U}) {
        bool rejected=false;
        try { validate_shared_tiles(16,{8,8},tiles); }
        catch (const std::invalid_argument&) { rejected=true; }
        assert(rejected);
    }
}
''')
    binary = tmp_path / "check"
    subprocess.run(["c++", "-std=c++17", "-I" + str(root / "src"),
                    str(source), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True)
