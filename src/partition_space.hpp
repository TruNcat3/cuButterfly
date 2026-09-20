#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <set>
#include <vector>

namespace cuntt::detail {
// Round-robin all group counts. A budget limits enumeration effort, never the
// legal number of groups. Budget zero enumerates the entire bounded composition.
inline std::vector<std::vector<std::uint32_t>> shared_partitions(unsigned stages,unsigned max_local,std::size_t budget,
                                                              unsigned min_local=1,bool balanced_first=false) {
    using Partition=std::vector<std::uint32_t>;
    if(!stages || !min_local || max_local<min_local) return {};
    auto fill=[&](Partition& p,std::size_t begin,unsigned remaining) {
        for(auto i=begin;i<p.size();++i) {
            p[i]=std::max(min_local,remaining>max_local*(p.size()-i-1) ? unsigned(remaining-max_local*(p.size()-i-1)) : min_local);
            remaining-=p[i];
        }
    };
    std::vector<Partition> cursors;
    for(unsigned groups=(stages+max_local-1)/max_local;groups<=stages/min_local;++groups) {
        Partition p(groups); fill(p,0,stages); cursors.push_back(std::move(p));
    }
    std::vector<Partition> out;
    std::set<Partition> seen;
    // One balanced representative of every feasible group count before
    // lexicographic exploration. This is a budget ordering, not a shape limit.
    if(balanced_first) for(const auto& cursor:cursors) {
        Partition p(cursor.size(),stages/cursor.size());
        for(unsigned i=0;i<stages%cursor.size();++i) ++p[i];
        seen.insert(p); out.push_back(std::move(p));
        if(budget && out.size()==budget) return out;
    }
    bool active=true;
    while(active && (!budget || out.size()<budget)) {
        active=false;
        for(auto& p:cursors) {
            if(p.empty()) continue;
            if(seen.insert(p).second) out.push_back(p);
            active=true;
            if(budget && out.size()==budget) break;
            bool next=false;
            for(std::size_t i=p.size()-1;i>0;) {
                --i;
                unsigned prefix=0;
                for(std::size_t j=0;j<i;++j) prefix+=p[j];
                const auto value=p[i]+1;
                const auto tail=p.size()-i-1;
                if(value<=max_local && prefix+value+min_local*tail<=stages && stages-prefix-value<=max_local*tail) {
                    p[i]=value; fill(p,i+1,stages-prefix-value); next=true; break;
                }
            }
            if(!next) p.clear();
        }
    }
    return out;
}
} // namespace cuntt::detail
