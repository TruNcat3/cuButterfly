#include "platform_tuning.hpp"
#include "platforms/legacy_identity.hpp"
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <iostream>
#include <unistd.h>

int main() try {
    using Json=nlohmann::json;
    using cuntt::detail::match_platform_rules;
    using cuntt::detail::platform_rule_candidates;
    const Json device={{"device","test GPU"},{"compute_capability","8.0"},{"global_memory_bytes",80ULL<<30}};
    const Json semantic={{"operator","ntt"},{"precision","word64"},{"direction","forward"},
        {"logN",20},{"batch",4},{"output_order","bit-reversed"}};
    const Json rule={{"match",{{"operator","ntt"},{"precision","word64"},{"direction","forward"},
        {"logN",20},{"batch",{{"min",1},{"max",16}}},{"output_order","bit-reversed"}}},
        {"mapping",{{"backend","compact-stage"}}}};
    Json doc={{"schema","cubutterfly-platform-tuning-v1"},
        {"targets",Json::array({{{"hardware",device},{"rules",Json::array({rule})}}})}};
    auto require=[](bool value) {if(!value) throw std::runtime_error("platform matching contract failed");};
    require(cuntt::detail::has_legacy_v100_prior("Tesla V100-SXM2-16GB", 7, 0));
    require(!cuntt::detail::has_legacy_v100_prior("TITAN V", 7, 0));
    require(!cuntt::detail::has_legacy_v100_prior("Quadro GV100", 7, 0));
    require(!cuntt::detail::has_legacy_v100_prior("Tesla V100-SXM2-32GB", 7, 0));
    require(!cuntt::detail::has_legacy_v100_prior("Tesla V100-SXM2-16GB", 8, 0));
    require(match_platform_rules(doc,device,semantic).size()==1);
    for(const auto* key: {"device","compute_capability","global_memory_bytes"}) {
        auto other=device;
        other[key]=std::string(key)=="global_memory_bytes" ? Json(40ULL<<30) : Json("different");
        require(match_platform_rules(doc,other,semantic).empty());
    }
    for (const auto* key:{"direction","precision","output_order"}) {
        auto other=semantic; other[key]="different";
        require(match_platform_rules(doc,device,other).empty());
    }
    auto large=semantic; large["batch"]=32;
    require(match_platform_rules(doc,device,large).empty());
    auto bad=doc; bad["targets"][0]["rules"][0]["match"]["unknown_preference"]=1;
    bool rejected=false;
    try {match_platform_rules(bad,device,semantic);} catch(const std::invalid_argument&) {rejected=true;}
    require(rejected);

    cuntt::ButterflyConfig inverse_fft;
    inverse_fft.op=cuntt::ButterflyOperator::Fft;
    inverse_fft.precision=cuntt::ButterflyPrecision::Fp32;
    inverse_fft.placement=cuntt::ButterflyPlacement::InPlace;
    inverse_fft.log_n=8;
    inverse_fft.batch=4;
    inverse_fft.inverse=true;
    inverse_fft.normalize_inverse=true;
    inverse_fft.element_stride=2;
    const Json inverse_fft_rule={{"match",{{"operator","fft"},{"precision","fp32"},{"direction","inverse"},
        {"normalization","inverse"},{"placement","in-place"},{"element_stride",2},{"batch_stride",511}}},
        {"mapping",{{"backend","temporal-tile"}}}};
    const Json inverse_fft_doc={{"schema","cubutterfly-platform-tuning-v1"},
        {"targets",Json::array({{{"hardware",device},{"rules",Json::array({inverse_fft_rule})}}})}};
    const auto inverse_candidates=platform_rule_candidates(inverse_fft,inverse_fft_doc,device);
    require(inverse_candidates.size()==1);
    auto forward_fft=inverse_fft;
    forward_fft.inverse=false;
    require(platform_rule_candidates(forward_fft,inverse_fft_doc,device).empty());

    cuntt::ButterflyConfig final_fwht;
    final_fwht.op=cuntt::ButterflyOperator::Fwht;
    final_fwht.precision=cuntt::ButterflyPrecision::Fp16;
    final_fwht.accumulation=cuntt::ButterflyAccumulation::Fp32;
    final_fwht.log_n=5;
    final_fwht.output_scale=0.125;
    final_fwht.final_storage_rounding=true;
    const Json final_fwht_rule={{"match",{{"operator","fwht"},{"precision","fp16"},{"direction","forward"},
        {"output_scale",0.125},{"output_rounding",1}}},
        {"mapping",{{"backend","temporal-tile"}}}};
    const Json final_fwht_doc={{"schema","cubutterfly-platform-tuning-v1"},
        {"targets",Json::array({{{"hardware",device},{"rules",Json::array({final_fwht_rule})}}})}};
    const auto final_candidates=platform_rule_candidates(final_fwht,final_fwht_doc,device);
    require(final_candidates.size()==1);
    require(final_candidates.front().output_scale==0.125 && final_candidates.front().final_storage_rounding);
    auto unscaled_fwht=final_fwht;
    unscaled_fwht.output_scale=1.0;
    require(platform_rule_candidates(unscaled_fwht,final_fwht_doc,device).empty());
    auto stage_rounded_fwht=final_fwht;
    stage_rounded_fwht.final_storage_rounding=false;
    require(platform_rule_candidates(stage_rounded_fwht,final_fwht_doc,device).empty());

    cuntt::ButterflyConfig structured;
    structured.op=cuntt::ButterflyOperator::Structured2x2;
    structured.precision=cuntt::ButterflyPrecision::Fp32;
    structured.log_n=8;
    structured.batch=2;
    structured.stage_matrices={{1.0,0.25,-0.5,1.0}};
    const Json structured_rule={{"match",{{"operator","structured-2x2"},{"precision","fp32"},{"direction","forward"},
        {"batch_stride",256},{"stage_matrices","1:0.25:-0.5:1"}}},
        {"mapping",{{"backend","temporal-tile"}}}};
    const Json structured_doc={{"schema","cubutterfly-platform-tuning-v1"},
        {"targets",Json::array({{{"hardware",device},{"rules",Json::array({structured_rule})}}})}};
    require(platform_rule_candidates(structured,structured_doc,device).size()==1);
    structured.stage_matrices[0].m00=2.0;
    require(platform_rule_candidates(structured,structured_doc,device).empty());

    cuntt::PlanConfig ntt;
    ntt.log_n=20;
    ntt.batch=4;
    ntt.word_bits=64;
    ntt.output_order=cuntt::OutputOrder::BitReversed;
    const Json ntt_match={{"operator","ntt"},{"precision","word64"},{"direction","forward"},
        {"logN",20},{"batch",4},{"input_order","natural"},{"output_order","bit-reversed"}};
    const Json valid_ntt_mapping={{"match",ntt_match},
        {"mapping",{{"backend","compact-stage"},{"compute_unit","radix2"}}}};
    const Json invalid_ntt_mapping={{"match",ntt_match},
        {"mapping",{{"backend","invalid-backend"}}}};
    const Json nonobject_ntt_mapping={{"match",ntt_match},{"mapping",Json::array({"not-a-mapping"})}};
    const Json second_ntt_mapping={{"match",ntt_match},
        {"mapping",{{"backend","shared-iterative"},{"compute_unit","radix2"}}}};
    const Json ntt_doc={{"schema","cubutterfly-platform-tuning-v1"},
        {"targets",Json::array({{{"hardware",device},{"rules",Json::array({valid_ntt_mapping,
            invalid_ntt_mapping,nonobject_ntt_mapping,second_ntt_mapping})}}})}};
    const auto ntt_candidates=platform_rule_candidates(ntt,ntt_doc,device);
    require(ntt_candidates.size()==2);
    require(ntt_candidates[0].backend==cuntt::Backend::CompactStage &&
            ntt_candidates[1].backend==cuntt::Backend::SharedIterative);

    const auto temporary=std::filesystem::temp_directory_path();
    const auto suffix=std::to_string(static_cast<unsigned long long>(::getpid()));
    const auto path_a=temporary/("cubutterfly_platform_tuning_a_"+suffix+".json");
    const auto path_b=temporary/("cubutterfly_platform_tuning_b_"+suffix+".json");
    const std::string multiline="{\n\t\"schema\":\"cubutterfly-platform-tuning-v1\"\n}";
    std::ofstream(path_a) << multiline;
    std::ofstream(path_b) << multiline;
    const char* previous=std::getenv("CUBUTTERFLY_PLATFORM_TUNING");
    const std::string previous_value=previous ? previous : "";
    const bool had_previous=previous != nullptr;
    auto restore_environment=[&]() {
        if (had_previous) setenv("CUBUTTERFLY_PLATFORM_TUNING",previous_value.c_str(),1);
        else unsetenv("CUBUTTERFLY_PLATFORM_TUNING");
    };
    require(setenv("CUBUTTERFLY_PLATFORM_TUNING",path_a.c_str(),1)==0);
    const auto identity_a=cuntt::detail::platform_tuning_identity();
    require(identity_a.find('\n')==std::string::npos && identity_a.find('\t')==std::string::npos);
    require(identity_a.find("7b0a0922")!=std::string::npos);
    require(setenv("CUBUTTERFLY_PLATFORM_TUNING",path_b.c_str(),1)==0);
    require(cuntt::detail::platform_tuning_identity()==identity_a);
    std::ofstream(path_a) << "{\"schema\":\"cubutterfly-platform-tuning-v1\"}";
    require(setenv("CUBUTTERFLY_PLATFORM_TUNING",path_a.c_str(),1)==0);
    require(cuntt::detail::platform_tuning_identity()!=identity_a);
    restore_environment();
    std::error_code cleanup_error;
    std::filesystem::remove(path_a,cleanup_error);
    std::filesystem::remove(path_b,cleanup_error);

    std::cout << "Platform rules preserve hardware identity, direction, precision, order and ranges\n";
    return 0;
} catch(const std::exception& error) { std::cerr<<error.what()<<'\n'; return 1; }
