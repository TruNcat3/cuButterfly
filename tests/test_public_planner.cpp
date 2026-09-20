#include <cubutterfly/plan.hpp>
#include <iostream>
#include <cstdlib>
#include <stdexcept>

int main() try {
    cubutterfly::Context context;
    if(std::getenv("CUBUTTERFLY_TEST_FACTOR_STREAMED")) {
        cubutterfly::Transform transform;
        transform.extents={4096}; transform.batch=3;
        transform.storage=CUBUTTERFLY_DATA_COMPLEX_FP64;
        cubutterfly::PlanOptions options;
        options.mapping=R"({"backend":"factor-streamed","fft_core":"cufftdx-block","compute_unit":"auto","shared_layout":"writer-aligned","cross_twiddle":"recurrence","stage_partition":[12],"factor_partition":[6,6],"factor_ept":8,"factor_columns":4,"data_tiles_per_cta":5,"prefetch_depth":2})";
        cubutterfly::Plan plan(context,transform,options);
        options.mapping=plan.mapping_json();
        cubutterfly::Plan replay(context,transform,options);
        if(replay.mapping_json()!=options.mapping || plan.workspace_size()!=3*4096*16)
            throw std::runtime_error("factor public mapping or workspace lost physical factors");
        // Canonical records include the core in the physical group mappings.
        for(auto pos=options.mapping.find("cufftdx-block");pos!=std::string::npos;pos=options.mapping.find("cufftdx-block"))
            options.mapping.replace(pos,std::string("cufftdx-block").size(),"register-tile");
        cubutterfly::Plan native(context,transform,options);
        options.mapping=native.mapping_json();
        cubutterfly::Plan native_replay(context,transform,options);
        if(native_replay.mapping_json()!=options.mapping)
            throw std::runtime_error("native factor core lost during public replay");
    }
    for (int op=0; op<6; ++op) {
        cubutterfly::Transform transform;
        transform.op = static_cast<cubutterflyOperator_t>(op);
        transform.extents = {256}; transform.batch = 2;
        transform.normalize_inverse = false;
        if (op==CUBUTTERFLY_OPERATOR_NTT) transform.storage = CUBUTTERFLY_DATA_UINT64;
        else if (op==CUBUTTERFLY_OPERATOR_SUBSET_ZETA || op==CUBUTTERFLY_OPERATOR_SUPERSET_ZETA)
            transform.storage = CUBUTTERFLY_DATA_UINT32;
        else if (op!=CUBUTTERFLY_OPERATOR_FFT) transform.storage = CUBUTTERFLY_DATA_FP32;
        if (op==CUBUTTERFLY_OPERATOR_STRUCTURED_2X2) transform.stage_matrices = {{{1,0.25,-0.5,1}}};
        cubutterfly::Plan plan(context, transform);
        const auto mapping = plan.mapping_json();
        cubutterfly::PlanOptions replay_options;
        replay_options.mapping = mapping;
        cubutterfly::Plan replay(context, transform, replay_options);
        if (mapping!=replay.mapping_json()) throw std::runtime_error("public mapping replay changed the execution design");
        if (op==CUBUTTERFLY_OPERATOR_NTT) {
            cuntt::PlanConfig config;
            config.log_n=8; config.batch=2; config.auto_select=true;
            cuntt::Plan legacy(config);
            if (mapping!=cubutterfly::serialize_mapping(legacy.config()))
                throw std::runtime_error("C/C++ NTT planners disagree");
        } else {
            cuntt::ButterflyConfig config;
            config.op=op==0 ? cuntt::ButterflyOperator::Fft : op==2 ? cuntt::ButterflyOperator::Fwht :
                op==3 ? cuntt::ButterflyOperator::SubsetZeta : op==4 ? cuntt::ButterflyOperator::SupersetZeta :
                cuntt::ButterflyOperator::Structured2x2;
            config.log_n=8; config.batch=2; config.normalize_inverse=false; config.auto_select=true;
            config.placement=cuntt::ButterflyPlacement::OutOfPlace;
            if(op==3 || op==4) config.precision=cuntt::ButterflyPrecision::Uint32;
            if(op==5) config.stage_matrices={{1,0.25,-0.5,1}};
            cuntt::ButterflyPlan legacy(config);
            if(mapping!=cubutterfly::serialize_mapping(legacy.config()))
                throw std::runtime_error("C/C++ butterfly planners disagree");
        }
        std::cout << "operator " << op << ": public/C/C++ mapping and replay agree\n";
        transform.batch=7;
        cubutterfly::PlanOptions stream_options;
        stream_options.mapping=op==CUBUTTERFLY_OPERATOR_NTT ?
            R"({"schema_version":1,"kind":"ntt","backend":"shared-iterative","compute_unit":"radix2","stage_partition":[2,3,3],"stage_overlap":true,"batch_tile_count":2,"threads_per_block":128})" :
            R"({"schema_version":1,"kind":"butterfly","backend":"shared-iterative","fft_core":"scalar","compute_unit":"radix2","stage_partition":[2,3,3],"stage_overlap":true,"batch_tile_count":2,"tile_threads":128,"shared_layout":"writer-aligned"})";
        cubutterfly::Plan streamed(context,transform,stream_options);
        stream_options.mapping=streamed.mapping_json();
        cubutterfly::Plan stream_replay(context,transform,stream_options);
        if(stream_replay.mapping_json()!=streamed.mapping_json()) throw std::runtime_error("public multi-group stream mapping changed");
        const std::size_t width=op==CUBUTTERFLY_OPERATOR_FFT || op==CUBUTTERFLY_OPERATOR_NTT ? 8 : 4;
        if(streamed.workspace_size()!=2*2*2*256*width) throw std::runtime_error("public stream workspace lost per-edge slots");
        transform.batch=2;
        transform.extents={8,16};
        if(op==CUBUTTERFLY_OPERATOR_STRUCTURED_2X2) transform.stage_matrices={{{1,0.25,-0.5,1}},{{1,0.25,-0.5,1}}};
        cubutterfly::Plan matrix(context,transform);
        replay_options.mapping=matrix.mapping_json();
        cubutterfly::Plan matrix_replay(context,transform,replay_options);
        if(matrix.mapping_json()!=matrix_replay.mapping_json()) throw std::runtime_error("rank2 replay changed an axis mapping");
        if(op==CUBUTTERFLY_OPERATOR_FFT || op==CUBUTTERFLY_OPERATOR_NTT) {
            transform.extents={3,5}; transform.modulus=241;
            cubutterfly::Plan arbitrary(context,transform);
            replay_options.mapping=arbitrary.mapping_json();
            cubutterfly::Plan arbitrary_replay(context,transform,replay_options);
            if(arbitrary.mapping_json()!=arbitrary_replay.mapping_json()) throw std::runtime_error("Bluestein replay changed an internal mapping");
        }
    }
    return 0;
} catch(const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
