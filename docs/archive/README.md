# Historical reports and document inventory

Return to the [task-based documentation entry](../README.md) for library use,
research methods, engineering and current evidence. Files keep their existing
paths so citations and reproducibility commands remain usable.

Dated records describe their original build, hardware, scope and pending work.
A pending item in an older campaign is not the current project status. For the
A100 closeout and November 9/8 follow-ups, use the
[current paper-evidence route](../README.md#read-the-a100-paper-evidence).

The [experimental results guide](../benchmarks/results.md) organizes the accepted
measurements by research question.

## v0.8 and V100 evidence

[Release v0.8](../release_v08.md) defines the frozen release boundary.
[V100 three-way comparison](../v100_three_way_comparison.md),
[external baselines](../v100_external_baselines.md),
[scaling](../v100_scaling_results.md) and
[v0.8/v0.6 reconciliation](../v08_v06_cufft_reconciliation.md)
retain their own populations and protocols. The
[V100 visual summary](../../figures/v100_library_comparison.svg) belongs to that
historical scope. [Release v0.9](../release_v09.md) records the later framework milestone.

Earlier RNS artifact reports linked from the former documentation homepage are
local artifacts absent from this checkout. Their original paths are retained for
recovery: `results/rns_exchange_20261021/README.md`,
`results/rns_single_warp_20261021/README.md`,
`results/rns_stage_role_20261021/README.md`, and
`results/rns_tile_io_20261021/README.md`.
The [artifact index](../../results/README.md) is the entry to available raw records.

## Dated reports

This inventory covers Markdown documents under `docs/`; it does not enumerate
the raw `results/` tree or replace the experiment index.
Dates below come from filenames and are not claims of current validation.

### 2026-11

- [A100 论文补证收尾（2026-11-07）](../a100_evidence_followup_20261107.md) — `a100_evidence_followup_20261107.md`
- [FFT-L 局部核心接入与验证](../fft_l_local_core_20261105.md) — `fft_l_local_core_20261105.md`
- [FWHT 原生返回弱点修复（2026-11-07）](../fwht_native_repair_20261107.md) — `fwht_native_repair_20261107.md`
- [应用与固定核心性能合同审计（2026-11-07）](../performance_contract_audit_20261107_applications.md) — `performance_contract_audit_20261107_applications.md`
- [算子性能合同 CPU 审计（2026-11-07）](../performance_contract_audit_20261107_operators.md) — `performance_contract_audit_20261107_operators.md`
- [PRO6000 research handoff, November 7](../pro6000_handoff_20261107.md) — `pro6000_handoff_20261107.md`
- [QuaRot 性能归因审计（2026-11-07）](../quarot_attribution_20261107.md) — `quarot_attribution_20261107.md`
- [QuaRot fused route and performance reporting, November 7](../quarot_fused_path_20261107.md) — `quarot_fused_path_20261107.md`
- [cuButterfly 研究与论文状态（2026-11-04）](../research_status_20261104.md) — `research_status_20261104.md`
- [A100 status and FFT-L scope update](../research_status_20261107.md) — `research_status_20261107.md`
- [静态选型准备开销修复（2026-11-08）](../static_planning_preparation_20261108.md) — `static_planning_preparation_20261108.md`

### 2026-10

- [Application component breakdown audit](../application_component_breakdown_audit_20261012.md) — `application_component_breakdown_audit_20261012.md`
- [Application weak-path repair (2026-10-12)](../application_weakness_repair_20261012.md) — `application_weakness_repair_20261012.md`
- [异步实验队列](../async_experiments_20261009.md) — `async_experiments_20261009.md`
- [Persistent compile queries and grouped measurement workers](../async_fast_dispatch_20261010.md) — `async_fast_dispatch_20261010.md`
- [Bounded GPU workers and checked experiment recovery](../async_recovery_20261010.md) — `async_recovery_20261010.md`
- [Bounded semantic support milestone, 0.9.3](../bounded_support_20261008.md) — `bounded_support_20261008.md`
- [A100 校准与重测入口审计](../calibration_audit_20261012.md) — `calibration_audit_20261012.md`
- [Calibration repair and recovery, 2026-10-14](../calibration_repair_20261014.md) — `calibration_repair_20261014.md`
- [非核心机制的跨算子推广](../common_organization_20261013.md) — `common_organization_20261013.md`
- [公共约束与静态排序回归](../common_planner_validation_20261024.md) — `common_planner_validation_20261024.md`
- [Composition calibration repair, 0.9.3.2](../composition_calibration_20261008.md) — `composition_calibration_20261008.md`
- [候选约束流程核查](../constraint_pipeline_audit_20261024.md) — `constraint_pipeline_audit_20261024.md`
- [Experiment continuation repair, 0.9.2.2](../continuation_repair_20261008.md) — `continuation_repair_20261008.md`
- [核心模板参数贯通与回归记录](../core_template_contract_20261013.md) — `core_template_contract_20261013.md`
- [Current finite calibration and comparison](../current_recalibration_20261012.md) — `current_recalibration_20261012.md`
- [Current Retest Campaign: 2026-10-06](../current_retest_campaign_20261006.md) — `current_retest_campaign_20261006.md`
- [2048², batch 16 xD trace](../e2e_b16_trace_20261017.md) — `e2e_b16_trace_20261017.md`
- [外部计算核心进入公共 xD 选择](../e2e_external_axis_20261013.md) — `e2e_external_axis_20261013.md`
- [xD FFT 端到端优化复测](../e2e_fft_optimization_20261013.md) — `e2e_fft_optimization_20261013.md`
- [端到端性能路径诊断（2026-10-21）](../e2e_framework_hotpath_20261021.md) — `e2e_framework_hotpath_20261021.md`
- [2026-10-13 CUDA Graph 应用复测](../e2e_graph_20261013.md) — `e2e_graph_20261013.md`
- [Application coverage and public selection, 2026-10-13](../e2e_unified_coverage_20261013.md) — `e2e_unified_coverage_20261013.md`
- [Engineering review and comparison revision 0.9.1-review.1](../engineering_review_20261007.md) — `engineering_review_20261007.md`
- [Experiment status for cuButterfly 0.9.3](../experiment_status_20261008_v093.md) — `experiment_status_20261008_v093.md`
- [FFT 固定核心的组织对照](../factor_core_organization_experiment_20261025.md) — `factor_core_organization_experiment_20261025.md`
- [FP64 factor-streamed FFT：CTA 内 tile 分组复测](../factor_graph_exploration_20261025.md) — `factor_graph_exploration_20261025.md`
- [FFT 边界布局与独立 tile 分组实验](../fft_boundary_exploration_20261024.md) — `fft_boundary_exploration_20261024.md`
- [FFT hybrid-dataflow validation on A100](../fft_hybrid_trace_20261022.md) — `fft_hybrid_trace_20261022.md`
- [可替换 FFT 核心与片上驻留的受控实验](../fft_radix4_subgraph_20261024.md) — `fft_radix4_subgraph_20261024.md`
- [长一维 FFT 的阶段内流水核查](../fft_stage_internal_pipeline_20261024.md) — `fft_stage_internal_pipeline_20261024.md`
- [FFT suffix exchange audit](../fft_suffix_exchange_20261023.md) — `fft_suffix_exchange_20261023.md`
- [FFT lag-point trace analysis on A100](../fft_trace_gap_analysis_20261022.md) — `fft_trace_gap_analysis_20261022.md`
- [FHE/CKKS 端到端劣势归因与下一步边界](../fhe_e2e_bottleneck_20261019.md) — `fhe_e2e_bottleneck_20261019.md`
- [Framework alignment audit, 2026-10-12](../framework_alignment_audit_20261012.md) — `framework_alignment_audit_20261012.md`
- [Full Benchmark Refresh, 2026-10-25](../full_benchmark_refresh_20261025.md) — `full_benchmark_refresh_20261025.md`
- [Independent installation comparison, 2026-10-13](../install_compare_20261013.md) — `install_compare_20261013.md`
- [Installation comparison results, 2026-10-17](../install_comparison_results_20261017.md) — `install_comparison_results_20261017.md`
- [JIT runtime scaling investigation](../jit_shared_runtime_20261010.md) — `jit_shared_runtime_20261010.md`
- [Explicit native suffix core, 2026-10-14](../native_suffix_core_20261014.md) — `native_suffix_core_20261014.md`
- [NTT APPT Variant Audit (2026-10-23)](../ntt_appt_variant_audit_20261023.md) — `ntt_appt_variant_audit_20261023.md`
- [NTT 专用机制与公共组织模板审查](../ntt_common_mechanisms_20261023.md) — `ntt_common_mechanisms_20261023.md`
- [NTT historical comparison audit (2026-10-17)](../ntt_comparison_history_20261017.md) — `ntt_comparison_history_20261017.md`
- [NTT comparison and bounded search repair](../ntt_search_repair_20261017.md) — `ntt_search_repair_20261017.md`
- [NTT planning and low-precision inverse repair](../numeric_plan_repair_20261009_v0933.md) — `numeric_plan_repair_20261009_v0933.md`
- [Range-aware inverse normalization and fresh experiment recovery](../numeric_plan_repair_20261009_v0934.md) — `numeric_plan_repair_20261009_v0934.md`
- [Real FFT input-range controls and failure classification](../numeric_range_controls_20261009_v2.md) — `numeric_range_controls_20261009_v2.md`
- [FP16 range diagnosis and reporting repair](../numeric_reporting_repair_20261008.md) — `numeric_reporting_repair_20261008.md`
- [Open-source FFT template exploration, 2026-10-23](../open_source_template_exploration_20261023.md) — `open_source_template_exploration_20261023.md`
- [Operator Coverage Contract (2026-10-06)](../operator_coverage_contract_20261006.md) — `operator_coverage_contract_20261006.md`
- [Parallel A100 80GB continuation](../parallel_resume_20261011.md) — `parallel_resume_20261011.md`
- [性能弱点复核与诊断记录，2026-10-19](../performance_followup_20261019.md) — `performance_followup_20261019.md`
- [NTT physical launch descriptor audit](../physical_launch_descriptor_audit_20261024.md) — `physical_launch_descriptor_audit_20261024.md`
- [Platform boundary and retest repair](../platform_boundary_repair_20261007.md) — `platform_boundary_repair_20261007.md`
- [cuFFTDx provider capability contract](../provider_capability_contract_20261014.md) — `provider_capability_contract_20261014.md`
- [Mature core reuse correction, 2026-10-14](../provider_reuse_20261014.md) — `provider_reuse_20261014.md`
- [Public frontend and dimension audit, 2026-10-07](../public_frontend_dimension_audit_20261007.md) — `public_frontend_dimension_audit_20261007.md`
- [公共 xD radix-4 pair 核心接入](../public_radix4_integration_20261024.md) — `public_radix4_integration_20261024.md`
- [Radix-4 的 FP64、实数和长一维接入](../radix4_completion_20261024.md) — `radix4_completion_20261024.md`
- [Real xD dispatch diagnosis (A100 80 GB)](../real_xd_dispatch_20261012.md) — `real_xd_dispatch_20261012.md`
- [A100 80GB 分层服务校准（2026-10-24）](../recalibration_20261024.md) — `recalibration_20261024.md`
- [Parameterization and recalibration retest](../recalibration_retest_20261007.md) — `recalibration_retest_20261007.md`
- [CKKS RNS NTT 的顺序支持与访存效率](../rns_ckks_boundary_20261018.md) — `rns_ckks_boundary_20261018.md`
- [GPU-NTT 核心对齐记录](../rns_gpu_ntt_core_20261018.md) — `rns_gpu_ntt_core_20261018.md`
- [Search-budget validation](../search_budget_validation_20261011.md) — `search_budget_validation_20261011.md`
- [Descriptor-based retest, 0.9.2](../semantic_campaign_20261007.md) — `semantic_campaign_20261007.md`
- [Semantic coverage audit, 2026-10-12](../semantic_coverage_audit_20261012.md) — `semantic_coverage_audit_20261012.md`
- [Stage calibration adapter repair, 2026-10-16](../stage_adapter_repair_20261016.md) — `stage_adapter_repair_20261016.md`
- [Explicit stage FFT subgraph experiment](../stockham_subgraph_experiment_20261023.md) — `stockham_subgraph_experiment_20261023.md`
- [跨算子模板与参数贯通审查](../template_parameter_audit_20261013.md) — `template_parameter_audit_20261013.md`
- [Weak-case diagnosis and common-path repair](../weak_case_diagnosis_20261017.md) — `weak_case_diagnosis_20261017.md`
- [Weak-region quick replay (2026-10-23)](../weak_regions_quick_20261023.md) — `weak_regions_quick_20261023.md`

### 2026-09

- [全应用方法对应、工程贯通与性能提升审计](../method_implementation_audit_20260912.md) — `method_implementation_audit_20260912.md`
- [Research convergence, 2026-09-19](../research_convergence_20260919.md) — `research_convergence_20260919.md`

## Undated guides and historical summaries

This complete inventory also includes maintained topic guides. Inclusion here
does not mark a guide obsolete; the task entry selects the primary reading routes.

- [项目与作者](../about.md) — `about.md`
- [ADR 0001: Static profiles supply platform facts](../adr/0001-static-profile-driven-planning.md) — `adr/0001-static-profile-driven-planning.md`
- [Architecture decisions](../adr/README.md) — `adr/README.md`
- [API Reference](../api_reference.md) — `api_reference.md`
- [Non-cryptographic application coverage roadmap](../application_coverage_roadmap.md) — `application_coverage_roadmap.md`
- [APPT Static Layout and Online Writer](../appt_static_layout.md) — `appt_static_layout.md`
- [Architecture Guardrails](../architecture_guardrails.md) — `architecture_guardrails.md`
- [Operator-Independent Butterfly Design Space](../butterfly_design_space.md) — `butterfly_design_space.md`
- [C Plan API](../c_api.md) — `c_api.md`
- [Independent CPU preparation for installation search](../calibration_cpu_preparation.md) — `calibration_cpu_preparation.md`
- [Single-GPU Comprehensive Benchmark](../comprehensive_benchmark.md) — `comprehensive_benchmark.md`
- [Comprehensive Butterfly Comparison](../comprehensive_butterfly_comparison.md) — `comprehensive_butterfly_comparison.md`
- [V100 Comprehensive Results](../comprehensive_v100_results.md) — `comprehensive_v100_results.md`
- [Processing-Core Integration and Library Comparison](../core_integration_strategy.md) — `core_integration_strategy.md`
- [External CPU comparisons](../cpu_external_baselines.md) — `cpu_external_baselines.md`
- [Cross-GPU Experiment Matrix](../cross_gpu_experiment.md) — `cross_gpu_experiment.md`
- [Cryptographic butterfly application coverage](../crypto_application_coverage.md) — `crypto_application_coverage.md`
- [Cryptographic extension comparison](../crypto_external_comparison.md) — `crypto_external_comparison.md`
- [cuButterfly Cross-Operator Results](../cubutterfly_cross_operator_results.md) — `cubutterfly_cross_operator_results.md`
- [cuButterfly Feature Coverage](../cubutterfly_feature_coverage.md) — `cubutterfly_feature_coverage.md`
- [Large-Length cuButterfly Results on V100](../cubutterfly_large_results.md) — `cubutterfly_large_results.md`
- [cuButterfly: Scope and Research Positioning](../cubutterfly_positioning.md) — `cubutterfly_positioning.md`
- [cuFFT Xt strategy audit](../cufft_xt_strategy_audit.md) — `cufft_xt_strategy_audit.md`
- [Design Overview](../design_overview.md) — `design_overview.md`
- [Device API](../device_api.md) — `device_api.md`
- [Error Handling](../error_handling.md) — `error_handling.md`
- [Examples](../examples.md) — `examples.md`
- [Experimental Results](../experiments.md) — `experiments.md`
- [CTA DFT8 Space-Time Mapping](../fft_cta_space_time_results.md) — `fft_cta_space_time_results.md`
- [FFT Architecture Design Space](../fft_design_space.md) — `fft_design_space.md`
- [FFT Library Comparison](../fft_library_comparison.md) — `fft_library_comparison.md`
- [FFT Pipeline Generator](../fft_pipeline_generator.md) — `fft_pipeline_generator.md`
- [Unified framework implementation](../framework_refactor_progress.md) — `framework_refactor_progress.md`
- [General-Shape V100 Selection Results](../general_shape_results.md) — `general_shape_results.md`
- [General Shape Mapping](../general_shapes.md) — `general_shapes.md`
- [Generated Processing Units](../generated_design_points.md) — `generated_design_points.md`
- [Getting Started](../getting_started.md) — `getting_started.md`
- [GPU-NTT Gap Analysis on V100](../gpu_ntt_gap_analysis.md) — `gpu_ntt_gap_analysis.md`
- [Hardware Profiles](../hardware/README.md) — `hardware/README.md`
- [<GPU model and form factor> <memory capacity>](../hardware/TEMPLATE.md) — `hardware/TEMPLATE.md`
- [NVIDIA A100 PCIe 40GB](../hardware/a100-pcie-40gb.md) — `hardware/a100-pcie-40gb.md`
- [NVIDIA A100 PCIe 80GB](../hardware/a100-pcie-80gb.md) — `hardware/a100-pcie-80gb.md`
- [RTX PRO 6000 Blackwell Server Edition 96GB](../hardware/pro6000_server96.md) — `hardware/pro6000_server96.md`
- [V100：同步源码后进行本机校准与综合实验](../hardware/v100.md) — `hardware/v100.md`
- [Hardware-Mapped Space-Time NTT Methodology](../hardware_mapping_methodology.md) — `hardware_mapping_methodology.md`
- [Hardware Profile Initialization](../hardware_profile_install.md) — `hardware_profile_install.md`
- [Hierarchical Dataflow NTT](../hierarchical_dataflow_ntt.md) — `hierarchical_dataflow_ntt.md`
- [Homogeneous Two-Level Subgraph Template](../homogeneous_subgraph_template.md) — `homogeneous_subgraph_template.md`
- [Hybrid2D GPU Mapping](../hybrid2d_architecture.md) — `hybrid2d_architecture.md`
- [Hybrid Dataflow NTT](../hybrid_dataflow_ntt.md) — `hybrid_dataflow_ntt.md`
- [Candidate Implementations](../implementation_candidates.md) — `implementation_candidates.md`
- [Integrated repair acceptance: proposed 0.9.1-rc.2](../integrated_acceptance_rc2.md) — `integrated_acceptance_rc2.md`
- [Low-precision local templates](../low_precision_templates.md) — `low_precision_templates.md`
- [Mixed-Dataflow Runtime Migration](../mixed_dataflow_runtime.md) — `mixed_dataflow_runtime.md`
- [Nsight Compute Profiling](../ncu_profiling.md) — `ncu_profiling.md`
- [Numeric-Regime Mapping Study](../next_phase_numeric_regimes.md) — `next_phase_numeric_regimes.md`
- [v0.3.0 Mapping-Selection Milestone](../next_phase_v0.3.md) — `next_phase_v0.3.md`
- [NTT output order](../ntt_output_order.md) — `ntt_output_order.md`
- [Operator Catalog](../operator_catalog.md) — `operator_catalog.md`
- [Physical-Chain Architecture Cost Model](../performance_model.md) — `performance_model.md`
- [Cross-Platform Ports](../platform_ports.md) — `platform_ports.md`
- [Platform Tuning Rules](../platform_tuning.md) — `platform_tuning.md`
- [Processing-Unit Design Space](../processing_unit_design_space.md) — `processing_unit_design_space.md`
- [Programming Guide](../programming_guide.md) — `programming_guide.md`
- [v0.8 Release Overview](../release_v08.md) — `release_v08.md`
- [v0.9 Framework Release](../release_v09.md) — `release_v09.md`
- [研究报告与证据合同](../reporting_contract.md) — `reporting_contract.md`
- [Reproducibility](../reproducibility.md) — `reproducibility.md`
- [Cross-operator research acceptance](../research_acceptance.md) — `research_acceptance.md`
- [cuButterfly 设计与实验原则](../research_principles.md) — `research_principles.md`
- [Research Status: V100 Evidence Closure](../research_status.md) — `research_status.md`
- [Resident M/G Lowering versus v0.6](../resident_v06_comparison.md) — `resident_v06_comparison.md`
- [Prepared RNS NTT integration](../rns_prepared_integration.md) — `rns_prepared_integration.md`
- [Runtime Mapping Selector](../runtime_selector.md) — `runtime_selector.md`
- [Structured 2x2 V100 Results](../structured_2x2_v100_results.md) — `structured_2x2_v100_results.md`
- [cuButterfly Summary And Example](../summary_example.md) — `summary_example.md`
- [Capability And Compatibility Matrix](../support_matrix.md) — `support_matrix.md`
- [Theory and implementation alignment audit (2026-09-10)](../theory_alignment_audit.md) — `theory_alignment_audit.md`
- [Unified planner and specialization modules](../unified_planner.md) — `unified_planner.md`
- [Unified public semantics, 0.9.3](../unified_semantics.md) — `unified_semantics.md`
- [v0.6 Implementation Status](../v0.6_implementation_status.md) — `v0.6_implementation_status.md`
- [v0.8 Nested Physical Space](../v0.8_nested_physical_space.md) — `v0.8_nested_physical_space.md`
- [V100 v0.6/v0.7 Comprehensive Comparison](../v06_v07_comprehensive_comparison.md) — `v06_v07_comprehensive_comparison.md`
- [v0.7 Global-Load Sector Root-Cause Audit](../v07_sector_root_cause.md) — `v07_sector_root_cause.md`
- [v0.6, v0.8, and Library Reconciliation](../v08_v06_cufft_reconciliation.md) — `v08_v06_cufft_reconciliation.md`
- [V100 Matching-Protocol External Baselines](../v100_external_baselines.md) — `v100_external_baselines.md`
- [V100 Initial Results](../v100_initial_results.md) — `v100_initial_results.md`
- [V100 Calibrated Mapping Selector](../v100_mapping_selector.md) — `v100_mapping_selector.md`
- [V100 Hybrid2D Parameter Matrix](../v100_matrix_results.md) — `v100_matrix_results.md`
- [V100 Counter Attribution](../v100_ncu_attribution.md) — `v100_ncu_attribution.md`
- [V100 Orthogonal Length/Batch Scaling](../v100_scaling_results.md) — `v100_scaling_results.md`
- [V100 Library/Base/Search Comparison](../v100_three_way_comparison.md) — `v100_three_way_comparison.md`
- [xD lowering](../xd_lowering.md) — `xd_lowering.md`
- [xD lowering: lessons from related GPU FFT systems](../xd_lowering_related_work.md) — `xd_lowering_related_work.md`
- [文档维护约定](../development/documentation.md) — `development/documentation.md`
- [工程地图](../development/repository-layout.md) — `development/repository-layout.md`
- [Frozen benchmark results and evidence](../benchmarks/results.md) — `benchmarks/results.md`
