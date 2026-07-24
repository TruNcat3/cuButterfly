#!/usr/bin/env python3
import argparse
import csv
import itertools
import json
import math
import pathlib


REQUIRED_OBJECTS = {
    "G_graph_semantics", "A_architecture", "P_processing_unit", "L_layout",
    "F_realization", "Q_generation_selection", "H_hardware",
}
REQUIRED_OPERATORS = {"fft", "ntt", "fwht", "xor-zeta", "subset-mobius", "structured-butterfly"}


def load_space(path):
    document = json.loads(path.read_text())
    if document.get("schema_version") != 1:
        raise ValueError("unsupported butterfly architecture-space schema")
    missing_objects = REQUIRED_OBJECTS - set(document.get("objects", {}))
    if missing_objects:
        raise ValueError(f"missing architecture objects: {sorted(missing_objects)}")
    projections = set(document.get("operator_projections", {}))
    missing_operators = REQUIRED_OPERATORS - projections
    if missing_operators:
        raise ValueError(f"missing operator projections: {sorted(missing_operators)}")
    if not document.get("implementation_families") or not document.get("invariants"):
        raise ValueError("architecture space must define implementation families and invariants")
    architecture = document["objects"]["A_architecture"]
    for axis in ("batch_space_Ub", "batch_time_Tb", "hierarchical_unfolding", "coefficient_supply"):
        if axis not in architecture:
            raise ValueError(f"architecture space lacks required axis: {axis}")
    dimension_fields = {
        "stage_count", "stage_space_Us", "stage_time_Ts", "data_space_Ud", "data_time_Td",
        "stage_handoff_Hs", "stage_time_residence_Rs", "data_time_residence_Rd",
        "processing_unit", "local_layout", "realization",
    }
    missing_dimension_fields = dimension_fields - set(architecture["per_dimension_mapping"])
    if missing_dimension_fields:
        raise ValueError(f"per-dimension mapping lacks fields: {sorted(missing_dimension_fields)}")
    semantic_operators = set(document["objects"]["G_graph_semantics"]["operator"])
    if semantic_operators != projections:
        raise ValueError("graph operator enumeration and operator projections differ")
    hierarchy_levels = set(architecture["hierarchical_unfolding"]["levels"])
    hardware_levels = set(document["objects"]["H_hardware"]["hierarchy"])
    if hierarchy_levels != hardware_levels:
        raise ValueError("architecture and hardware hierarchy levels differ")
    processing = document["objects"]["P_processing_unit"]
    coefficient_forms = set(processing["coefficient_form"])
    for name, projection in document["operator_projections"].items():
        unknown = set(projection["coefficient"]) - coefficient_forms
        if unknown:
            raise ValueError(f"operator {name} uses unknown coefficient forms: {sorted(unknown)}")
        if not projection.get("traits") or not projection.get("cores"):
            raise ValueError(f"operator {name} lacks traits or cores")
    core_bindings = document.get("processing_core_bindings", {})
    if set(core_bindings) != projections:
        raise ValueError("processing-core bindings must cover every operator projection")
    for operator, bindings in core_bindings.items():
        projection = document["operator_projections"][operator]
        if set(bindings) != set(projection["cores"]):
            raise ValueError(f"processing-core bindings differ from {operator} core projection")
        for core, binding in bindings.items():
            if binding["stage_group"] not in processing["stage_group"]:
                raise ValueError(f"core {operator}/{core} uses unknown stage group")
            if binding["arithmetic_core"] not in processing["arithmetic_core"]:
                raise ValueError(f"core {operator}/{core} uses unknown arithmetic core")
            if not set(binding.get("precision", projection["precision"])) <= set(projection["precision"]):
                raise ValueError(f"core {operator}/{core} uses invalid precision")
            if not set(binding["coefficient"]) <= set(projection["coefficient"]):
                raise ValueError(f"core {operator}/{core} uses invalid coefficients")
            if not set(binding["exchange"]) <= set(processing["local_exchange"]):
                raise ValueError(f"core {operator}/{core} uses invalid local exchange")
    precision_bindings = document.get("precision_bindings", {})
    if set(precision_bindings) != projections:
        raise ValueError("precision bindings must cover every operator projection")
    for operator, bindings in precision_bindings.items():
        if set(bindings) != set(document["operator_projections"][operator]["precision"]):
            raise ValueError(f"precision bindings differ from {operator} projection")
        for precision, binding in bindings.items():
            if not set(binding["word_bits"]) <= set(processing["word_bits"]):
                raise ValueError(f"precision {operator}/{precision} uses invalid word width")
            if not set(binding["accumulator_bits"]) <= set(processing["accumulator_bits"]):
                raise ValueError(f"precision {operator}/{precision} uses invalid accumulator width")
    families = set(document["implementation_families"])
    kernel_forms = set(document["objects"]["F_realization"]["kernel_form"])
    if not families <= kernel_forms:
        raise ValueError(f"implementation families lack realization forms: {sorted(families - kernel_forms)}")
    runtime_bindings = document.get("runtime_bindings", {})
    if set(runtime_bindings) != {"ntt_backends", "butterfly_backends"}:
        raise ValueError("runtime bindings must cover NTT and common butterfly APIs")
    for binding_group, bindings in runtime_bindings.items():
        unknown = set(bindings.values()) - families
        if unknown:
            raise ValueError(f"runtime binding {binding_group} uses unknown families: {sorted(unknown)}")
    return document


def validate_fft_projection(general, fft):
    projection = general["operator_projections"]["fft"]
    axes = fft["axes"]
    checks = {
        "precision": (set(axes["precision"]), set(projection["precision"])),
        "direction": (set(axes["direction"]), set(general["objects"]["G_graph_semantics"]["direction"])),
        "placement": (set(axes["placement"]), set(general["objects"]["G_graph_semantics"]["placement"])),
        "boundary": (set(axes["boundary"]), set(general["objects"]["A_architecture"]["dimension_boundary"])),
        "reorder": (set(axes["reorder"]), set(general["objects"]["L_layout"]["intermediate_permutation"])),
        "core": (set(axes["dimension"]["core"]), set(projection["cores"])),
        "exchange": (set(axes["dimension"]["exchange"]), set(general["objects"]["P_processing_unit"]["local_exchange"])),
    }
    twiddle_map = {"table": "native-table", "recurrence": "recurrence", "fused-core": "generated", "generated": "generated"}
    checks["cross_twiddle"] = ({twiddle_map[value] for value in axes["cross_twiddle"]}, set(projection["coefficient"]))
    failures = [name for name, (selected, allowed) in checks.items() if not selected <= allowed]
    if failures:
        raise ValueError(f"FFT space is not a valid general-space projection: {failures}")
    return True


def derive_architecture_point(space, operator, log_n, us, ud, word_bytes, stage_residence, data_residence,
                              batch=1, ub=1, batch_residence="global"):
    if operator not in space["operator_projections"]:
        raise ValueError(f"unknown operator: {operator}")
    if log_n < 1 or us < 1 or ud < 1 or word_bytes < 1 or batch < 1 or ub < 1:
        raise ValueError("logN, Us, Ud, Ub, batch, and word bytes must be positive")
    n = 1 << log_n
    butterflies = n // 2
    ts = math.ceil(log_n / us)
    td = math.ceil(butterflies / ud)
    tb = math.ceil(batch / ub)
    traits = space["operator_projections"][operator]["traits"]
    coefficient_streams = us * ud * ub if "stage-coefficient" in traits else 0
    stage_feedback_folds = max(0, ts - 1)
    data_stream_folds = max(0, td - 1)
    on_chip = {"register", "shared", "cluster-shared"}
    stage_feedback_bytes = 0 if stage_residence in on_chip else 2 * n * batch * word_bytes * stage_feedback_folds
    data_feedback_bytes = 0 if data_residence in on_chip else 2 * n * batch * word_bytes * data_stream_folds
    return {
        "operator": operator,
        "logN": log_n,
        "N": n,
        "storage_word_bytes": word_bytes,
        "butterflies_per_stage": butterflies,
        "Us": us,
        "Ts": ts,
        "Ud": ud,
        "Td": td,
        "batch": batch,
        "Ub": ub,
        "Tb": tb,
        "stage_utilization": log_n / (us * ts),
        "data_utilization": butterflies / (ud * td),
        "batch_utilization": batch / (ub * tb),
        "total_utilization": (log_n / (us * ts)) * (butterflies / (ud * td)) * (batch / (ub * tb)),
        "spatial_butterfly_cells": us * ud * ub,
        "ideal_body_cycles": ts * td * tb,
        "boundary_coefficient_words_per_cycle": 2 * ud * ub,
        "interstage_coefficient_words_per_cycle": 2 * ud * max(0, us - 1) * ub,
        "coefficient_streams_upper_bound": coefficient_streams,
        "stage_feedback_folds": stage_feedback_folds,
        "data_stream_folds": data_stream_folds,
        "stage_feedback_bytes_above_residence": stage_feedback_bytes,
        "data_feedback_bytes_above_residence": data_feedback_bytes,
        "stage_residence": stage_residence,
        "data_residence": data_residence,
        "batch_residence": batch_residence,
        "status": "abstract"
    }


def derive_dimension_point(stage_count, independent_work, us, ud):
    if min(stage_count, independent_work, us, ud) < 1:
        raise ValueError("dimension stage count, work, Us, and Ud must be positive")
    ts = math.ceil(stage_count / us)
    td = math.ceil(independent_work / ud)
    return {
        "stage_count": stage_count,
        "independent_work": independent_work,
        "Us": us,
        "Ts": ts,
        "Ud": ud,
        "Td": td,
        "stage_utilization": stage_count / (us * ts),
        "data_utilization": independent_work / (ud * td),
    }


def validate_hierarchy_factors(us, ud, ub, level_factors):
    products = {"Us": 1, "Ud": 1, "Ub": 1}
    for level, factors in level_factors.items():
        for axis in products:
            value = factors.get(axis, 1)
            if not isinstance(value, int) or value < 1:
                raise ValueError(f"invalid {axis} factor at hierarchy level {level}")
            products[axis] *= value
    expected = {"Us": us, "Ud": ud, "Ub": ub}
    if products != expected:
        raise ValueError(f"hierarchy products {products} do not match logical factors {expected}")
    return True


def derive_multidimensional_point(log_n, partition, mappings):
    if sum(partition) != log_n or len(partition) != len(mappings) or any(value < 1 for value in partition):
        raise ValueError("ordered dimension partition and mappings must cover logN exactly")
    dimensions = []
    for stage_count, mapping in zip(partition, mappings):
        local_work = 1 << max(0, stage_count - 1)
        dimensions.append(derive_dimension_point(stage_count, local_work, mapping["Us"], mapping["Ud"]))
    return {
        "logN": log_n,
        "factorization_rank": len(partition),
        "dimension_stage_counts": list(partition),
        "dimensions": dimensions,
        "dimension_boundaries": max(0, len(partition) - 1),
    }


def expand_design_choices(space, operator, selections):
    projection = space["operator_projections"][operator]
    graph = space["objects"]["G_graph_semantics"]
    architecture = space["objects"]["A_architecture"]
    processing = space["objects"]["P_processing_unit"]
    layout = space["objects"]["L_layout"]
    realization = space["objects"]["F_realization"]
    generation = space["objects"]["Q_generation_selection"]
    axes = {
        "direction": graph["direction"],
        "normalization": graph["normalization"],
        "placement": graph["placement"],
        "input_order": graph["input_order"],
        "output_order": graph["output_order"],
        "schedule": architecture["schedule"],
        "tail_policy": architecture["tail_policy"],
        "coefficient_residence": architecture["coefficient_supply"]["residence"],
        "coefficient_sharing_scope": architecture["coefficient_supply"]["sharing_scope"],
        "coefficient_schedule": architecture["coefficient_supply"]["schedule"],
        "elements_per_thread": processing["elements_per_thread"],
        "units_per_cta": processing["units_per_cta"],
        "dimension_boundary": architecture["dimension_boundary"],
        "global_order": layout["global_order"],
        "intermediate_permutation": layout["intermediate_permutation"],
        "bank_mapping": layout["bank_mapping"],
        "shared_padding": layout["shared_padding"],
        "vector_width_bytes": layout["vector_width_bytes"],
        "alignment_bytes": layout["alignment_bytes"],
        "kernel_form": realization["kernel_form"],
        "threads_per_cta": realization["threads_per_cta"],
        "warps_per_cta": realization["warps_per_cta"],
        "ctas_per_work_unit": realization["ctas_per_work_unit"],
        "pipeline_buffers": realization["pipeline_buffers"],
        "async_copy": realization["async_copy"],
        "synchronization": realization["synchronization"],
        "specialization": generation["specialization"],
        "selection_method": generation["selection"],
        "candidate_status": generation["status"],
        "objective": generation["objectives"],
    }
    positive_axes = ("element_stride", "batch_stride", "pipeline_initiation_interval", "pipeline_depth")
    known_selections = set(axes) | set(positive_axes) | {
        "precision", "word_bits", "accumulator_bits", "operator_core", "stage_group",
        "arithmetic_core", "coefficient_form", "local_exchange",
    }
    unknown_selections = set(selections) - known_selections
    if unknown_selections:
        raise ValueError(f"unknown design-choice axes: {sorted(unknown_selections)}")
    selected = {}
    for name, allowed in axes.items():
        values = selections.get(name) or [allowed[0]]
        unknown = set(values) - set(allowed)
        if unknown:
            raise ValueError(f"unknown {name} choices for {operator}: {sorted(unknown, key=str)}")
        selected[name] = values
    for name in positive_axes:
        values = selections.get(name) or [1]
        if any(not isinstance(value, int) or value < 1 for value in values):
            raise ValueError(f"{name} choices must be positive integers")
        selected[name] = values
    cores = selections.get("operator_core") or [projection["cores"][0]]
    unknown_cores = set(cores) - set(projection["cores"])
    if unknown_cores:
        raise ValueError(f"unknown operator_core choices for {operator}: {sorted(unknown_cores)}")
    requested_stage_groups = selections.get("stage_group")
    requested_arithmetic = selections.get("arithmetic_core")
    requested_coefficients = selections.get("coefficient_form")
    requested_exchanges = selections.get("local_exchange")
    requested_precisions = selections.get("precision")
    requested_word_bits = selections.get("word_bits")
    requested_accumulator_bits = selections.get("accumulator_bits")
    for name, values, allowed in (
        ("precision", requested_precisions, projection["precision"]),
        ("word_bits", requested_word_bits, processing["word_bits"]),
        ("accumulator_bits", requested_accumulator_bits, processing["accumulator_bits"]),
        ("stage_group", requested_stage_groups, processing["stage_group"]),
        ("arithmetic_core", requested_arithmetic, processing["arithmetic_core"]),
        ("coefficient_form", requested_coefficients, projection["coefficient"]),
        ("local_exchange", requested_exchanges, processing["local_exchange"]),
    ):
        unknown = set(values or ()) - set(allowed)
        if unknown:
            raise ValueError(f"unknown {name} choices for {operator}: {sorted(unknown)}")
    names = list(selected)
    independent = [dict(zip(names, values)) for values in itertools.product(*(selected[name] for name in names))]
    points = []
    bindings = space["processing_core_bindings"][operator]
    precision_bindings = space["precision_bindings"][operator]
    for core in cores:
        binding = bindings[core]
        if requested_stage_groups and binding["stage_group"] not in requested_stage_groups:
            continue
        if requested_arithmetic and binding["arithmetic_core"] not in requested_arithmetic:
            continue
        coefficients = requested_coefficients or [binding["coefficient"][0]]
        exchanges = requested_exchanges or [binding["exchange"][0]]
        precisions = requested_precisions or [binding.get("precision", projection["precision"])[0]]
        precisions = [value for value in precisions if value in binding.get("precision", projection["precision"])]
        coefficients = [value for value in coefficients if value in binding["coefficient"]]
        exchanges = [value for value in exchanges if value in binding["exchange"]]
        for precision in precisions:
            precision_binding = precision_bindings[precision]
            word_bits = requested_word_bits or [precision_binding["word_bits"][0]]
            accumulator_bits = requested_accumulator_bits or [precision_binding["accumulator_bits"][0]]
            word_bits = [value for value in word_bits if value in precision_binding["word_bits"]]
            accumulator_bits = [value for value in accumulator_bits if value in precision_binding["accumulator_bits"]]
            for coefficient, exchange, word_bit, accumulator_bit, point in itertools.product(
                    coefficients, exchanges, word_bits, accumulator_bits, independent):
                points.append({
                    "precision": precision,
                    "word_bits": word_bit,
                    "accumulator_bits": accumulator_bit,
                    "operator_core": core,
                    "stage_group": binding["stage_group"],
                    "arithmetic_core": binding["arithmetic_core"],
                    "coefficient_form": coefficient,
                    "local_exchange": exchange,
                    **point,
                })
    if not points:
        raise ValueError(f"processing-unit filters produce no legal choices for {operator}")
    return points


def powers_up_to(limit):
    values = []
    value = 1
    while value <= limit:
        values.append(value)
        value *= 2
    if limit not in values:
        values.append(limit)
    return values


def stage_partitions(total, rank):
    if rank < 1 or total < rank:
        return []
    if rank == 1:
        return [(total,)]
    return [
        (first, *suffix)
        for first in range(1, total - rank + 2)
        for suffix in stage_partitions(total - first, rank - 1)
    ]


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Validate and enumerate operator-independent butterfly architecture points.")
    parser.add_argument("--spec", type=pathlib.Path, default=pathlib.Path("config/butterfly_architecture_space.json"))
    parser.add_argument("--fft-spec", type=pathlib.Path, default=pathlib.Path("config/fft_architecture_space.json"))
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--operator", choices=sorted(REQUIRED_OPERATORS), default="fft")
    parser.add_argument("--logN", type=int, default=16)
    parser.add_argument("--Us", nargs="+", type=int)
    parser.add_argument("--Ud", nargs="+", type=int)
    parser.add_argument("--fixed-spatial-cells", type=int,
                        help="Enumerate Us*Ud=C factorizations instead of the Us/Ud Cartesian product.")
    parser.add_argument("--factorization-rank", type=int, default=2)
    parser.add_argument("--word-bytes", type=int,
                        help="Override storage bytes; otherwise derive from each candidate's word_bits.")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--Ub", nargs="+", type=int, default=(1,))
    parser.add_argument("--stage-residence", nargs="+", default=("shared", "global"))
    parser.add_argument("--data-residence", nargs="+", default=("shared", "global"))
    parser.add_argument("--operator-core", nargs="+")
    parser.add_argument("--stage-group", nargs="+")
    parser.add_argument("--arithmetic-core", nargs="+")
    parser.add_argument("--coefficient-form", nargs="+")
    parser.add_argument("--local-exchange", nargs="+")
    parser.add_argument("--boundary", nargs="+")
    parser.add_argument("--permutation", nargs="+")
    parser.add_argument("--kernel-form", nargs="+")
    parser.add_argument("--threads", nargs="+", type=int)
    parser.add_argument("--pipeline-buffers", nargs="+", type=int)
    parser.add_argument("--synchronization", nargs="+")
    parser.add_argument("--specialization", nargs="+")
    parser.add_argument("--objective", nargs="+")
    parser.add_argument("--choices", type=pathlib.Path,
                        help="JSON object of additional axis lists; explicit CLI lists take precedence.")
    parser.add_argument("--max-candidates", type=int, default=100000)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()

    space = load_space(args.spec)
    fft = json.loads(args.fft_spec.read_text())
    validate_fft_projection(space, fft)
    if args.validate_only:
        return
    us_values = args.Us or powers_up_to(args.logN)
    ud_values = args.Ud or powers_up_to(min((1 << args.logN) // 2, 1024))
    if args.fixed_spatial_cells:
        architecture_pairs = [(us, args.fixed_spatial_cells // us) for us in us_values
                              if args.fixed_spatial_cells % us == 0]
    else:
        architecture_pairs = [(us, ud) for us in us_values for ud in ud_values]
    choice_selections = json.loads(args.choices.read_text()) if args.choices else {}
    explicit_choices = {
        "operator_core": args.operator_core,
        "stage_group": args.stage_group,
        "arithmetic_core": args.arithmetic_core,
        "coefficient_form": args.coefficient_form,
        "local_exchange": args.local_exchange,
        "dimension_boundary": args.boundary,
        "intermediate_permutation": args.permutation,
        "kernel_form": args.kernel_form,
        "threads_per_cta": args.threads,
        "pipeline_buffers": args.pipeline_buffers,
        "synchronization": args.synchronization,
        "specialization": args.specialization,
        "objective": args.objective,
    }
    choice_selections.update({name: values for name, values in explicit_choices.items() if values is not None})
    choices = expand_design_choices(space, args.operator, choice_selections)
    candidate_count = (len(stage_partitions(args.logN, args.factorization_rank)) *
                       len(architecture_pairs) * len(args.Ub) * len(args.stage_residence) *
                       len(args.data_residence) * len(choices))
    if candidate_count > args.max_candidates:
        raise ValueError(f"candidate product {candidate_count} exceeds --max-candidates={args.max_candidates}")
    rows = []
    for partition in stage_partitions(args.logN, args.factorization_rank):
        for us, ud in architecture_pairs:
            for ub in args.Ub:
                for rs in args.stage_residence:
                    for rd in args.data_residence:
                        for choice in choices:
                            if args.word_bytes is not None:
                                word_bytes = args.word_bytes
                            elif isinstance(choice["word_bits"], int):
                                word_bytes = math.ceil(choice["word_bits"] / 8)
                            else:
                                raise ValueError("--word-bytes is required for operator-defined word width")
                            point = derive_architecture_point(space, args.operator, args.logN, us, ud,
                                                              word_bytes, rs, rd, args.batch, ub)
                            point["factorization_rank"] = args.factorization_rank
                            point["dimension_stage_counts"] = "+".join(str(value) for value in partition)
                            rows.append({**point, **choice})
    if args.output:
        write_csv(args.output, rows)
    else:
        print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
