#!/usr/bin/env python3
"""Summarize CKKS pairs by baseline, submission, batch range and timing mode."""
import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

from run_heongpu_ckks_e2e import bootstrap_profile, parse_gpu_processes, target_gpu_uuid

LOOP = ("HEonGPU_native", "independent_public_API_loop")
STAGE_COMPARISONS = (
    LOOP,
    ("HEonGPU_native_loop", "independent_graphs_stage_batch_pack_ntt_scatter"),
    ("HEonGPU_native_stage_batch", "matched_stage_batch_pack_ntt_scatter"),
)
AUTO_COMPARISONS = STAGE_COMPARISONS + (
    ("HEonGPU_native_loop", "measured_auto_loop_or_stage_batch"),
    ("HEonGPU_native_stage_batch", "measured_auto_loop_or_stage_batch"),
)
ORGANIZATIONS = {
    LOOP: ("HEonGPU 逐密文调用", "cuButterfly 逐密文调用"),
    STAGE_COMPARISONS[1]: ("HEonGPU 逐密文调用", "cuButterfly 跨密文阶段合批（含 gather/scatter）"),
    STAGE_COMPARISONS[2]: ("HEonGPU 跨密文阶段合批（本轮原生核心控制，含 gather/scatter）", "cuButterfly 跨密文阶段合批（含 gather/scatter）"),
    AUTO_COMPARISONS[3]: ("HEonGPU 逐密文调用", "cuButterfly 完整 Graph 实测自动选择"),
    AUTO_COMPARISONS[4]: ("HEonGPU 跨密文阶段合批（本轮原生核心控制，含 gather/scatter）", "cuButterfly 完整 Graph 实测自动选择"),
}
SCENES = {"keyswitch": "密钥切换", "square_relinearize_rescale": "平方→重线性化→rescale",
          "square_rescale_keyswitch": "平方→重线性化→rescale→密钥切换",
          "rotate_rows_shift1": "槽旋转 +1"}
SCENE_LABELS = {**SCENES, "regular_bootstrapping_v2": "完整 CKKS regular bootstrap v2"}


def option(argv, name, default):
    if name not in argv:
        return default
    position = argv.index(name) + 1
    if position == len(argv):
        raise ValueError(f"missing benchmark argument for {name}")
    return argv[position]


def positive_int(value):
    result = int(value)
    if result < 1:
        raise ValueError(f"expected positive integer, got {value!r}")
    return result


def parse_rows(rows, meta, path):
    """Validate the exact requested matrix and normalize legacy optional fields."""
    argv = meta["benchmark_argv"]
    parameter_set = option(argv, "--parameter-set", "short-chain")
    bootstrap_profiles = {"bootstrap-v2": "primary", "bootstrap-v2-mid": "bootstrap_v2_mid",
                          "bootstrap-v2-small": "bootstrap_v2_small"}
    if parameter_set not in ("short-chain", *bootstrap_profiles):
        raise ValueError(f"unknown parameter-set in {path}")
    bootstrap = parameter_set in bootstrap_profiles
    profile = bootstrap_profile(bootstrap_profiles[parameter_set]) if bootstrap else None
    ns = [positive_int(n) for n in option(argv, "--sizes", "4096,16384").split(",")]
    bs = [positive_int(b) for b in option(argv, "--batches", "1,8").split(",")]
    if len(set(ns)) != len(ns) or len(set(bs)) != len(bs):
        raise ValueError(f"duplicate requested size or batch in {path}")
    scenes = list(SCENES) if "--rotate" in argv else list(SCENES)[:3]
    if bootstrap:
        scenes = ["regular_bootstrapping_v2"]
    scenes = option(argv, "--scenes", ",".join(scenes)).split(",")
    allowed_scenes = {"regular_bootstrapping_v2"} if bootstrap else SCENES.keys()
    if len(set(scenes)) != len(scenes) or any(scene not in allowed_scenes for scene in scenes):
        raise ValueError(f"invalid requested scenes in {path}")
    if "rotate_rows_shift1" in scenes and "--rotate" not in argv:
        raise ValueError(f"rotation scene requires --rotate in {path}")
    stage_batch = "--stage-batch" in argv
    auto = "--auto-schedule" in argv
    if auto and not stage_batch:
        raise ValueError(f"auto-schedule requires stage-batch in {path}")
    comparisons = AUTO_COMPARISONS if auto else STAGE_COMPARISONS if stage_batch else (LOOP,)
    mode = "cuda_graph" if "--graph" in argv else "eager"
    execution = option(argv, "--ntt-execution", None)
    if stage_batch and (mode != "cuda_graph" or execution not in (None, "native-rns")):
        raise ValueError(f"invalid stage-batch benchmark configuration in {path}")
    if stage_batch:
        execution = "native-rns"
    repetitions = positive_int(option(argv, "--repetitions", "10"))
    replays = positive_int(option(argv, "--graph-replays", "1"))
    aux = option(argv, "--aux-primes", "1")
    if aux not in ((str(profile["p_primes"]),) if bootstrap else ("1", "2")):
        raise ValueError(f"invalid aux-primes in {path}")
    if bootstrap and (ns != [profile["N"]] or "--rotate" in argv):
        raise ValueError(f"bootstrap parameter shape differs in {path}")
    expected = {(scene, n, b, baseline, submission)
                for scene in scenes for n in ns for b in bs
                for baseline, submission in comparisons}
    indexed = {}
    normalized = []
    for source in rows:
        row = dict(source)
        row.setdefault("baseline", LOOP[0])
        row.setdefault("batch_submission", LOOP[1])
        row.setdefault("ntt_execution", "grouped")
        row.setdefault("graph_replays_per_sample", "1")
        row.setdefault("parameter_set", "short-chain")
        identity = (row["scene"], positive_int(row["N"]), positive_int(row["batch"]),
                    row["baseline"], row["batch_submission"])
        if identity not in expected:
            raise ValueError(f"unexpected matrix identity in {path}: {identity}")
        if identity in indexed:
            raise ValueError(f"duplicate matrix identity in {path}: {identity}")
        required = {"mode": mode, "keyswitch_method": "I" if aux == "1" else "II",
                    "parameter_set": parameter_set}
        if execution is not None:
            required["ntt_execution"] = execution
        for key, value in required.items():
            if row.get(key) != value:
                raise ValueError(f"wrong configuration {key} in {path}: {row.get(key)!r}, expected {value!r}")
        for key, value in {"type": "NTT", "sec_level": "none",
                           "Q_bits": profile["Q_bits"] if bootstrap else "50:40:40",
                           "P_bits": profile["P_bits"] if bootstrap else ("50" if aux == "1" else "50:50"),
                           "scale": "1099511627776"}.items():
            if key in row and row[key] != value:
                raise ValueError(f"wrong configuration {key} in {path}")
        if row["ntt_execution"] not in ("native-rns", "grouped"):
            raise ValueError(f"invalid NTT execution mode in {path}: {row['ntt_execution']!r}")
        if "repetitions" in row and positive_int(row["repetitions"]) != repetitions:
            raise ValueError(f"wrong sample count in {path}")
        replay_count = int(row["graph_replays_per_sample"])
        if (mode == "cuda_graph" and replay_count != replays) or (mode == "eager" and replay_count not in (0, 1)):
            raise ValueError(f"wrong graph replay count in {path}")
        native, treatment = float(row["native_ms"]), float(row["cubutterfly_ms"])
        if not (math.isfinite(native) and math.isfinite(treatment) and native > 0 and treatment > 0):
            raise ValueError(f"invalid paired timing in {path}")
        ratio, stored_ratio = native / treatment, float(row["speedup"])
        if not (math.isfinite(ratio) and ratio > 0 and math.isfinite(stored_ratio)):
            raise ValueError(f"invalid paired speedup in {path}")
        if not math.isclose(ratio, stored_ratio, rel_tol=1e-8):
            raise ValueError(f"stored speedup disagrees with paired times in {path}")
        if row["exact_ciphertext"] not in ("0", "1") or int(row["mismatch_count"]) < 0:
            raise ValueError(f"invalid ciphertext verification in {path}")
        if (row["exact_ciphertext"] == "1") != (int(row["mismatch_count"]) == 0):
            raise ValueError(f"inconsistent ciphertext verification in {path}")
        indexed[identity] = row
        normalized.append(row)
    missing = expected - indexed.keys()
    if missing:
        raise ValueError(f"incomplete matrix in {path}: missing {len(missing)} identities, including {sorted(missing)[0]}")
    if stage_batch:
        for scene in scenes:
            for n in ns:
                for b in bs:
                    loop, mixed, matched = [indexed[(scene, n, b, *comparison)] for comparison in STAGE_COMPARISONS]
                    # These are two views of the same measured variants, not
                    # independently sampled baselines or treatments.
                    for left, right, prefix in ((loop, mixed, "native"), (mixed, matched, "cubutterfly")):
                        fields = [f"{prefix}_ms", f"{prefix}_kernel_nodes"]
                        suffix = "native" if prefix == "native" else "cu"
                        fields += [f"maxabs_{suffix}", f"relL2_{suffix}"]
                        if prefix == "cubutterfly":
                            fields += ["replacement_calls", "polynomials"]
                        for field in fields:
                            if left.get(field) != right.get(field):
                                raise ValueError(f"shared stage measurement {field} differs in {path}: {(scene, n, b)}")
                    if auto:
                        auto_loop, auto_batch = [indexed[(scene, n, b, *comparison)] for comparison in AUTO_COMPARISONS[3:]]
                        for left, right, prefix in ((loop, auto_loop, "native"), (matched, auto_batch, "native"),
                                                    (auto_loop, auto_batch, "cubutterfly")):
                            for field in (f"{prefix}_ms", f"{prefix}_kernel_nodes"):
                                if left.get(field) != right.get(field):
                                    raise ValueError(f"shared auto measurement {field} differs in {path}: {(scene, n, b)}")
    return normalized


def validate_selection_evidence(rows, meta, path):
    """Validate separate selection/holdout files and return request-level regrets.

    CSV records establish distinct measurement phases, not statistical independence
    of the device state. Equal numeric timings across phases are legal.
    """
    argv = meta["benchmark_argv"]
    if "--auto-schedule" not in argv:
        return []
    rows = parse_rows(rows, meta, path)
    rounds = positive_int(option(argv, "--schedule-rounds", "7"))
    selection_replays = positive_int(option(argv, "--schedule-replays", "5"))
    repetitions = positive_int(option(argv, "--repetitions", "10"))
    holdout_replays = positive_int(option(argv, "--graph-replays", "1"))
    requests = {(row["scene"], int(row["N"]), int(row["batch"]), row["keyswitch_method"]) for row in rows}

    def finite_positive(value, field):
        number = float(value)
        if not math.isfinite(number) or number <= 0:
            raise ValueError(f"invalid positive finite {field} in {path}")
        return number

    def read_phase(suffix, kind, variants, count, replays, fields):
        source = Path(str(path) + suffix)
        artifact_name = "selection_samples" if kind == "candidate" else "holdout_samples"
        artifact = meta.get("artifacts", {}).get(artifact_name)
        if artifact is not None:
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            if artifact.get("sha256") != digest or artifact.get("size_bytes") != source.stat().st_size:
                raise ValueError(f"{kind} sidecar differs from metadata artifact in {source}")
        expected = {(request, variant, r) for request in requests for variant in variants for r in range(count)}
        indexed = {}
        with source.open() as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or len(reader.fieldnames) != len(fields) or set(reader.fieldnames) != set(fields):
                raise ValueError(f"wrong {kind} sidecar schema in {source}")
            for row in reader:
                if None in row or any(value is None for value in row.values()):
                    raise ValueError(f"malformed {kind} sidecar row in {source}")
                request = (row["scene"], positive_int(row["N"]), positive_int(row["batch"]), row["keyswitch_method"])
                identity = (request, row[kind], int(row["round"]))
                if identity not in expected:
                    raise ValueError(f"unexpected {kind} identity in {source}: {identity}")
                if identity in indexed:
                    raise ValueError(f"duplicate {kind} identity in {source}: {identity}")
                if positive_int(row["replays"]) != replays:
                    raise ValueError(f"wrong {kind} replay budget in {source}")
                finite_positive(row["ms"], f"{kind} ms")
                indexed[identity] = row
        if expected != indexed.keys():
            raise ValueError(f"incomplete {kind} sidecar in {source}")
        return indexed

    common = ["scene", "N", "batch", "keyswitch_method", "round", "ms", "replays"]
    selection = read_phase(".selection.csv", "candidate", ("loop", "stage_batch"), rounds, selection_replays,
                           common + ["candidate", "median_ms", "selected", "selection_wall_ms", "confidence"])
    variants = ("native_loop", "cu_loop", "native_stage_batch", "cu_stage_batch", "cu_auto")
    holdout = read_phase(".samples.csv", "variant", variants, repetitions, holdout_replays, common + ["variant"])
    indexed_rows = {(row["scene"], int(row["N"]), int(row["batch"]), row["keyswitch_method"],
                     row["baseline"], row["batch_submission"]): row for row in rows}
    result = []
    for request in sorted(requests):
        medians, flags, walls = [], [], []
        for candidate in ("loop", "stage_batch"):
            samples = [selection[(request, candidate, r)] for r in range(rounds)]
            median = statistics.median(float(sample["ms"]) for sample in samples)
            medians.append(median)
            selected = {sample["selected"] for sample in samples}
            if len(selected) != 1 or not selected <= {"0", "1"}:
                raise ValueError(f"inconsistent candidate selected flags in {path}: {request}")
            flags.append(selected == {"1"})
            for sample in samples:
                stored = finite_positive(sample["median_ms"], "median_ms")
                if not math.isclose(stored, median, rel_tol=1e-8):
                    raise ValueError(f"selection median disagrees with samples in {path}: {request}")
                if sample["confidence"] != "measured-full-graph-request":
                    raise ValueError(f"wrong selection confidence in {path}")
                walls.append(finite_positive(sample["selection_wall_ms"], "selection_wall_ms"))
        winner = 0 if medians[0] <= medians[1] else 1
        if flags != [winner == 0, winner == 1]:
            raise ValueError(f"selected candidate disagrees with median in {path}: {request}")
        if any(wall != walls[0] for wall in walls):
            raise ValueError(f"selection wall time differs within request in {path}: {request}")
        means = {variant: statistics.mean(float(holdout[(request, variant, r)]["ms"]) for r in range(repetitions))
                 for variant in variants}
        pairs = (("native_loop", "cu_loop"), ("native_loop", "cu_stage_batch"),
                 ("native_stage_batch", "cu_stage_batch"), ("native_loop", "cu_auto"),
                 ("native_stage_batch", "cu_auto"))
        for comparison, (native, treatment) in zip(AUTO_COMPARISONS, pairs):
            row = indexed_rows[(*request, *comparison)]
            if any(not math.isclose(float(row[field]), means[variant], rel_tol=1e-8)
                   for field, variant in (("native_ms", native), ("cubutterfly_ms", treatment))):
                raise ValueError(f"holdout mean disagrees with main CSV in {path}: {request}")
        best = min(means["cu_loop"], means["cu_stage_batch"])
        result.append(dict(zip(("scene", "N", "batch", "keyswitch_method"), request),
                           selected=("loop", "stage_batch")[winner], selection_wall_ms=walls[0],
                           selection_medians_ms=medians, auto_ms=means["cu_auto"], best_fixed_ms=best,
                           regret=means["cu_auto"] / best - 1))
    return result


def group_rows(rows, by_scene):
    groups = defaultdict(list)
    for row in rows:
        key = (row["baseline"], row["batch_submission"],
               "B=1" if int(row["batch"]) == 1 else "B>1",
               row["mode"], row["ntt_execution"], row["keyswitch_method"],
               row["graph_replays_per_sample"])
        if by_scene:
            key += (row["scene"],)
        groups[key].append(float(row["native_ms"]) / float(row["cubutterfly_ms"]))
    return groups


def require_performance_evidence(meta, path):
    if meta.get("status") != "completed" or meta.get("returncode") != 0:
        raise ValueError(f"completed successful run metadata is required for {path}")
    if "performance_eligible" in meta:
        if meta["performance_eligible"] is not True:
            raise ValueError(f"performance evidence is ineligible for {path}")
        return
    try:
        gpu_uuid = target_gpu_uuid(meta)
        for name in ("compute_processes_before", "compute_processes_after"):
            probe = meta.get(name, {})
            if probe.get("returncode") != 0 or not isinstance(probe.get("stdout"), str):
                raise ValueError(f"{name} query is missing or failed")
            foreign = parse_gpu_processes(probe["stdout"], gpu_uuid, meta.get("benchmark_pid"))
            if foreign:
                pids = ", ".join(str(process["pid"]) for process in foreign)
                raise ValueError(f"{name} records foreign target-GPU PIDs: {pids}")
    except (TypeError, ValueError, csv.Error) as error:
        raise ValueError(f"unusable legacy performance evidence for {path}: {error}") from error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    lines = ["# CKKS 完整调用对比", "", "项目作者与归属见 [项目说明](../../docs/about.md)。",
             "", "每项使用相同输入密文、密钥、精度和 CUDA stream。加速比为原库延迟 / cuButterfly 适配延迟，小于 1 表示适配路径较慢。",
             "", "Graph 行为完整公共调用的 GPU event 时间；eager 行为包含分配、提交和同步的主机墙钟时间。两种口径分别报告。",
             "", "短链参数为 Q=[50,40,40]、P=[50] 或 [50,50]；bootstrap-v2 使用固定上游示例的 25 个 Q 和 5 个 P 素数。实际位数见各 CSV，不能用示例注释代替素数的 bit_length。两者均 scale=2^40、sec_level=none，属于功能和集成性能测试，未作生产安全参数验证。",
             "", "batch 表示同一 stream 上处理的独立密文数。逐密文调用保留公共 API 循环；阶段合批保留各密文完整 DAG，在对应 NTT 阶段执行 gather→合批 NTT→scatter，计时包含数据打包与还原。",
             "", "阶段合批记录三种比较：双方逐密文调用、原库逐密文调用对 cuButterfly 阶段合批、双方阶段合批。三者分别统计；双方合批行同时检验原库合批控制的整数密文一致性。所有延迟均对应整批相同工作量。B=1 与 B>1 分开报告。",
             "", "原生核心的阶段合批控制由本轮相同 Graph lowering 构造，不代表上游已提供跨密文合批 API。它与 cuButterfly 使用同一打包、依赖重建及非 NTT 节点组织。",
             "", "NTT 执行模式取自 CSV 的 ntt_execution；缺少该列的旧记录按 grouped（旧版）读取。统计按对照组织、适配组织、batch 范围、执行模式、计时模式与重放次数分组，新路径的性能只由对应记录支持。",
             "", "## 实测结果", ""]
    total = 0
    all_valid = True
    for path in args.inputs:
        with path.open() as handle:
            rows = list(csv.DictReader(handle))
        metadata_path = path.with_suffix(".metadata.json")
        meta = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
        require_performance_evidence(meta, path)
        rows = parse_rows(rows, meta, path)
        selections = validate_selection_evidence(rows, meta, path) if "--auto-schedule" in meta["benchmark_argv"] else []
        expected_rows = len(rows)
        lines += [f"### {path.name}", "", f"原始记录：[CSV]({path.name}) · [日志]({path.with_suffix('.log').name}) · [元数据]({metadata_path.name})。",
                  "", f"已完成 {len(rows)}/{expected_rows} 条结果。GPU：`{meta.get('CUDA_VISIBLE_DEVICES')}`。该文件的样本单独统计，不与其他设备或参数链混合。",
                  "", f"温控协议：`{meta.get('thermal_protocol', 'legacy-see-log')}`；二进制身份见该文件元数据。", "",
                  "| 场景（N，batch） | 计时 / NTT 执行 / 方法 | 对照组织 | 适配组织 | 样本数 / 每样本 graph 重放 | 原库 ms | 适配 ms | 加速比 | 原库/适配 kernel 数 | 整数密文一致 |",
                  "|---|---|---|---|---|---:|---:|---:|---:|---|"]
        for row in rows:
            native, treatment = float(row["native_ms"]), float(row["cubutterfly_ms"])
            ratio = native / treatment
            exact = row["exact_ciphertext"] == "1" and row["mismatch_count"] == "0"
            all_valid &= exact
            mode, method = row["mode"], row["keyswitch_method"]
            execution = row["ntt_execution"]
            label = SCENE_LABELS[row["scene"]]
            kernels = f"{row.get('native_kernel_nodes', '')}/{row.get('cubutterfly_kernel_nodes', '')}" if mode == "cuda_graph" else "未采集"
            baseline, treatment_label = ORGANIZATIONS[(row["baseline"], row["batch_submission"])]
            samples = f"{row.get('repetitions', '未记录')} / {row['graph_replays_per_sample']}"
            lines.append(f"| {label}（{row['N']}，{row['batch']}） | {mode} / {execution} / {method} | {baseline} | {treatment_label} | {samples} | {native:.6f} | {treatment:.6f} | {ratio:.3f}× | {kernels} | {'是' if exact else '否'} |")
        if selections:
            lines += ["", "#### 完整 Graph 自动选择与独立复测", "",
                      f"选拔证据：[selection CSV]({path.name}.selection.csv)；独立复测：[samples CSV]({path.name}.samples.csv)。",
                      "选拔使用完整 Graph 的每候选轮次中位数；随后独立复测五种执行变体，主表延迟为复测均值。选拔墙钟时间单列，不计入稳态延迟。结果仅供当前实例复用，未持久化。",
                      "regret = 自动路径复测均值 / 同请求逐密文与阶段合批复测均值的较小值 − 1；保留实际负值，不将选拔样本当作独立复测。", "",
                      "| 场景（N，batch） / 方法 | 选中候选 | 选拔 median ms（loop / stage_batch） | 选拔墙钟 ms | 自动路径独立复测 ms | 固定候选最佳独立复测 ms | regret |",
                      "|---|---|---:|---:|---:|---:|---:|"]
            for selection in selections:
                medians = selection["selection_medians_ms"]
                lines.append(f"| {SCENE_LABELS[selection['scene']]}（{selection['N']}，{selection['batch']}） / {selection['keyswitch_method']} | {selection['selected']} | {medians[0]:.6f} / {medians[1]:.6f} | {selection['selection_wall_ms']:.6f} | {selection['auto_ms']:.6f} | {selection['best_fixed_ms']:.6f} | {selection['regret']:.3%} |")
        for by_scene, heading in ((False, "按对照组织汇总"), (True, "按场景与对照组织汇总")):
            lines += ["", f"#### {heading}", "",
                      "| 场景 | 对照组织 | 适配组织 | batch 范围 | 计时 / NTT 执行 / 方法 | 每样本 graph 重放 | 配对数 | 加速比范围 | 几何均值 | 胜出数 |",
                      "|---|---|---|---|---|---:|---:|---:|---:|---:|"]
            for key, ratios in group_rows(rows, by_scene).items():
                baseline, submission, batch_range, mode, execution, method, replays = key[:7]
                reference_label, treatment_label = ORGANIZATIONS[(baseline, submission)]
                scene = SCENE_LABELS[key[7]] if by_scene else "全部场景"
                gm = math.exp(sum(map(math.log, ratios)) / len(ratios))
                lines.append(f"| {scene} | {reference_label} | {treatment_label} | {batch_range} | {mode} / {execution} / {method} | {replays} | {len(ratios)} | {min(ratios):.3f}–{max(ratios):.3f}× | {gm:.3f}× | {sum(r >= 1 for r in ratios)} |")
        if rows:
            error = max(float(r["maxabs_cu"]) for r in rows)
            rel = max(float(r["relL2_cu"]) for r in rows)
            equal_error = all(r["maxabs_cu"] == r["maxabs_native"] and r["relL2_cu"] == r["relL2_native"] for r in rows)
            lines += ["", f"解密输出相对 CPU double 明文运算的最大绝对误差为 {error:.4g}，最大相对 L2 为 {rel:.4g}。两后端误差报告相同：{equal_error}。浮点误差仅报告，未暗设验收阈值。", ""]
        total += len(rows)
    lines += ["## 证据边界", "", f"合计 {total} 条配对结果，全部整数密文一致：{all_valid}。",
              "", "替换覆盖完整操作内的正逆 NTT、根排列、negacyclic twist 与布局转换。非蝶形的基转换、乘累加、模降及密钥沿用 HEonGPU。",
              "", "选择来源保存在运行日志的 ntt_adapter_plan 和 rns_stage_batch_plan 行。native-rns 在冷准备阶段调用 RnsPlan::tune，测量该构建已接入的合法核心、分段、行分组、线程和边界映射，并复测前三名。具体候选范围以各次日志及源码身份为准，不用新版本的候选范围解释旧数据。confidence=measured-rns-request 标识请求级实测；selection_calibrated=0 表示未使用安装级成本模型，与实际请求调优可以同时成立。该搜索不覆盖尚未接入的核心族。",
              "", "grouped 包含逐模数 pack/unpack 和所选核心的成本，选择依据沿用 cuntt::Plan 日志（旧字段 calibrated，新字段 selection_calibrated）。native-rns 将边界变换融合在两遍内；性能差异需按对应模式的记录分析。公共 RnsPlan 是新增 C++ API，尚未纳入公共 C ABI 或通用 planner 的全局 RNS 选择。",
              "", "只有 regular_bootstrapping_v2 行代表完整 CKKS bootstrap；其他行仍是完整操作或短链，不能外推为 bootstrap。TFHE 的设备适配与结果独立记录；其 standalone key switching 没有 NTT，不能用它宣称蝶形加速。", ""]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines))
    print(f"wrote {args.output}: {total} pairs, exact={all_valid}")
    return 0 if all_valid and total else 1


if __name__ == "__main__":
    raise SystemExit(main())
