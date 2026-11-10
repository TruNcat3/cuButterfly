#!/usr/bin/env bash
set -euo pipefail

root_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
build_type=${BUILD_TYPE:-Release}
arch=${CUDA_ARCHITECTURES:-80}
out_file=${APPT_RESULT_FILE:-"${root_dir}/results/appt_layout_sweep.csv"}
log_dir=${APPT_LOG_DIR:-"${root_dir}/results/appt_layout_logs"}

mkdir -p "${root_dir}/results"
mkdir -p "${log_dir}"
printf 'n_part_mode,raw_benchmark_csv\n' > "${out_file}"

for mode in remaining local; do
  build_dir="${root_dir}/build-appt-${mode}"
  local_flag=0
  [[ "${mode}" == local ]] && local_flag=1
  config_log="${log_dir}/${mode}-configure.log"
  build_log="${log_dir}/${mode}-build.log"
  run_log="${log_dir}/${mode}-run.log"
  cmake -S "${root_dir}" -B "${build_dir}" \
    -DCMAKE_BUILD_TYPE="${build_type}" \
    -DCMAKE_CUDA_ARCHITECTURES="${arch}" \
    -DCMAKE_CUDA_FLAGS="-DCUBUTTERFLY_APPT_LAYOUT=1 -DCUBUTTERFLY_APPT_NPART_LOCAL=${local_flag}" \
    -DCMAKE_CXX_FLAGS="-DCUBUTTERFLY_APPT_LAYOUT=1 -DCUBUTTERFLY_APPT_NPART_LOCAL=${local_flag}" \
    -DCUBUTTERFLY_ENABLE_CUFFTDX=ON >"${config_log}" 2>&1 || {
      echo "APPT ${mode}: configure failed; see ${config_log}" >&2
      exit 1
    }
  cmake --build "${build_dir}" -j"${JOBS:-2}" --target cubutterfly_bench >"${build_log}" 2>&1 || {
    echo "APPT ${mode}: build failed; see ${build_log}" >&2
    exit 1
  }

  for shape in '12 16' '18 64'; do
    read -r log_n batch <<<"${shape}"
    raw_log="${log_dir}/${mode}-logN${log_n}-batch${batch}.log"
    set +e
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "${build_dir}/cubutterfly_bench" \
      --operator fft --backend online-reorder --fft-core cufftdx-block \
      --logN "${log_n}" --batch "${batch}" --placement in-place --local-stages "$((log_n / 2))" \
      --prefix-threads 256 --suffix-threads 256 --prefix-ept 8 --suffix-ept 8 \
      --reorder-columns 1 --cross-twiddle recurrence --warmup 1 --repeat 2 --verify --csv \
      >"${raw_log}" 2>&1
    bench_rc=$?
    set -e
    line=$(tail -n 1 "${raw_log}")
    [[ -n "${line}" && "${line}" == *,* ]] || {
      echo "APPT ${mode} logN=${log_n} batch=${batch}: no CSV (rc=${bench_rc}); see ${raw_log}" >&2
      exit 1
    }
    printf '%s,"%s"\n' "${mode}" "${line//\"/\"\"}" >> "${out_file}"
  done
done

cat "${out_file}"
