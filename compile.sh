#!/usr/bin/env bash
set -euo pipefail

BUILD_TYPE="${1:-standard}"
BUILD_DIR="cmake-build-${BUILD_TYPE}"

IMAGE="/work/asr4/berger/apptainer/images/torch-2.8_onnx-1.22.sif"

echo "==> Build type: ${BUILD_TYPE}"
echo "==> Build dir : ${BUILD_DIR}"
echo "==> Image     : ${IMAGE}"

mkdir -p "${BUILD_DIR}"

run_in_apptainer() {
    apptainer exec \
        -B ${PWD} \
        --pwd ${PWD} \
        --nv \
        "${IMAGE}" \
        "$@"
}

run_in_apptainer cmake -S . -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" -DCMAKE_EXPORT_COMPILE_COMMANDS=1 -DMODULE_TENSORFLOW=0 -DMODULE_LM_TFRNN=0
mv "${BUILD_DIR}/compile_commands.json" .
run_in_apptainer cmake --build "${BUILD_DIR}" --parallel
run_in_apptainer cmake --install "${BUILD_DIR}"
