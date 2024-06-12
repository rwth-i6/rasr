#!/bin/bash
mkdir -p cmake-build-standard
pushd cmake-build-standard || exit
apptainer exec -B /u/berger -B /work/asr4/berger /work/asr4/berger/apptainer/images/i6_tensorflow-2.8_onnx-1.15.sif /u/berger/tools/clion-2023.2.2/bin/cmake/linux/x64/bin/cmake .. -DCMAKE_BUILD_TYPE=standard -G Ninja -DCMAKE_MAKE_PROGRAM=/u/berger/tools/clion-2023.2.2/bin/ninja/linux/x64/ninja
apptainer exec -B /u/berger -B /work/asr4/berger /work/asr4/berger/apptainer/images/i6_tensorflow-2.8_onnx-1.15.sif /u/berger/tools/clion-2023.2.2/bin/cmake/linux/x64/bin/cmake --build . --target all -j 10
apptainer exec -B /u/berger -B /work/asr4/berger /work/asr4/berger/apptainer/images/i6_tensorflow-2.8_onnx-1.15.sif /u/berger/tools/clion-2023.2.2/bin/cmake/linux/x64/bin/cmake --install .
popd || exit
