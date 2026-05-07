# RASR (RWTH ASR)

RASR is the RWTH Aachen University speech recognition toolkit, developed by the Machine Learning and Human Language
Technology Group at the Chair for Computer Science 6, RWTH Aachen University
(formerly: Human Language Technology and Pattern Recognition).

It contains tools and libraries for building automatic speech recognition systems, including acoustic-model training,
decoding, lattice processing, feature extraction and related speech-processing workflows. \
It is primarily intended for research and advanced ASR experimentation.

## Requirements

RASR can either be built directly on the host system or inside one of the provided Apptainer images.

For a native build, the following tools and libraries are required:

- CMake >= 3.22
- a C++ compiler with C++20 support, for example GCC or Clang
- Bison
- libxml2
- BLAS/LAPACK, for example OpenBLAS
- zlib
- libsndfile

Depending on the enabled modules and tools, additional dependencies may be required, for example
TensorFlow, ONNX Runtime, CUDA, OpenMP, FFmpeg, OpenFST or FLAC, and Python 3, NumPy and pybind11 for `LibRASR`.
Modules and tools can be enabled or disabled through CMake options, see `cmake_resources/Modules.cmake`.

### Apptainer images

For reproducible builds, especially on HPC systems, the repository provides Apptainer definitions in `apptainer/`.
These images contain preconfigured build environments for common RASR setups, including TensorFlow/ONNX-based
configurations.

Use one of these images if you do not want to install all build dependencies manually on the host system.

## Building

Configure and build RASR with CMake:

```
cmake -S . -B build
```

creates the build system in the directory `build/`. \
Useful options:

- `-G Ninja` to use Ninja as build system (usually faster)
- `-DCMAKE_EXPORT_COMPILE_COMMANDS=1` to generate file `compile_commands.json` which can be used for parsing the project
  by some LSPs
- `-DMODULE_<NAME>=0` to disable a module, for example `-DMODULE_TENSORFLOW=0 -DMODULE_LM_TFRNN=0` to disable
  TensorFlow
- `-D<TOOL>=0` to disable a tool, for example `-DLibRASR=0` to disable `LibRASR`
- `-DCMAKE_BUILD_TYPE=<type>` to select the build type. Supported values are `standard`, `debug` and `release`.
  For example, use `-DCMAKE_BUILD_TYPE=debug` to build with debug flags. If unset, RASR uses `standard`.

```
cmake --build build
```

compiles the project. Use the flag `--parallel` for parallel builds and `--target <NAME>` for compiling only a certain
target and its dependencies.

```
cmake --install build
```

installs the executables and libraries in `arch/...`. The installation prefix can be changed with `--prefix`.

## Documentation

Additional documentation is available in `doc/`.

## License and citation

RASR is distributed under the RWTH ASR license. See `RWTH-ASR-License.txt` for details.
If you publish results obtained with RASR, please cite the publication listed in the license file.

(c) 2000-2026 RWTH Aachen University, Lehrstuhl fuer Informatik 6 \
http://www-i6.informatik.rwth-aachen.de/rwth-asr/  \
rwthasr@i6.informatik.rwth-aachen.de