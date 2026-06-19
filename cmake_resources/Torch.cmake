find_package(Python3 REQUIRED COMPONENTS Interpreter)

set(_torch_root "" CACHE PATH "Optional root directory of a LibTorch installation")
set(_torch_cmake_prefix "" CACHE PATH "Optional Torch CMake prefix directory")
set(_torch_lib_dir "" CACHE PATH "Optional Torch library directory")

# Optional manual override
if (_torch_root)
    list(PREPEND CMAKE_PREFIX_PATH "${_torch_root}")
endif ()

# Auto-discover Torch from the active Python environment
if (NOT _torch_cmake_prefix)
    execute_process(
        COMMAND
        "${Python3_EXECUTABLE}" -c
        "import torch; print(torch.utils.cmake_prefix_path)"
        RESULT_VARIABLE _torch_cmake_prefix_res
        OUTPUT_VARIABLE _torch_cmake_prefix
        ERROR_VARIABLE _torch_cmake_prefix_err
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    if (NOT _torch_cmake_prefix_res EQUAL 0)
        message(
            FATAL_ERROR
            "MODULE_TORCH=ON, but Python could not import torch.\n"
            "Python executable: ${Python3_EXECUTABLE}\n"
            "Error:\n${_torch_cmake_prefix_err}"
        )
    endif ()
endif ()

message(STATUS "Torch CMake prefix: ${_torch_cmake_prefix}")
list(PREPEND CMAKE_PREFIX_PATH "${_torch_cmake_prefix}")

find_package(Torch REQUIRED CONFIG)

if (NOT _torch_lib_dir)
    execute_process(
        COMMAND
        "${Python3_EXECUTABLE}" -c
        "import torch.utils.cpp_extension as e; print(e.TORCH_LIB_PATH)"
        RESULT_VARIABLE _torch_lib_dir_res
        OUTPUT_VARIABLE _torch_lib_dir
        ERROR_VARIABLE _torch_lib_dir_err
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    if (NOT _torch_lib_dir_res EQUAL 0)
        message(
            FATAL_ERROR
            "Could not determine Torch library directory via torch.utils.cpp_extension.\n"
            "Python executable: ${Python3_EXECUTABLE}\n"
            "Error:\n${_torch_lib_dir_err}"
        )
    endif ()
endif ()

message(STATUS "Torch library directory: ${_torch_lib_dir}")
message(STATUS "Torch include dirs: ${TORCH_INCLUDE_DIRS}")
message(STATUS "Torch libraries: ${TORCH_LIBRARIES}")
message(STATUS "Torch CXX flags: ${TORCH_CXX_FLAGS}")

add_library(RasrExternalTorch INTERFACE)

if (TARGET torch)
    target_link_libraries(RasrExternalTorch INTERFACE torch)
else ()
    target_include_directories(RasrExternalTorch INTERFACE ${TORCH_INCLUDE_DIRS})
    target_link_libraries(RasrExternalTorch INTERFACE ${TORCH_LIBRARIES})
endif ()

if (TORCH_CXX_FLAGS)
    separate_arguments(_torch_cxx_flags NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
    target_compile_options(RasrExternalTorch INTERFACE ${_torch_cxx_flags})
endif ()

# For running RASR binaries without manually setting LD_LIBRARY_PATH
target_link_options(RasrExternalTorch INTERFACE "LINKER:-rpath,${_torch_lib_dir}")