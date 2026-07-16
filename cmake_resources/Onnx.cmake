set(ONNXRUNTIME_ROOT
    ""
    CACHE PATH "Optional root directory of an ONNX Runtime installation"
)

find_path(
    onnxruntime_INCLUDE_DIR
    NAMES onnxruntime_cxx_api.h
    HINTS ${ONNXRUNTIME_ROOT}
    PATH_SUFFIXES include include/onnxruntime
)

find_library(
    onnxruntime_LIBRARY
    NAMES onnxruntime
    HINTS ${ONNXRUNTIME_ROOT}
    PATH_SUFFIXES lib lib64
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
    ONNXRuntime REQUIRED_VARS onnxruntime_LIBRARY onnxruntime_INCLUDE_DIR
)

add_library(ONNXRuntime::ONNXRuntime UNKNOWN IMPORTED)
set_target_properties(
    ONNXRuntime::ONNXRuntime
    PROPERTIES IMPORTED_LOCATION "${onnxruntime_LIBRARY}"
               INTERFACE_INCLUDE_DIRECTORIES "${onnxruntime_INCLUDE_DIR}"
)
