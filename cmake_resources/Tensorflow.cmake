find_package(Python3 REQUIRED COMPONENTS Interpreter)

set(_tensorflow_root
    ""
    CACHE PATH "Optional root directory of a TensorFlow C++ installation"
)
set(_tensorflow_include_dir
    ""
    CACHE PATH "TensorFlow include directory"
)
set(_tensorflow_lib_dir
    ""
    CACHE PATH "TensorFlow library directory"
)

# Get the TensorFlow include directory by executing a Python command
if(NOT _tensorflow_include_dir)
    execute_process(
        COMMAND "${Python3_EXECUTABLE}" -c
                "import tensorflow as tf; print(tf.sysconfig.get_include())"
        RESULT_VARIABLE _tensorflow_include_res
        OUTPUT_VARIABLE _tensorflow_include_dir
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    if(NOT _tensorflow_include_res EQUAL "0")
        message(FATAL_ERROR "Failed to get TensorFlow include directory")
    endif()
endif()

if(NOT _tensorflow_lib_dir)
    execute_process(
        COMMAND "${Python3_EXECUTABLE}" -c
                "import tensorflow as tf; print(tf.sysconfig.get_lib())"
        RESULT_VARIABLE _tensorflow_lib_res
        OUTPUT_VARIABLE _tensorflow_lib_dir
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    if(NOT _tensorflow_lib_res EQUAL "0")
        message(FATAL_ERROR "Failed to get TensorFlow library directory")
    endif()
endif()

# Output the found directory for debugging purposes
message(STATUS "Tensorflow include directory: ${_tensorflow_include_dir}")

find_library(
    Tensorflow_CC
    NAMES tensorflow_cc libtensorflow_cc.so.2 libtensorflow_cc.so.1
    HINTS ${_tensorflow_lib_dir} ${_tensorflow_root}
    PATH_SUFFIXES lib lib64 tensorflow REQUIRED
)

find_library(
    Tensorflow_FRAMEWORK
    NAMES tensorflow_framework libtensorflow_framework.so.2
          libtensorflow_framework.so.1
    HINTS ${_tensorflow_lib_dir} ${_tensorflow_root}
    PATH_SUFFIXES lib lib64 tensorflow REQUIRED
)

if(Tensorflow_CC)
    message(STATUS "Tensorflow CC library found at ${Tensorflow_CC}")
else()
    message(FATAL_ERROR "Tensorflow CC library not found")
endif()

if(Tensorflow_FRAMEWORK)
    message(
        STATUS "Tensorflow Framework library found at ${Tensorflow_FRAMEWORK}"
    )
else()
    message(FATAL_ERROR "TensorFlow Framework library not found")
endif()

find_package(OpenSSL REQUIRED)

add_library(Tensorflow::CC UNKNOWN IMPORTED)
set_target_properties(
    Tensorflow::CC
    PROPERTIES IMPORTED_LOCATION "${Tensorflow_CC}"
               INTERFACE_INCLUDE_DIRECTORIES "${_tensorflow_include_dir}"
)

add_library(Tensorflow::Framework UNKNOWN IMPORTED)
set_target_properties(
    Tensorflow::Framework
    PROPERTIES IMPORTED_LOCATION "${Tensorflow_FRAMEWORK}"
               INTERFACE_INCLUDE_DIRECTORIES "${_tensorflow_include_dir}"
)

add_library(Tensorflow::Tensorflow INTERFACE IMPORTED)
set_target_properties(
    Tensorflow::Tensorflow
    PROPERTIES INTERFACE_LINK_LIBRARIES
               "OpenSSL::Crypto;Tensorflow::CC;Tensorflow::Framework"
               INTERFACE_LINK_OPTIONS
               "LINKER:--no-as-needed;LINKER:--allow-multiple-definition"
)
