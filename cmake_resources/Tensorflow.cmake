find_package(Python3 REQUIRED COMPONENTS Interpreter)

set(Tensorflow_ROOT
    ""
    CACHE PATH "Optional root directory of a TensorFlow C++ installation"
)
set(Tensorflow_INCLUDE_DIR
    ""
    CACHE PATH "TensorFlow include directory"
)
set(Tensorflow_LIB_DIR
    ""
    CACHE PATH "TensorFlow library directory"
)

# Get the TensorFlow include directory by executing a Python command
if(NOT Tensorflow_INCLUDE_DIR)
    execute_process(
        COMMAND "${Python3_EXECUTABLE}" -c
                "import tensorflow as tf; print(tf.sysconfig.get_include())"
        RESULT_VARIABLE _tensorflow_include_res
        OUTPUT_VARIABLE Tensorflow_INCLUDE_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    if(NOT _tensorflow_include_res EQUAL "0")
        message(FATAL_ERROR "Failed to get TensorFlow include directory")
    endif()
endif()

if(NOT Tensorflow_LIB_DIR)
    execute_process(
        COMMAND "${Python3_EXECUTABLE}" -c
                "import tensorflow as tf; print(tf.sysconfig.get_lib())"
        RESULT_VARIABLE _tensorflow_lib_res
        OUTPUT_VARIABLE Tensorflow_LIB_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    if(NOT _tensorflow_lib_res EQUAL "0")
        message(FATAL_ERROR "Failed to get TensorFlow library directory")
    endif()
endif()

# Output the found directory for debugging purposes
message(STATUS "Tensorflow include directory: ${Tensorflow_INCLUDE_DIR}")

find_library(
    Tensorflow_CC
    NAMES tensorflow_cc libtensorflow_cc.so.2 libtensorflow_cc.so.1
    HINTS ${Tensorflow_LIB_DIR} ${Tensorflow_ROOT}
    PATH_SUFFIXES lib lib64 tensorflow REQUIRED
)

find_library(
    Tensorflow_FRAMEWORK
    NAMES tensorflow_framework libtensorflow_framework.so.2
          libtensorflow_framework.so.1
    HINTS ${Tensorflow_LIB_DIR} ${Tensorflow_ROOT}
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
               INTERFACE_INCLUDE_DIRECTORIES "${Tensorflow_INCLUDE_DIR}"
)

add_library(Tensorflow::Framework UNKNOWN IMPORTED)
set_target_properties(
    Tensorflow::Framework
    PROPERTIES IMPORTED_LOCATION "${Tensorflow_FRAMEWORK}"
               INTERFACE_INCLUDE_DIRECTORIES "${Tensorflow_INCLUDE_DIR}"
)

function(add_tf_dependencies target)
    target_link_options(
        ${target} PUBLIC "LINKER:--no-as-needed"
        "LINKER:--allow-multiple-definition"
    )
    target_link_libraries(
        ${target} PUBLIC OpenSSL::Crypto Tensorflow::CC Tensorflow::Framework
    )
endfunction()
