find_package(Python3 REQUIRED COMPONENTS Interpreter)

# Get the TensorFlow include directory by executing a Python command
execute_process(
        COMMAND "${Python3_EXECUTABLE}" -c
        "import tensorflow as tf; print(tf.sysconfig.get_include())"
        RESULT_VARIABLE _tensorflow_include_res
        OUTPUT_VARIABLE Tensorflow_INCLUDE_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE)

execute_process(
        COMMAND "${Python3_EXECUTABLE}" -c
        "import tensorflow as tf; print(tf.sysconfig.get_lib())"
        RESULT_VARIABLE _tensorflow_lib_res
        OUTPUT_VARIABLE Tensorflow_LIB_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE)

# Check if we got the include directory
if (NOT _tensorflow_include_res EQUAL "0")
        message(FATAL_ERROR "Failed to get TensorFlow include directory")
endif ()

# Output the found directory for debugging purposes
message(STATUS "Tensorflow include directory: ${Tensorflow_INCLUDE_DIR}")

set(CMAKE_FIND_LIBRARY_SUFFIXES .so .so.1 .so.2)

find_library(
        Tensorflow_CC
        NAMES tensorflow_cc
        HINTS ${Tensorflow_LIB_DIR} /usr/local/lib/tensorflow
        PATH_SUFFIXES lib REQUIRED)

find_library(
        Tensorflow_FRAMEWORK
        NAMES tensorflow_framework
        HINTS ${Tensorflow_LIB_DIR} /usr/local/lib/tensorflow
        PATH_SUFFIXES lib REQUIRED)

if (Tensorflow_CC)
        message(STATUS "Tensorflow CC library found at ${Tensorflow_CC}")
else ()
        message(FATAL_ERROR "Tensorflow CC library not found")
endif ()

if (Tensorflow_FRAMEWORK)
        message(
            STATUS "Tensorflow Framework library found at ${Tensorflow_FRAMEWORK}")
else ()
        message(FATAL_ERROR "TensorFlow Framework library not found")
endif ()

find_package(OpenSSL REQUIRED)

function(add_tf_dependencies TARGET)
        target_include_directories(${TARGET} PUBLIC ${Tensorflow_INCLUDE_DIR})
        target_link_options(${TARGET} PUBLIC "LINKER:--no-as-needed" "LINKER:--allow-multiple-definition")
        target_link_libraries(${TARGET} PUBLIC OpenSSL::Crypto)
        target_link_libraries(${TARGET} PUBLIC ${Tensorflow_CC} ${Tensorflow_FRAMEWORK})
endfunction()
