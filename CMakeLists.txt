cmake_minimum_required(VERSION 3.14)
project(RASR)

find_library(
  TENSORFLOW_CC
  NAMES tensorflow_cc
  HINTS /usr/local/lib/tensorflow/
  PATH_SUFFIXES lib REQUIRED)

find_library(
  TENSORFLOW_FRAMEWORK
  NAMES tensorflow_framework
  HINTS /usr/local/lib/tensorflow/
  PATH_SUFFIXES lib REQUIRED)

if(TENSORFLOW_CC)
    message(STATUS "TensorFlow CC library found at ${TENSORFLOW_CC}")
else()
    message(FATAL_ERROR "TensorFlow CC library not found")
endif()

if(TENSORFLOW_FRAMEWORK)
    message(STATUS "TensorFlow Framework library found at ${TENSORFLOW_FRAMEWORK}")
else()
    message(FATAL_ERROR "TensorFlow Framework library not found")
endif()

# Find Python interpreter
find_package(PythonInterp REQUIRED)

# Get the TensorFlow include directory by executing a Python command
execute_process(
    COMMAND "${PYTHON_EXECUTABLE}" -c "import tensorflow as tf; print(tf.sysconfig.get_include())"
    RESULT_VARIABLE _tensorflow_include_res
    OUTPUT_VARIABLE TENSORFLOW_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Check if we got the include directory
if(NOT _tensorflow_include_res EQUAL "0")
    message(FATAL_ERROR "Failed to get TensorFlow include directory")
endif()

# Output the found directory for debugging purposes
message(STATUS "TensorFlow include directory: ${TENSORFLOW_INCLUDE_DIR}")

string(TOLOWER ${CMAKE_SYSTEM_NAME} SYSTEM_NAME)
if(CMAKE_BUILD_TYPE MATCHES Debug)
  set(BUILD_SUFFIX "debug")
else()
  set(BUILD_SUFFIX "standard")
endif()
set(ARCH_DESCRIPTION "${SYSTEM_NAME}-${CMAKE_HOST_SYSTEM_PROCESSOR}-${BUILD_SUFFIX}")
set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_SOURCE_DIR}/arch/${ARCH_DESCRIPTION}/")

function(add_install_executable EXECUTABLE_NAME)
  install(TARGETS ${EXECUTABLE_NAME} RUNTIME DESTINATION .)
  set_target_properties(${EXECUTABLE_NAME} PROPERTIES OUTPUT_NAME "${EXECUTABLE_NAME}.${ARCH_DESCRIPTION}") 
endfunction()

add_compile_definitions(CMAKE_DISABLE_MODULES_HH)

function(add_module_option MODULE_NAME DEFAULT_VALUE)
  option(${MODULE_NAME} "Enable module ${MODULE_NAME})" ${DEFAULT_VALUE})
  if(${MODULE_NAME})
    add_compile_definitions(${MODULE_NAME})
    message(STATUS "Module ${MODULE_NAME} is enabled")
  else()
    message(STATUS "Module ${MODULE_NAME} is not enabled")
  endif()
endfunction()

add_module_option(MODULE_AUDIO_FFMPEG ON)
add_module_option(MODULE_CORE_CACHE_MANAGER ON)
add_module_option(MODULE_CART ON)

add_custom_command(
  OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/src/SourceVersion.cc
  COMMAND
    ${CMAKE_COMMAND} -E copy
    ${CMAKE_CURRENT_SOURCE_DIR}/src/SourceVersion.release
    ${CMAKE_CURRENT_SOURCE_DIR}/src/SourceVersion.cc
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/SourceVersion.release)

add_custom_target(SourceVersion
                  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/SourceVersion.cc)

find_package(LibXml2 REQUIRED)
find_package(BISON REQUIRED)
find_package(Threads REQUIRED)
find_package(ZLIB REQUIRED)

include_directories(${CMAKE_SOURCE_DIR}/src)
# include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${LIBXML2_INCLUDE_DIR})

add_subdirectory(src/Cart)
add_subdirectory(src/Core)
add_subdirectory(src/Tools)
