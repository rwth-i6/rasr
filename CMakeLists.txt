cmake_minimum_required(VERSION 3.22)
project(RASR)

include(cmake_resources/ConfigurationTypes.cmake)
include(cmake_resources/Modules.cmake)
include(cmake_resources/GeneralCompileOptions.cmake)

if (${MODULE_PYTHON})
    include(cmake_resources/Python.cmake)
endif ()

if (${MODULE_TENSORFLOW})
    include(cmake_resources/Tensorflow.cmake)
endif ()

if (${MODULE_ONNX})
    include(cmake_resources/Onnx.cmake)
endif ()

string(TOLOWER ${CMAKE_SYSTEM_NAME} SYSTEM_NAME)
string(TOLOWER ${CMAKE_BUILD_TYPE} BUILD_TYPE)
set(ARCH_DESCRIPTION
        "${SYSTEM_NAME}-${CMAKE_HOST_SYSTEM_PROCESSOR}-${BUILD_TYPE}")
set(CMAKE_INSTALL_PREFIX
        "${CMAKE_CURRENT_SOURCE_DIR}/arch/${ARCH_DESCRIPTION}/")
set(INSTALL_LIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/lib/${ARCH_DESCRIPTION}")

function(add_install_executable EXECUTABLE_NAME)
    install(TARGETS ${EXECUTABLE_NAME} RUNTIME DESTINATION .)
    set_target_properties(
            ${EXECUTABLE_NAME} PROPERTIES OUTPUT_NAME
            "${EXECUTABLE_NAME}.${ARCH_DESCRIPTION}")
endfunction()

function(add_install_library LIBRARY_NAME)
    install(TARGETS ${LIBRARY_NAME} LIBRARY DESTINATION ${INSTALL_LIB_DIR})
endfunction()

add_custom_command(
        OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/src/SourceVersion.cc
        COMMAND
        ${CMAKE_COMMAND} -E copy
        ${CMAKE_CURRENT_SOURCE_DIR}/src/SourceVersion.release
        ${CMAKE_CURRENT_SOURCE_DIR}/src/SourceVersion.cc
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/SourceVersion.release)

add_custom_target(SourceVersion
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/SourceVersion.cc)

add_subdirectory(src/)
