cmake_minimum_required(VERSION 3.22)
project(RASR)

include(cmake_resources/ConfigurationTypes.cmake)
include(cmake_resources/Modules.cmake)
include(cmake_resources/CompileOptions.cmake)

option(RASR_BUILD_CHECKS "Build check executables" OFF)

if(${MODULE_PYTHON})
    include(cmake_resources/Python.cmake)
endif()

if(${MODULE_TENSORFLOW})
    include(cmake_resources/Tensorflow.cmake)
endif()

if(${MODULE_ONNX})
    include(cmake_resources/Onnx.cmake)
endif()

string(TOLOWER ${CMAKE_SYSTEM_NAME} SYSTEM_NAME)
string(TOLOWER ${CMAKE_BUILD_TYPE} BUILD_TYPE)
set(ARCH_DESCRIPTION
    "${SYSTEM_NAME}-${CMAKE_HOST_SYSTEM_PROCESSOR}-${BUILD_TYPE}"
)
set(CMAKE_INSTALL_PREFIX
    "${CMAKE_CURRENT_SOURCE_DIR}/arch/${ARCH_DESCRIPTION}/"
)

function(add_install_executable executable_name)
    install(TARGETS ${executable_name} RUNTIME DESTINATION .)
    set_target_properties(
        ${executable_name} PROPERTIES OUTPUT_NAME
                                      "${executable_name}.${ARCH_DESCRIPTION}"
    )
endfunction()

function(add_install_library library_name)
    install(TARGETS ${library_name} LIBRARY DESTINATION .)
endfunction()

function(add_rasr_check_executable executable_name)
    if(RASR_BUILD_CHECKS)
        add_executable(${executable_name} ${ARGN})
    endif()
endfunction()

function(target_link_rasr_check_libraries executable_name)
    if(RASR_BUILD_CHECKS)
        target_link_libraries(${executable_name} ${ARGN})
    endif()
endfunction()

set(SOURCE_VERSION_FILE "${CMAKE_CURRENT_BINARY_DIR}/src/SourceVersion.cc")

add_custom_command(
    OUTPUT "${SOURCE_VERSION_FILE}"
    COMMAND
        ${CMAKE_COMMAND} "-DOUTPUT_FILE=${SOURCE_VERSION_FILE}"
        "-DSOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}" -P
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake_resources/GenerateSourceVersion.cmake"
    COMMENT "Generating source version information"
    VERBATIM
)

add_custom_target(SourceVersion DEPENDS "${SOURCE_VERSION_FILE}")

add_subdirectory(src/)
