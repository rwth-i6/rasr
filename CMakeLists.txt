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

function(add_install_executable EXECUTABLE_NAME)
    install(TARGETS ${EXECUTABLE_NAME} RUNTIME DESTINATION .)
    set_target_properties(
        ${EXECUTABLE_NAME} PROPERTIES OUTPUT_NAME
                                      "${EXECUTABLE_NAME}.${ARCH_DESCRIPTION}"
    )
endfunction()

function(add_install_library LIBRARY_NAME)
    install(TARGETS ${LIBRARY_NAME} LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX})
endfunction()

function(add_rasr_check_executable EXECUTABLE_NAME)
    if(RASR_BUILD_CHECKS)
        add_executable(${EXECUTABLE_NAME} ${ARGN})
    endif()
endfunction()

function(target_link_rasr_check_libraries EXECUTABLE_NAME)
    if(RASR_BUILD_CHECKS)
        target_link_libraries(${EXECUTABLE_NAME} ${ARGN})
    endif()
endfunction()

set(SOURCE_VERSION_FILE "${CMAKE_CURRENT_BINARY_DIR}/src/SourceVersion.cc")
set(SOURCE_VERSION_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake_resources/GenerateSourceVersion.cmake"
)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/SourceVersion.release")
    list(APPEND SOURCE_VERSION_DEPENDS
         "${CMAKE_CURRENT_SOURCE_DIR}/src/SourceVersion.release"
    )
endif()

add_custom_command(
    OUTPUT "${SOURCE_VERSION_FILE}"
    COMMAND
        ${CMAKE_COMMAND} "-DOUTPUT_FILE=${SOURCE_VERSION_FILE}"
        "-DSOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}" -P
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake_resources/GenerateSourceVersion.cmake"
    DEPENDS ${SOURCE_VERSION_DEPENDS}
    COMMENT "Generating source version information"
    VERBATIM
)

add_custom_target(SourceVersion DEPENDS "${SOURCE_VERSION_FILE}")

add_subdirectory(src/)
