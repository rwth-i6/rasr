cmake_minimum_required(VERSION 3.14)
project(RASR)

#set(CMAKE_VERBOSE_MAKEFILE ON CACHE BOOL "ON" FORCE)

include(cmake_resources/ConfigurationTypes.cmake)
include(cmake_resources/Modules.cmake)
include(cmake_resources/Python.cmake)
include(cmake_resources/UniversalSettings.cmake)

string(TOLOWER ${CMAKE_SYSTEM_NAME} SYSTEM_NAME)
string(TOLOWER ${CMAKE_BUILD_TYPE} BUILD_TYPE)
set(ARCH_DESCRIPTION "${SYSTEM_NAME}-${CMAKE_HOST_SYSTEM_PROCESSOR}-${BUILD_TYPE}")
set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_SOURCE_DIR}/arch/${ARCH_DESCRIPTION}/")

function(add_install_executable EXECUTABLE_NAME)
  install(TARGETS ${EXECUTABLE_NAME} RUNTIME DESTINATION .)
  set_target_properties(${EXECUTABLE_NAME} PROPERTIES OUTPUT_NAME "${EXECUTABLE_NAME}.${ARCH_DESCRIPTION}") 
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