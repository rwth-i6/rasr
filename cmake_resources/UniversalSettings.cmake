set(CMAKE_CXX_STANDARD 17)

add_compile_definitions(CMAKE_DISABLE_MODULES_HH)

add_compile_definitions(_GNU_SOURCE)
add_compile_options(
        -pipe
        -funsigned-char
        -fno-exceptions
        -Wno-unknown-pragmas
        -Wall
        -Wno-long-long
        -Wno-deprecated
        -fno-strict-aliasing
        -ffast-math
        -msse3
)

if (NOT DEFINED MARCH)
    set(MARCH "native" CACHE STRING "Default target architecture")
endif ()
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-march=${MARCH}" COMPILER_SUPPORTS_MARCH)
if (COMPILER_SUPPORTS_MARCH)
    add_compile_options("-march=${MARCH}")
    message(STATUS "Using -march=${MARCH}")
else ()
    message(STATUS "Architecture -march=${MARCH} is not supported")
endif ()

find_package(LibXml2 REQUIRED)
find_package(Threads REQUIRED)
find_package(ZLIB REQUIRED)
find_package(LAPACK REQUIRED)

find_library(LIB_RT rt REQUIRED)

link_libraries(LibXml2::LibXml2 ZLIB::ZLIB Threads::Threads LAPACK::LAPACK ${LIB_RT})

include_directories(${LIBXML2_INCLUDE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/src)

if (${MODULE_AUDIO_FFMPEG})
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(AVFORMAT REQUIRED IMPORTED_TARGET libavformat)
    pkg_check_modules(SWRESAMPLE REQUIRED IMPORTED_TARGET libswresample)
    pkg_check_modules(AVCODEC REQUIRED IMPORTED_TARGET libavcodec)
    pkg_check_modules(AVUTIL REQUIRED IMPORTED_TARGET libavutil)

    include_directories(${AVFORMAT_INCLUDE_DIRS} ${SWRESAMPLE_INCLUDE_DIRS} ${AVCODEC_INCLUDE_DIRS} ${AVUTIL_INCLUDE_DIRS})
    link_libraries(PkgConfig::AVFORMAT PkgConfig::SWRESAMPLE PkgConfig::AVCODEC PkgConfig::AVUTIL)
endif ()

if (${MODULE_OPENMP})
    find_package(OpenMP REQUIRED)
    add_compile_options(OpenMP_CXX_FLAGS)
    include_directories(OpenMP_CXX_INCLUDE_DIRS)
    link_libraries(OpenMP::OpenMP_CXX)
endif ()

if (${MODULE_TENSORFLOW})
    include(cmake_resources/Tensorflow.cmake)
endif ()

if(${MODULE_CUDA})
    enable_language(CUDA)
    set(CMAKE_CUDA_ARCHITECTURES 61;75;86)
endif()

if (${MODULE_AUDIO_FLAC})
    link_libraries(FLAC)
endif ()

if (${MODULE_AUDIO_WAV_SYSTEM})
    link_libraries(sndfile)
endif ()

if (${MODULE_PYTHON})
    include(cmake_resources/Python.cmake)
endif ()