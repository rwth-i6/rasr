set(CMAKE_CXX_STANDARD 20)
include(CheckCXXCompilerFlag)

add_compile_definitions(_GNU_SOURCE)
add_compile_options(
    -fPIC
    -pipe
    -funsigned-char
    -Wno-unknown-pragmas
    -Wall
    -Wno-long-long
    -Wno-deprecated
    -fno-strict-aliasing
    "$<$<COMPILE_LANGUAGE:CXX>:-Werror=format-security>"
    "$<$<COMPILE_LANGUAGE:CXX>:-Werror=reorder>"
    "$<$<COMPILE_LANGUAGE:CXX>:-Werror=return-type>"
    "$<$<COMPILE_LANGUAGE:CXX>:-Werror=delete-non-virtual-dtor>"
    "$<$<COMPILE_LANGUAGE:CXX>:-Werror=unused-but-set-variable>"
)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|i[3-6]86)$")
    check_cxx_compiler_flag("-msse3" COMPILER_SUPPORTS_MSSE3)
    if(COMPILER_SUPPORTS_MSSE3)
        add_compile_options(-msse3)
    endif()
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    message(STATUS "Using GCC compiler")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    message(STATUS "Using Clang compiler")
    add_compile_definitions(_GLIBCXX_PERMIT_BACKWARD_HASH)
    add_compile_options(-D__float128=void
    )# hack: http://llvm.org/bugs/show_bug.cgi?id=13530
else()
    message(WARNING "Compiler might not be supported. Use at your own risk")
endif()

if(NOT DEFINED MARCH)
    set(MARCH
        "native"
        CACHE STRING "Default target architecture"
    )
endif()
check_cxx_compiler_flag("-march=${MARCH}" COMPILER_SUPPORTS_MARCH)
if(COMPILER_SUPPORTS_MARCH)
    add_compile_options("-march=${MARCH}")
    message(STATUS "Using -march=${MARCH}")
else()
    message(STATUS "Architecture -march=${MARCH} is not supported")
endif()

find_package(LibXml2 REQUIRED)
find_package(Threads REQUIRED)
find_package(ZLIB REQUIRED)
find_package(LAPACK REQUIRED)

find_library(LIB_RT rt REQUIRED)

add_library(RasrSystemDependencies INTERFACE)
target_link_libraries(
    RasrSystemDependencies INTERFACE LibXml2::LibXml2 ZLIB::ZLIB
                                     Threads::Threads ${LIB_RT}
)

add_library(RasrLapackDependencies INTERFACE)
target_link_libraries(RasrLapackDependencies INTERFACE LAPACK::LAPACK)

add_library(RasrAudioDependencies INTERFACE)

include_directories(${LIBXML2_INCLUDE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/src)

if(${MODULE_AUDIO_FFMPEG})
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(AVFORMAT REQUIRED IMPORTED_TARGET libavformat)
    pkg_check_modules(SWRESAMPLE REQUIRED IMPORTED_TARGET libswresample)
    pkg_check_modules(AVCODEC REQUIRED IMPORTED_TARGET libavcodec)
    pkg_check_modules(AVUTIL REQUIRED IMPORTED_TARGET libavutil)

    target_include_directories(
        RasrAudioDependencies
        INTERFACE ${AVFORMAT_INCLUDE_DIRS} ${SWRESAMPLE_INCLUDE_DIRS}
                  ${AVCODEC_INCLUDE_DIRS} ${AVUTIL_INCLUDE_DIRS}
    )
    target_link_libraries(
        RasrAudioDependencies
        INTERFACE PkgConfig::AVFORMAT PkgConfig::SWRESAMPLE PkgConfig::AVCODEC
                  PkgConfig::AVUTIL
    )
endif()

if(${MODULE_OPENMP})
    find_package(OpenMP REQUIRED)
    target_link_libraries(RasrSystemDependencies INTERFACE OpenMP::OpenMP_CXX)
endif()

if(${MODULE_CUDA})
    set(CMAKE_CUDA_ARCHITECTURES native)
    enable_language(CUDA)
endif()

if(${MODULE_AUDIO_FLAC})
    target_link_libraries(RasrAudioDependencies INTERFACE FLAC)
endif()

if(${MODULE_AUDIO_WAV_SYSTEM})
    target_link_libraries(RasrAudioDependencies INTERFACE sndfile)
endif()
