set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "standard" "debug"
                                             "release")

if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "Setting build type to 'standard' as none was specified.")
  set(CMAKE_BUILD_TYPE
      standard
      CACHE STRING "Choose the type of build." FORCE)
endif()

set(CMAKE_C_FLAGS_STANDARD "-O2")
set(CMAKE_CXX_FLAGS_STANDARD "-O2")
set(CMAKE_EXE_LINKER_FLAGS_STANDARD "")
set(CMAKE_DEF_FLAGS_STANDARD "-DSPRINT_STANDARD_BUILD")

set(CMAKE_C_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_EXE_LINKER_FLAGS_RELEASE "-rdynamic")
set(CMAKE_DEF_FLAGS_RELEASE "-DSPRINT_RELEASE_BUILD -DNDEBUG")

set(CMAKE_C_FLAGS_DEBUG "-O0 -g")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")
set(CMAKE_EXE_LINKER_FLAGS_DEBUG "")
set(CMAKE_DEF_FLAGS_DEBUG "-D_GLIBCXX_DEBUG -DDEBUG")
