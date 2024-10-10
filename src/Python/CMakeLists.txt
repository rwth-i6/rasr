set(RasrPython_SOURCES
        AllophoneStateFsaBuilder.cc
        Configuration.cc
        Init.cc
        Numpy.cc
        Utilities.cc
)

add_library(RasrPython STATIC ${RasrPython_SOURCES})
add_python_dependencies(RasrPython)
target_compile_options(RasrPython PRIVATE "-fexceptions")
