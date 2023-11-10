set(RasrPython_SOURCES
        Init.cc
        Numpy.cc
        Utilities.cc
)

add_library(RasrPython STATIC ${RasrPython_SOURCES})

target_link_libraries(RasrPython INTERFACE RasrMath RasrCore)