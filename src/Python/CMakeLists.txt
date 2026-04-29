set(
    RasrPython_SOURCES
    AllophoneStateFsaBuilder.cc
    Configuration.cc
    Init.cc
    LabelScorer.cc
    Numpy.cc
    ScoringContext.cc
    Search.cc
    Utilities.cc
)

add_library(RasrPython STATIC ${RasrPython_SOURCES})
add_python_dependencies(RasrPython)
target_compile_options(RasrPython PRIVATE "-fexceptions")

add_executable(check-python check.cc)
target_link_libraries(check-python PRIVATE RasrPython)
add_python_dependencies(check-python)
