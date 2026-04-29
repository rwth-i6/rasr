set(RasrPython_SOURCES
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
target_link_libraries(
    RasrPython PUBLIC RasrBliss RasrCore RasrMath RasrNn RasrSearch RasrSpeech
)
add_python_dependencies(RasrPython)

add_executable(check-python check.cc)
target_link_libraries(check-python PRIVATE RasrPython RasrCore)
