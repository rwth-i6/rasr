set(RasrPython_SOURCES
    Align.cc
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

target_link_libraries(
    RasrPython
    PUBLIC RasrBliss
           RasrCore
           RasrFlow
           RasrLm
           RasrMath
           RasrMm
           RasrNn
           RasrSearch
           RasrSpeech
)
