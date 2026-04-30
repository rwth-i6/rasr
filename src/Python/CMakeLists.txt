add_library(
    RasrPython STATIC
    AllophoneStateFsaBuilder.cc
    Configuration.cc
    Init.cc
    LabelScorer.cc
    Numpy.cc
    ScoringContext.cc
    Search.cc
    Utilities.cc
)

target_link_libraries(
    RasrPython
    PUBLIC RasrBliss
           RasrCore
           RasrMath
           RasrNn
           RasrPythonDependencies
           RasrSearch
           RasrSpeech
)

add_rasr_check_executable(check-python check.cc)

target_link_rasr_check_libraries(check-python PRIVATE RasrPython RasrCore)
