add_library(
    RasrLexiconfreeLabelsyncBeamSearch STATIC LexiconfreeLabelsyncBeamSearch.cc
)

target_link_libraries(
    RasrLexiconfreeLabelsyncBeamSearch PUBLIC RasrBliss RasrCore RasrNn
                                              RasrSearch
)
