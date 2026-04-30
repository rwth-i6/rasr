add_library(
    RasrLexiconfreeTimesyncBeamSearch STATIC LexiconfreeTimesyncBeamSearch.cc
)

target_link_libraries(
    RasrLexiconfreeTimesyncBeamSearch PUBLIC RasrBliss RasrCore RasrLattice
                                             RasrNn RasrSearch
)
