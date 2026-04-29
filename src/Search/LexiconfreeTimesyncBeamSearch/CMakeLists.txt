set(RasrLexiconfreeTimesyncBeamSearch_SOURCES LexiconfreeTimesyncBeamSearch.cc)

add_library(
    RasrLexiconfreeTimesyncBeamSearch STATIC
    ${RasrLexiconfreeTimesyncBeamSearch_SOURCES}
)
target_link_libraries(
    RasrLexiconfreeTimesyncBeamSearch PUBLIC RasrBliss RasrCore RasrLattice
                                             RasrNn RasrSearch
)
