set(RasrTreeTimesyncBeamSearch_SOURCES TreeTimesyncBeamSearch.cc)

add_library(
    RasrTreeTimesyncBeamSearch STATIC ${RasrTreeTimesyncBeamSearch_SOURCES}
)
target_link_libraries(
    RasrTreeTimesyncBeamSearch PUBLIC RasrAm RasrBliss RasrCore RasrLattice
                                      RasrNn RasrSearch
)
