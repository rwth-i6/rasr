add_library(RasrTreeTimesyncBeamSearch STATIC TreeTimesyncBeamSearch.cc)

target_link_libraries(
    RasrTreeTimesyncBeamSearch PUBLIC RasrAm RasrBliss RasrCore RasrLattice
                                      RasrNn RasrSearch
)
