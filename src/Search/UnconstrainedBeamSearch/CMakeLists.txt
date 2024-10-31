set(RasrUnconstrainedBeamSearch_SOURCES
        UnconstrainedBeamSearch.cc
)

add_library(RasrUnconstrainedBeamSearch STATIC ${RasrUnconstrainedBeamSearch_SOURCES})
target_link_libraries(RasrUnconstrainedBeamSearch INTERFACE
        RasrSearch
        RasrBliss
        RasrFsa
        RasrCore
)
