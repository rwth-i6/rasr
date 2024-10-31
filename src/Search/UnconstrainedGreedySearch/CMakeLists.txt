set(RasrUnconstrainedGreedySearch_SOURCES
        UnconstrainedGreedySearch.cc
)

add_library(RasrUnconstrainedGreedySearch STATIC ${RasrUnconstrainedGreedySearch_SOURCES})
target_link_libraries(RasrUnconstrainedGreedySearch INTERFACE
        RasrSearch
        RasrBliss
        RasrFsa
        RasrCore
)
