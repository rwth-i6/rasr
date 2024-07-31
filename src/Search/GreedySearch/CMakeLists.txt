set(RasrGreedySearch_SOURCES
        GreedySearch.cc
)

add_library(RasrGreedySearch STATIC ${RasrGreedySearch_SOURCES})
target_link_libraries(RasrGreedySearch INTERFACE
        RasrSearch
        RasrBliss
        RasrFsa
        RasrCore
)
