if (${MODULE_SEARCH_WFST})
    add_subdirectory(Wfst)
endif ()
if (${MODULE_ADVANCED_TREE_SEARCH})
    add_subdirectory(AdvancedTreeSearch)
endif ()
if (${MODULE_GENERIC_SEQ2SEQ_TREE_SEARCH})
    add_subdirectory(GenericSeq2SeqTreeSearch)
endif ()

set(RasrSearch_SOURCES
        Aligner.cc
        LatticeHandler.cc
        LanguageModelLookahead.cc
        Module.cc
        Search.cc
        StateTree.cc
        StateTreeIo.cc
        WordConditionedTreeSearch.cc
)

if (${MODULE_SEARCH_MBR})
    list(APPEND RasrSearch_SOURCES
            MinimumBayesRiskSearch.cc
            MinimumBayesRiskAStarSearch.cc
            MinimumBayesRiskNBestListSearch.cc
            MinimumBayesRiskSearchUtil.cc
    )
endif ()
if (${MODULE_SEARCH_LINEAR})
    list(APPEND RasrSearch_SOURCES LinearSearch.cc)
endif ()

add_library(RasrSearch STATIC ${RasrSearch_SOURCES})

target_link_libraries(RasrSearch INTERFACE
        RasrBliss
        RasrFsa
        RasrCore
)