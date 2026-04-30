if(${MODULE_SEARCH_WFST})
    add_subdirectory(Wfst)
endif()
if(${MODULE_ADVANCED_TREE_SEARCH})
    add_subdirectory(AdvancedTreeSearch)
endif()
add_subdirectory(LexiconfreeLabelsyncBeamSearch)
add_subdirectory(LexiconfreeTimesyncBeamSearch)
add_subdirectory(TreeTimesyncBeamSearch)

add_library(
    RasrSearch STATIC
    Aligner.cc
    LanguageModelLookahead.cc
    LatticeHandler.cc
    Module.cc
    PersistentStateTree.cc
    Search.cc
    StateTree.cc
    StateTreeIo.cc
    Traceback.cc
    TreeBuilder.cc
    TreeStructure.cc
)

if(${MODULE_SEARCH_MBR})
    target_sources(
        RasrSearch
        PRIVATE MinimumBayesRiskSearch.cc MinimumBayesRiskAStarSearch.cc
                MinimumBayesRiskNBestListSearch.cc
                MinimumBayesRiskSearchUtil.cc
    )
endif()
if(${MODULE_SEARCH_LINEAR})
    target_sources(RasrSearch PRIVATE LinearSearch.cc)
endif()

target_link_libraries(
    RasrSearch
    PUBLIC RasrAm
           RasrBliss
           RasrCore
           RasrFsa
           RasrLattice
           RasrLexiconfreeLabelsyncBeamSearch
           RasrLexiconfreeTimesyncBeamSearch
           RasrLm
           RasrMm
           RasrNn
           RasrSignal
           RasrSpeech
           RasrTreeTimesyncBeamSearch
)

if(${MODULE_ADVANCED_TREE_SEARCH})
    target_link_libraries(RasrSearch PUBLIC RasrAdvancedTreeSearch)
endif()

if(${MODULE_SEARCH_WFST})
    target_link_libraries(RasrSearch PUBLIC RasrOpenFst RasrSearchWfst)
endif()

add_executable(check-search check.cc)

target_link_libraries(check-search PRIVATE RasrSearch)
