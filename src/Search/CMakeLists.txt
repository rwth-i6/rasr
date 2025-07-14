if(${MODULE_SEARCH_WFST})
  add_subdirectory(Wfst)
endif()
if(${MODULE_ADVANCED_TREE_SEARCH})
  add_subdirectory(AdvancedTreeSearch)
endif()

add_subdirectory(LexiconfreeTimesyncBeamSearch)
add_subdirectory(LexiconfreeLabelsyncBeamSearch)
add_subdirectory(NonAutoregressiveSearch)
add_subdirectory(TreeTimesyncBeamSearch)

set(RasrSearch_SOURCES
    Aligner.cc
    LatticeHandler.cc
    LanguageModelLookahead.cc
    Module.cc
    PersistentStateTree.cc
    Search.cc
    StateTree.cc
    StateTreeIo.cc
    Traceback.cc
    TreeBuilder.cc
    TreeStructure.cc)

if(${MODULE_SEARCH_MBR})
  list(APPEND RasrSearch_SOURCES MinimumBayesRiskSearch.cc
       MinimumBayesRiskAStarSearch.cc MinimumBayesRiskNBestListSearch.cc
       MinimumBayesRiskSearchUtil.cc)
endif()
if(${MODULE_SEARCH_LINEAR})
  list(APPEND RasrSearch_SOURCES LinearSearch.cc)
endif()

add_library(RasrSearch STATIC ${RasrSearch_SOURCES})

target_link_libraries(RasrSearch INTERFACE RasrBliss RasrFsa RasrCore
                                           RasrLattice)

if(${MODULE_PYTHON})
  add_python_dependencies(RasrSearch)
endif()

add_executable(check-search check.cc)
target_link_libraries(check-search PRIVATE RasrSearch)
