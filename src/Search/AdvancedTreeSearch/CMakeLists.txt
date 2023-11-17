set(RasrAdvancedTreeSearch_SOURCES
        AcousticLookAhead.cc
        AdvancedTreeSearch.cc
        DynamicBeamPruningStrategy.cc
        Helpers.cc
        LanguageModelLookahead.cc
        PathRecombination.cc
        PathRecombinationApproximation.cc
        PersistentStateTree.cc
        PrefixFilter.cc
        SearchSpace.cc
        SearchSpaceHelpers.cc
        SearchSpaceStatistics.cc
        SimpleThreadPool.cc
        Trace.cc
        TreeBuilder.cc
        TreeStructure.cc
)

add_library(RasrAdvancedTreeSearch STATIC ${RasrAdvancedTreeSearch_SOURCES})
target_link_libraries(RasrAdvancedTreeSearch INTERFACE
        RasrSearch
        RasrBliss
        RasrFsa
        RasrCore
)

target_compile_options(RasrAdvancedTreeSearch PRIVATE
        "-Wno-sign-compare -Winline --param max-inline-insns-auto=10000 --param max-inline-insns-single=10000 --param large-function-growth=25000 --param inline-unit-growth=400"
)