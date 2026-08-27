target_sources(
    RasrSearch
    PRIVATE AcousticLookAhead.cc
            AdvancedTreeSearch.cc
            DynamicBeamPruningStrategy.cc
            Helpers.cc
            PathRecombination.cc
            PathRecombinationApproximation.cc
            PrefixFilter.cc
            ScoreDependentStatistics.cc
            SearchSpace.cc
            SearchSpaceHelpers.cc
            SearchSpaceStatistics.cc
            SimpleThreadPool.cc
            Trace.cc
)

target_compile_options(
    RasrSearch
    PRIVATE -Wno-sign-compare -Winline --param=max-inline-insns-auto=10000
            --param=max-inline-insns-single=10000
            --param=large-function-growth=25000 --param=inline-unit-growth=400
)
