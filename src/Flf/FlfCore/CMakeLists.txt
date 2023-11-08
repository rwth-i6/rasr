set(FlfCore_SOURCES
        Ftl.cc
        Basic.cc
        Boundaries.cc
        Lattice.cc
        LatticeInternal.cc
        Semiring.cc
        TopologicalOrderQueue.cc
        Traverse.cc
        Types.cc
        Utility.cc
        Weight.cc
)

add_library(RasrFlfCore STATIC ${FlfCore_SOURCES})