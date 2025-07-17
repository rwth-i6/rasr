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

target_link_libraries(RasrFlfCore PUBLIC RasrBliss RasrCore RasrFlf RasrFsa RasrLattice RasrSpeech)

add_executable(check-flf-core check.cc)
target_link_libraries(check-flf-core PRIVATE RasrFlfCore)
