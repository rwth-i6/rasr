add_library(
    RasrFlfCore STATIC
    Basic.cc
    Boundaries.cc
    Ftl.cc
    Lattice.cc
    LatticeInternal.cc
    Semiring.cc
    TopologicalOrderQueue.cc
    Traverse.cc
    Types.cc
    Utility.cc
    Weight.cc
)

target_link_libraries(
    RasrFlfCore PUBLIC RasrBliss RasrCore RasrFsa RasrLattice RasrSpeech
)

add_rasr_check_executable(check-flf-core check.cc)

target_link_rasr_check_libraries(check-flf-core PRIVATE RasrFlfCore)
