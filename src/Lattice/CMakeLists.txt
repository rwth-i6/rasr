add_library(
    RasrLattice STATIC Archive.cc Basic.cc Lattice.cc LatticeAdaptor.cc
                       Morphism.cc Utilities.cc
)

if(${MODULE_LATTICE_BASIC})
    target_sources(RasrLattice PRIVATE Cache.cc Static.cc Compose.cc)
endif()

if(${MODULE_LATTICE_DT})
    target_sources(
        RasrLattice
        PRIVATE Accuracy.cc
                Arithmetic.cc
                Best.cc
                Merge.cc
                Posterior.cc
                Rational.cc
                RemoveEpsilons.cc
                SmoothedAccuracy.cc
                SmoothingFunction.cc
                TimeframeError.cc
    )
endif()

if(${MODULE_LATTICE_HTK})
    target_sources(RasrLattice PRIVATE HtkReader.cc HtkWriter.cc)
endif()

target_link_libraries(
    RasrLattice
    PUBLIC RasrAm
           RasrBliss
           RasrCore
           RasrFsa
           RasrLm
           RasrMm
           RasrSearch
           RasrSpeech
)

# check.cc code is broken add_executable(check-lattice check.cc)
# target_link_libraries(check-lattice PRIVATE RasrLattice)
