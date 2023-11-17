set(Lattice_SOURCES
        Archive.cc
        Basic.cc
        Lattice.cc
        LatticeAdaptor.cc
        Morphism.cc
        Utilities.cc
)

if(${MODULE_LATTICE_BASIC})
    list(APPEND Lattice_SOURCES Cache.cc Static.cc Compose.cc)
endif()

if(${MODULE_LATTICE_DT})
    list(APPEND Lattice_SOURCES
            Accuracy.cc
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
    list(APPEND Lattice_SOURCES HtkReader.cc HtkWriter.cc)
endif()

add_library(RasrLattice STATIC ${Lattice_SOURCES})

target_link_libraries(RasrLattice INTERFACE
        RasrSpeech
        RasrLm
        RasrAm
        RasrMm
        RasrMc
        RasrBliss
        RasrFsa
        RasrAudio
        RasrSignal
        RasrFlow
        RasrMath
        RasrMathLapack
        RasrCore
        RasrCart
)

if(${MODULE_MATH_NR})
        target_link_libraries(RasrLattice INTERFACE RasrMathNr)
endif()