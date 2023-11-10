set(Fsa_SOURCES
        Accessible.cc
        Alphabet.cc
        AlphabetUtility.cc
        AlphabetXml.cc
        Archive.cc
        Arithmetic.cc
        Basic.cc
        Best.cc
        Cache.cc
        Compose.cc
        Determinize.cc
        Input.cc
        Levenshtein.cc
        Linear.cc
        Minimize.cc
        Output.cc
        Packed.cc
        Permute.cc
        Properties.cc
        Project.cc
        Prune.cc
        Random.cc
        Resources.cc
        Rational.cc
        RemoveEpsilons.cc
        Semiring.cc
        Semiring64.cc
        Sort.cc
        Sssp.cc
        Sssp4SpecialSymbols.cc
        Static.cc
        Storage.cc
        Types.cc
        Utility.cc
)

add_library(RasrFsa STATIC ${Fsa_SOURCES})

target_link_libraries(RasrFsa INTERFACE RasrCore)

if(${MODULE_PYTHON})
    find_package(SWIG REQUIRED)
    include(${SWIG_USE_FILE})
    set_source_files_properties(fsa.i PROPERTIES CPLUSPLUS ON)
    swig_add_library(_fsa TYPE MODULE LANGUAGE python OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR} SOURCES fsa.i )
    target_include_directories(_fsa PRIVATE ${Python3_INCLUDE_DIRS})
    target_link_libraries(_fsa INTERFACE ${Python3_LIBRARIES})
endif()