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
        Utility.cc)

add_library(RasrFsa STATIC ${Fsa_SOURCES})

target_link_libraries(RasrFsa INTERFACE RasrCore)

# check.cc code is broken
# add_executable(check-fsa check.cc)
# target_link_libraries(check-fsa PRIVATE RasrFsa)
