add_library(
    RasrFsa STATIC
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
    Project.cc
    Properties.cc
    Prune.cc
    Random.cc
    Rational.cc
    RemoveEpsilons.cc
    Resources.cc
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

target_link_libraries(RasrFsa PUBLIC RasrCore)

# check.cc code is broken add_executable(check-fsa check.cc)
# target_link_libraries(check-fsa PRIVATE RasrFsa)
