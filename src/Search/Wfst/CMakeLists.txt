set(RasrSearchWfst_SOURCES
    BookKeeping.cc
    Builder.cc
    ComposedNetwork.cc
    CompressedNetwork.cc
    ContextTransducerBuilder.cc
    CreateOperations.cc
    DynamicLmFst.cc
    ExpandingFsaSearch.cc
    GrammarFst.cc
    IoOperations.cc
    Lattice.cc
    LatticeAdaptor.cc
    LatticeArchive.cc
    LatticeGenerator.cc
    LatticeHandler.cc
    LatticeNetwork.cc
    LatticeReader.cc
    LexiconBuilder.cc
    LexiconFst.cc
    Module.cc
    NonWordTokens.cc
    Network.cc
    SearchSpace.cc
    StateSequence.cc
    StateTree.cc
    Traceback.cc
    FstOperations.cc
    UtilityOperations.cc
    WordEnd.cc
)

add_library(RasrSearchWfst STATIC ${RasrSearchWfst_SOURCES})

target_compile_options(RasrSearchWfst PRIVATE "-Wno-sign-compare")

target_link_libraries(
    RasrSearchWfst
    PUBLIC RasrAm
           RasrBliss
           RasrCore
           RasrFlf
           RasrFsa
           RasrLattice
           RasrLm
           RasrOpenFst
           RasrSearch
           RasrSpeech
)

add_executable(check-search-wfst check.cc)
target_link_libraries(check-search-wfst PRIVATE RasrSearchWfst)

add_executable(check-search-wfst-ctrans check_ctrans.cc)
target_link_libraries(check-search-wfst-ctrans PRIVATE RasrSearchWfst)
