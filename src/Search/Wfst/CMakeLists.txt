target_sources(
    RasrSearch
    PRIVATE BookKeeping.cc
            Builder.cc
            ComposedNetwork.cc
            CompressedNetwork.cc
            ContextTransducerBuilder.cc
            CreateOperations.cc
            DynamicLmFst.cc
            ExpandingFsaSearch.cc
            FstOperations.cc
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
            Network.cc
            NonWordTokens.cc
            SearchSpace.cc
            StateSequence.cc
            StateTree.cc
            Traceback.cc
            UtilityOperations.cc
            WordEnd.cc
)

target_compile_options(RasrSearch PRIVATE "-Wno-sign-compare")

target_link_libraries(RasrSearch PUBLIC RasrFlf RasrOpenFst)

add_rasr_check_executable(check-search-wfst check.cc)

target_link_rasr_check_libraries(check-search-wfst PRIVATE RasrSearch)

add_rasr_check_executable(check-search-wfst-ctrans check_ctrans.cc)

target_link_rasr_check_libraries(check-search-wfst-ctrans PRIVATE RasrSearch)
