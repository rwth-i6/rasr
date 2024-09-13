add_subdirectory(FlfCore)

if (${MODULE_FLF_EXT})
        add_subdirectory(FlfExt)
endif ()

set(Flf_SOURCES
        Archive.cc
        Best.cc
        Cache.cc
        CenterFrameConfusionNetworkBuilder.cc
        Combination.cc
        Compose.cc
        Concatenate.cc
        ConfusionNetwork.cc
        ConfusionNetworkCombination.cc
        ConfusionNetworkIo.cc
        CorpusProcessor.cc
        Copy.cc
        Convert.cc
        Draw.cc
        Determinize.cc
        EpsilonRemoval.cc
        Evaluate.cc
        Filter.cc
        Formattings.cc
        FlfIo.cc
        FwdBwd.cc
        GammaCorrection.cc
        HtkSlfIo.cc
        Info.cc
        Io.cc
        LanguageModel.cc
        LatticeAdaptor.cc
        LatticeHandler.cc
        Lexicon.cc
        LocalCostDecoder.cc
        Map.cc
        Miscellaneous.cc
        Module.cc
        NBest.cc
        Network.cc
        NodeFactory.cc
        NonWordFilter.cc
        PivotArcConfusionNetworkBuilder.cc
        Processor.cc
        Prune.cc
        PushForwardRescoring.cc
        Recognizer.cc
        RecognizerV2.cc
        IncrementalRecognizer.cc
        Rescore.cc
        RescoreLm.cc
        RescoreInternal.cc
        Rescale.cc
        Segment.cc
        SegmentwiseSpeechProcessor.cc
        StateClusterConfusionNetworkBuilder.cc
        TimeAlignment.cc
        TimeframeConfusionNetwork.cc
        TimeframeConfusionNetworkBuilder.cc
        TimeframeConfusionNetworkCombination.cc
        TimeframeConfusionNetworkIo.cc
        TimeframeError.cc
        Traceback.cc
        Union.cc)

add_library(RasrFlf STATIC ${Flf_SOURCES})

target_link_libraries(RasrFlf INTERFACE RasrFlfCore)

if (${MODULE_FLF_EXT})
        target_link_libraries(RasrFlf INTERFACE RasrFlfExt)
endif ()

target_link_libraries(
        RasrFlf
        INTERFACE RasrSpeech
        RasrLattice
        RasrLm
        RasrAm
        RasrMm
        RasrMc
        RasrCart
        RasrBliss
        RasrFlow
        RasrFsa
        RasrCore)

add_executable(check-flf check.cc)
target_link_libraries(check-flf PRIVATE RasrFlf)
