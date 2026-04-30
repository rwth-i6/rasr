add_subdirectory(FlfCore)

if(${MODULE_FLF_EXT})
    add_subdirectory(FlfExt)
endif()

add_library(
    RasrFlf STATIC
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
    Convert.cc
    Copy.cc
    CorpusProcessor.cc
    Determinize.cc
    Draw.cc
    EpsilonRemoval.cc
    Evaluate.cc
    Filter.cc
    FlfIo.cc
    Formattings.cc
    FwdBwd.cc
    GammaCorrection.cc
    HtkSlfIo.cc
    IncrementalRecognizer.cc
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
    Rescale.cc
    Rescore.cc
    RescoreInternal.cc
    RescoreLm.cc
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
    Union.cc
)

target_link_libraries(
    RasrFlf
    PUBLIC RasrAm
           RasrBliss
           RasrCore
           RasrFlfCore
           RasrFlow
           RasrFsa
           RasrLattice
           RasrLm
           RasrMath
           RasrMm
           RasrSearch
           RasrSpeech
)

if(${MODULE_FLF_EXT})
    target_link_libraries(RasrFlf PUBLIC RasrFlfExt)
endif()

add_executable(check-flf check.cc)

target_link_libraries(check-flf PRIVATE RasrFlf)
