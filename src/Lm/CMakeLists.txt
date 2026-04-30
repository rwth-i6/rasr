add_library(
    RasrLm STATIC
    AbstractNNLanguageModel.cc
    BackingOff.cc
    ClassLm.cc
    CombineLm.cc
    Compose.cc
    CorpusStatistics.cc
    IndexMap.cc
    LanguageModel.cc
    Module.cc
    NNHistoryManager.cc
    RecurrentLanguageModel.cc
    ReverseArpaLm.cc
    ScaledLanguageModel.cc
    WordlistInterface.cc
)

if(${MODULE_LM_ARPA})
    target_sources(RasrLm PRIVATE ArpaLm.cc)
endif()

if(${MODULE_LM_FSA})
    target_sources(RasrLm PRIVATE FsaLm.cc CheatingSegmentLm.cc)
endif()

if(${MODULE_LM_ZEROGRAM})
    target_sources(RasrLm PRIVATE Zerogram.cc)
endif()

if(${MODULE_TENSORFLOW} AND ${MODULE_LM_TFRNN})
    target_sources(
        RasrLm
        PRIVATE TFBlasNceSoftmaxAdapter.cc TFNceSoftmaxAdapter.cc
                TFPassthroughSoftmaxAdapter.cc
                TFQuantizedBlasNceSoftmaxAdapter.cc TFRecurrentLanguageModel.cc
    )
endif()

if(${MODULE_ONNX} AND ${MODULE_LM_ONNX})
    target_sources(
        RasrLm
        PRIVATE OnnxNceSoftmaxAdapter.cc OnnxPassthroughSoftmaxAdapter.cc
                OnnxRecurrentLanguageModel.cc OnnxStatelessLanguageModel.cc
    )
endif()

target_link_libraries(
    RasrLm
    PUBLIC RasrBliss
           RasrCore
           RasrFlf
           RasrFlow
           RasrFsa
           RasrMath
           RasrMc
           RasrMm
           RasrNn
           RasrSearch
           RasrSignal
           RasrSpeech
)
if(${MODULE_TENSORFLOW} AND ${MODULE_LM_TFRNN})
    target_link_libraries(RasrLm PUBLIC RasrTensorflow)
endif()
if(${MODULE_ONNX})
    target_link_libraries(RasrLm PUBLIC RasrOnnx)
endif()

# check.cc code is broken add_executable(check-lm check.cc)
# target_link_libraries(check-lm PRIVATE RasrLm)
