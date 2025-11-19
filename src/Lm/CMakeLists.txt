set(Lm_SOURCES
    AbstractNNLanguageModel.cc
    BackingOff.cc
    ClassLm.cc
    CombineLm.cc
    Compose.cc
    CompressedVector.cc
    CorpusStatistics.cc
    FixedQuantizationCompressedVectorFactory.cc
    IndexMap.cc
    LanguageModel.cc
    Module.cc
    NNHistoryManager.cc
    QuantizedCompressedVectorFactory.cc
    RecurrentLanguageModel.cc
    ReducedPrecisionCompressedVectorFactory.cc
    ReverseArpaLm.cc
    ScaledLanguageModel.cc
    WordlistInterface.cc
)

if(${MODULE_LM_ARPA})
    list(APPEND Lm_SOURCES ArpaLm.cc)
endif()

if(${MODULE_LM_FSA})
    list(APPEND Lm_SOURCES FsaLm.cc CheatingSegmentLm.cc)
endif()

if(${MODULE_LM_ZEROGRAM})
    list(APPEND Lm_SOURCES Zerogram.cc)
endif()

if(${MODULE_LM_TFRNN})
    list(
        APPEND
        Lm_SOURCES
        TFBlasNceSoftmaxAdapter.cc
        TFLstmStateManager.cc
        TFNceSoftmaxAdapter.cc
        TFPassthroughSoftmaxAdapter.cc
        TFQuantizedBlasNceSoftmaxAdapter.cc
        TFRecurrentLanguageModel.cc
        TFTransformerStateManager.cc
    )
endif()

if(${MODULE_LM_ONNX})
    list(
        APPEND
        Lm_SOURCES
        OnnxStatelessLanguageModel.cc
        OnnxLstmStateManager.cc
        OnnxPassthroughSoftmaxAdapter.cc
        OnnxNceSoftmaxAdapter.cc
        OnnxRecurrentLanguageModel.cc
    )
endif()

add_library(RasrLm STATIC ${Lm_SOURCES})

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

if(${MODULE_LM_TFRNN})
    target_link_libraries(RasrLm PUBLIC RasrTensorflow)
endif()

if(${MODULE_LM_ONNX})
    target_link_libraries(RasrLm INTERFACE RasrOnnx)
    add_onnx_dependencies(RasrLm)
endif()

# check.cc code is broken add_executable(check-lm check.cc) target_link_libraries(check-lm PRIVATE RasrLm)
