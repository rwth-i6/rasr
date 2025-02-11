set(RasrLabelScorer_SOURCES
    BufferedLabelScorer.cc
    CombineLabelScorer.cc
    CTCPrefixLabelScorer.cc
    EncoderFactory.cc
    NoOpLabelScorer.cc
    LegacyFeatureScorerLabelScorer.cc
    Encoder.cc
    EncoderDecoderLabelScorer.cc
    LabelScorer.cc
    LabelScorerFactory.cc
    ScaledLabelScorer.cc
    ScoringContext.cc
    SharedDataHolder.cc)
if(${MODULE_ONNX})
  list(APPEND RasrLabelScorer_SOURCES LimitedCtxOnnxLabelScorer.cc
       NoCtxOnnxLabelScorer.cc StatefulOnnxLabelScorer.cc OnnxEncoder.cc)
endif()

add_library(RasrLabelScorer STATIC ${RasrLabelScorer_SOURCES})

target_link_libraries(
  RasrLabelScorer
  PRIVATE RasrNn
          RasrCore
          RasrMath
          RasrMathLapack
          RasrFlow
          RasrSignal
          RasrSpeech)

if(${MODULE_ONNX})
  target_link_libraries(RasrLabelScorer INTERFACE RasrOnnx)
endif()
