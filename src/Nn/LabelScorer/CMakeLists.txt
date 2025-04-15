set(RasrLabelScorer_SOURCES
    BufferedLabelScorer.cc
    CombineLabelScorer.cc
    CTCPrefixLabelScorer.cc
    DataView.cc
    EncoderFactory.cc
    NoOpLabelScorer.cc
    NoCtxOnnxLabelScorer.cc
    LegacyFeatureScorerLabelScorer.cc
    Encoder.cc
    EncoderDecoderLabelScorer.cc
    LabelScorer.cc
    LabelScorerFactory.cc
    ScoringContext.cc)

if(${MODULE_ONNX})
  list(APPEND RasrLabelScorer_SOURCES LimitedCtxOnnxLabelScorer.cc
       NoCtxOnnxLabelScorer.cc StatefulOnnxLabelScorer.cc)
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

if(${MODULE_PYTHON})
  add_python_dependencies(RasrLabelScorer)
endif()
