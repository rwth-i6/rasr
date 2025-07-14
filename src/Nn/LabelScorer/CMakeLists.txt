set(RasrLabelScorer_SOURCES
    BufferedLabelScorer.cc
    CombineLabelScorer.cc
    CTCPrefixLabelScorer.cc
    DataView.cc
    EncoderFactory.cc
    FixedContextOnnxLabelScorer.cc
    NoOpLabelScorer.cc
    NoContextOnnxLabelScorer.cc
    LegacyFeatureScorerLabelScorer.cc
    Encoder.cc
    EncoderDecoderLabelScorer.cc
    LabelScorer.cc
    LabelScorerFactory.cc
    ScoringContext.cc
    StatefulOnnxLabelScorer.cc)

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
