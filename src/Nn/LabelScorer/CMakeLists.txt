set(RasrLabelScorer_SOURCES
    BufferedLabelScorer.cc
    CombineLabelScorer.cc
    CTCPrefixLabelScorer.cc
    DataView.cc
    EncoderFactory.cc
    NoOpLabelScorer.cc
    LegacyFeatureScorerLabelScorer.cc
    Encoder.cc
    EncoderDecoderLabelScorer.cc
    LabelScorer.cc
    LabelScorerFactory.cc
    ScoringContext.cc
)

if(${MODULE_ONNX})
    list(APPEND RasrLabelScorer_SOURCES FixedContextOnnxLabelScorer.cc NoContextOnnxLabelScorer.cc
         StatefulOnnxLabelScorer.cc
    )
endif()

add_library(RasrLabelScorer STATIC ${RasrLabelScorer_SOURCES})

target_link_libraries(
    RasrLabelScorer
    PUBLIC RasrAm
           RasrCore
           RasrFlow
           RasrMath
           RasrMm
           RasrNn
           RasrSearch
           RasrSpeech
)

if(${MODULE_ONNX})
    target_link_libraries(RasrLabelScorer PUBLIC RasrOnnx)
endif()

if(${MODULE_PYTHON})
    add_python_dependencies(RasrLabelScorer)
endif()
