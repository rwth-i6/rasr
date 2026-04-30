add_library(
    RasrNnLabelScorer STATIC
    BufferedLabelScorer.cc
    CombineLabelScorer.cc
    DataView.cc
    Encoder.cc
    EncoderDecoderLabelScorer.cc
    EncoderFactory.cc
    LabelScorer.cc
    LabelScorerFactory.cc
    NoOpLabelScorer.cc
    PriorLabelScorer.cc
    ScaledLabelScorer.cc
    ScoreAccessor.cc
    ScoringContext.cc
    TransitionLabelScorer.cc
    TransitionTypes.cc
)

if(${MODULE_ONNX})
    target_sources(
        RasrNnLabelScorer
        PRIVATE FixedContextOnnxLabelScorer.cc NoContextOnnxLabelScorer.cc
                StatefulOnnxLabelScorer.cc
    )
endif()

target_link_libraries(
    RasrNnLabelScorer
    PUBLIC RasrCore
           RasrFlow
           RasrMath
           RasrMm
           RasrNn
           RasrSearch
           RasrSpeech
)

if(${MODULE_PYTHON})
    find_package(pybind11 REQUIRED)
    add_python_dependencies(RasrNnLabelScorer)
    target_include_directories(RasrNnLabelScorer PUBLIC pybind11_INCLUDE_DIRS)
endif()

if(${MODULE_ONNX})
    target_link_libraries(RasrNnLabelScorer PUBLIC RasrOnnx)
endif()
