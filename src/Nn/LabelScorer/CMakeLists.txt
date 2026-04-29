set(RasrNnLabelScorer_SOURCES
    BufferedLabelScorer.cc
    CombineLabelScorer.cc
    DataView.cc
    Encoder.cc
    EncoderDecoderLabelScorer.cc
    EncoderFactory.cc
    FixedContextOnnxLabelScorer.cc
    LabelScorer.cc
    LabelScorerFactory.cc
    NoContextOnnxLabelScorer.cc
    NoOpLabelScorer.cc
    PriorLabelScorer.cc
    ScaledLabelScorer.cc
    ScoreAccessor.cc
    ScoringContext.cc
    StatefulOnnxLabelScorer.cc
    TransitionLabelScorer.cc
    TransitionTypes.cc
)

add_library(RasrNnLabelScorer STATIC ${RasrNnLabelScorer_SOURCES})

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
    add_onnx_dependencies(RasrNnLabelScorer)
endif()
