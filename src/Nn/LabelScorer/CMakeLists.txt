target_sources(
    RasrNn
    PRIVATE BufferedLabelScorer.cc
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
        RasrNn PRIVATE FixedContextOnnxLabelScorer.cc
                       NoContextOnnxLabelScorer.cc StatefulOnnxLabelScorer.cc
    )
endif()

if(${MODULE_PYTHON})
    find_package(pybind11 REQUIRED)
    target_link_libraries(
        RasrNn PUBLIC RasrPythonDependencies pybind11::headers
    )
endif()

if(${MODULE_ONNX})
    target_link_libraries(RasrNn PUBLIC RasrOnnx)
endif()
