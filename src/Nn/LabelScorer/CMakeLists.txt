set(RasrLabelScorer_SOURCES
            Encoder.cc
            EncoderDecoderLabelScorer.cc
            LabelScorer.cc
            ScoringContext.cc)
if (${MODULE_ONNX})
            list(APPEND RasrLabelScorer_SOURCES
            LimitedCtxOnnxLabelScorer.cc
            StatefulOnnxLabelScorer.cc
            OnnxEncoder.cc)
endif()

add_library(RasrLabelScorer STATIC ${RasrLabelScorer_SOURCES})

target_link_libraries(RasrLabelScorer PRIVATE RasrNn RasrCore RasrMath RasrMathLapack RasrFlow RasrSignal RasrSpeech)

if (${MODULE_ONNX})
            target_link_libraries(RasrLabelScorer INTERFACE RasrOnnx)
endif()
