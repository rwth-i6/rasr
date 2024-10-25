set(RasrLabelScorer_SOURCES
            Decoder.cc
            Encoder.cc
            EncoderDecoderLabelScorer.cc
            LabelHistory.cc
            LabelScorer.cc)
if (${MODULE_ONNX})
            list(APPEND RasrLabelScorer_SOURCES
            LimitCtxTimesyncOnnxDecoder.cc
            StatefulFullEncOnnxDecoder.cc
            OnnxEncoder.cc)
endif()

add_library(RasrLabelScorer STATIC ${RasrLabelScorer_SOURCES})

target_link_libraries(RasrLabelScorer PRIVATE RasrNn RasrCore RasrMath RasrMathLapack RasrFlow RasrSignal RasrSpeech)

if (${MODULE_ONNX})
            target_link_libraries(RasrLabelScorer INTERFACE RasrOnnx)
endif()
