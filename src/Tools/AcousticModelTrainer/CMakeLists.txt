add_executable(acoustic-model-trainer AcousticModelTrainer.cc)
add_install_executable(acoustic-model-trainer)
target_link_libraries(
    acoustic-model-trainer
    PRIVATE RasrAm
            RasrAudio
            RasrBliss
            RasrCore
            RasrFlow
            RasrLm
            RasrMath
            RasrMm
            RasrSignal
            RasrSpeech
)

add_executable(allophone-tool AllophoneTool.cc)
add_install_executable(allophone-tool)
target_link_libraries(allophone-tool PRIVATE RasrAm RasrBliss RasrCore)

if(${MODULE_CART})
    target_link_libraries(acoustic-model-trainer PRIVATE RasrCart)
endif()

if(${MODULE_ONNX})
    target_link_libraries(acoustic-model-trainer PRIVATE RasrOnnx)
endif()

if(${MODULE_NN})
    target_link_libraries(acoustic-model-trainer PRIVATE RasrNn)
endif()

if(${MODULE_TENSORFLOW})
    target_link_libraries(acoustic-model-trainer PRIVATE RasrTensorflow)
endif()
