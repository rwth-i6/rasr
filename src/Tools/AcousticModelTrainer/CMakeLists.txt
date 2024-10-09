add_executable(acoustic-model-trainer AcousticModelTrainer.cc)
add_install_executable(acoustic-model-trainer)
target_link_libraries(
        acoustic-model-trainer
        PRIVATE RasrSpeech
        RasrFlow
        RasrAm
        RasrMm
        RasrLm
        RasrMc
        RasrBliss
        RasrAudio
        RasrSignal
        RasrMath
        RasrMathLapack
        RasrCore
        RasrFsa
        ${SEARCH_LIBS})

add_executable(allophone-tool AllophoneTool.cc)
add_install_executable(allophone-tool)
target_link_libraries(
        allophone-tool
        PRIVATE RasrAm
        RasrMm
        RasrBliss
        RasrFsa
        RasrMc
        RasrCore
        RasrMathLapack)

if (${MODULE_CART})
        target_link_libraries(acoustic-model-trainer PRIVATE RasrCart)
        target_link_libraries(allophone-tool PRIVATE RasrCart)
endif ()

if (${MODULE_FLF_CORE})
        target_link_libraries(acoustic-model-trainer PRIVATE RasrFlfCore)
        target_link_libraries(allophone-tool PRIVATE RasrFlfCore)
endif ()

if (${MODULE_MATH_NR})
        target_link_libraries(acoustic-model-trainer PRIVATE RasrMathNr)
        target_link_libraries(allophone-tool PRIVATE RasrMathNr)
endif ()

if (${MODULE_ONNX})
        add_onnx_dependencies(acoustic-model-trainer)
        target_link_libraries(acoustic-model-trainer PRIVATE RasrOnnx)
endif ()

if (${MODULE_PYTHON})
        target_link_libraries(acoustic-model-trainer PRIVATE RasrPython)
        add_python_dependencies(acoustic-model-trainer)
endif ()

if (${MODULE_NN})
        target_link_libraries(acoustic-model-trainer PRIVATE RasrNn RasrLabelScorer)
endif ()

if (${MODULE_TENSORFLOW})
        add_tf_dependencies(acoustic-model-trainer)
        target_link_libraries(acoustic-model-trainer PRIVATE RasrTensorflow)
endif ()
