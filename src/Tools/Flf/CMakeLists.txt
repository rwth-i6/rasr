add_executable(flf-tool FlfTool.cc)
add_install_executable(flf-tool)

target_link_libraries(
        flf-tool
        PRIVATE RasrFlf
        RasrFlfCore
        RasrSpeech
        RasrLm
        RasrAm
        RasrMm
        RasrMc
        RasrBliss
        RasrAudio
        RasrCore
        RasrFsa
        ${SEARCH_LIBS}
        RasrLattice)

if (${MODULE_FLF_EXT})
        target_link_libraries(flf-tool PRIVATE RasrFlfExt)
endif ()

target_link_libraries(flf-tool PRIVATE RasrSignal RasrFlow RasrMath
        RasrMathLapack)

if (${MODULE_CART})
        target_link_libraries(flf-tool PRIVATE RasrCart)
endif ()

if (${MODULE_MATH_NR})
        target_link_libraries(flf-tool PRIVATE RasrMathNr)
endif ()

if (${MODULE_PYTHON})
        target_link_libraries(flf-tool PRIVATE RasrPython)
        add_python_dependencies(flf-tool)
endif ()

if (${MODULE_NN})
        target_link_libraries(flf-tool PRIVATE RasrNn RasrLabelScorer)
endif ()

if (${MODULE_ONNX})
        add_onnx_dependencies(flf-tool)
        target_link_libraries(flf-tool PRIVATE RasrOnnx)
endif ()

if (${MODULE_TENSORFLOW})
        target_link_libraries(flf-tool PRIVATE RasrTensorflow)
        add_tf_dependencies(flf-tool)
endif ()
