add_executable(nn-trainer NnTrainer.cc)
add_install_executable(nn-trainer)

target_link_libraries(
        nn-trainer
        PRIVATE RasrAm
        RasrAudio
        RasrBliss
        RasrFlow
        RasrLm
        RasrMathLapack
        RasrMath
        RasrMc
        RasrMm
        RasrSignal
        RasrSpeech
        ${SEARCH_LIBS}
        RasrNn
        RasrLabelScorer
        RasrFsa
        RasrCore)

if (${MODULE_CART})
        target_link_libraries(nn-trainer PRIVATE RasrCart)
endif ()

if (${MODULE_FLF_CORE})
        target_link_libraries(nn-trainer PRIVATE RasrFlfCore)
endif ()

if (${MODULE_MATH_NR})
        target_link_libraries(nn-trainer PRIVATE RasrMathNr)
endif ()

if (${MODULE_PYTHON})
        target_link_libraries(nn-trainer PRIVATE RasrPython)
        add_python_dependencies(nn-trainer)
endif ()

if (${MODULE_TENSORFLOW})
        target_link_libraries(nn-trainer PRIVATE RasrTensorflow)
        add_tf_dependencies(nn-trainer)
endif ()
