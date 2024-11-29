add_executable(speech-recognizer SpeechRecognizer.cc)
add_install_executable(speech-recognizer)

set(SpeechRecognizer_LIBRARIES
        RasrSpeech
        RasrLattice
        RasrAm
        RasrLm
        RasrMm
        RasrMc
        RasrBliss
        RasrAudio
        RasrSignal
        RasrFlow
        RasrMath
        RasrMathLapack
        ${SEARCH_LIBS}
        RasrCore
        RasrFsa)

if (${MODULE_CART})
    list(APPEND SpeechRecognizer_LIBRARIES RasrCart)
endif ()

if (${MODULE_FLF_CORE})
    list(APPEND SpeechRecognizer_LIBRARIES RasrFlfCore)
endif ()

if (${MODULE_MATH_NR})
    list(APPEND SpeechRecognizer_LIBRARIES RasrMathNr)
endif ()

if (${MODULE_SEARCH_WFST})
    list(APPEND SpeechRecognizer_LIBRARIES RasrSearchWfst)
endif ()

if (${MODULE_PYTHON})
    list(APPEND SpeechRecognizer_LIBRARIES RasrPython)
endif ()

if (${MODULE_ONNX})
    list(APPEND SpeechRecognizer_LIBRARIES RasrOnnx)
endif ()

if (${MODULE_NN})
    list(APPEND SpeechRecognizer_LIBRARIES RasrNn)
endif ()

if (${MODULE_TENSORFLOW})
    list(APPEND SpeechRecognizer_LIBRARIES RasrTensorflow)
endif ()

target_link_libraries(speech-recognizer PRIVATE ${SpeechRecognizer_LIBRARIES})

if (${MODULE_ONNX})
    add_onnx_dependencies(speech-recognizer)
endif ()

if (${MODULE_SEARCH_WFST})
    add_executable(fsa-search-builder FsaSearchBuilder.cc)
    add_install_executable(fsa-search-builder)
    target_link_libraries(fsa-search-builder
            PRIVATE ${SpeechRecognizer_LIBRARIES})
endif ()
