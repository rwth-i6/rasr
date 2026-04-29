add_executable(speech-recognizer SpeechRecognizer.cc)
add_install_executable(speech-recognizer)

set(SpeechRecognizer_LIBRARIES
    RasrAm
    RasrAudio
    RasrBliss
    RasrCore
    RasrFlow
    RasrLm
    RasrMath
    RasrMm
    RasrSearch
    RasrSignal
    RasrSpeech
)

if(${MODULE_SEARCH_WFST})
    list(APPEND SpeechRecognizer_LIBRARIES RasrSearchWfst RasrOpenFst)
endif()

if(${MODULE_ONNX})
    list(APPEND SpeechRecognizer_LIBRARIES RasrOnnx)
endif()

if(${MODULE_NN})
    list(APPEND SpeechRecognizer_LIBRARIES RasrNn)
endif()

if(${MODULE_TENSORFLOW})
    list(APPEND SpeechRecognizer_LIBRARIES RasrTensorflow)
endif()

target_link_libraries(speech-recognizer PRIVATE ${SpeechRecognizer_LIBRARIES})

if(${MODULE_SEARCH_WFST})
    add_executable(fsa-search-builder FsaSearchBuilder.cc)
    add_install_executable(fsa-search-builder)
    target_link_libraries(
        fsa-search-builder PRIVATE ${SpeechRecognizer_LIBRARIES}
    )
endif()
