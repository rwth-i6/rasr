add_executable(speech-recognizer SpeechRecognizer.cc)
add_install_executable(speech-recognizer)

target_link_libraries(
    speech-recognizer
    PRIVATE RasrAm
            RasrAudio
            RasrBliss
            RasrFlow
            RasrLm
            RasrCore
            RasrMath
            RasrMm
            RasrNn
            RasrSearch
            RasrSignal
            RasrSpeech
)

if(${MODULE_ONNX})
    target_link_libraries(speech-recognizer PRIVATE RasrOnnx)
endif()

if(${MODULE_TENSORFLOW})
    target_link_libraries(speech-recognizer PRIVATE RasrTensorflow)
endif()

if(${MODULE_SEARCH_WFST})
    add_executable(fsa-search-builder FsaSearchBuilder.cc)
    add_install_executable(fsa-search-builder)

    target_link_libraries(fsa-search-builder PRIVATE RasrAm RasrCore RasrOpenFst RasrSearch RasrSpeech)
endif()
