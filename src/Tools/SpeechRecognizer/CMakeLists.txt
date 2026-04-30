add_executable(speech-recognizer SpeechRecognizer.cc)

add_install_executable(speech-recognizer)

set(libraries
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
    list(APPEND libraries RasrSearchWfst RasrOpenFst)
endif()

if(${MODULE_ONNX})
    list(APPEND libraries RasrOnnx)
endif()

if(${MODULE_NN})
    list(APPEND libraries RasrNn)
endif()

if(${MODULE_TENSORFLOW})
    list(APPEND libraries RasrTensorflow)
endif()

target_link_libraries(speech-recognizer PRIVATE ${libraries})

if(${MODULE_SEARCH_WFST})
    add_executable(fsa-search-builder)

    target_sources(fsa-search-builder PRIVATE FsaSearchBuilder.cc)
    add_install_executable(fsa-search-builder)
    target_link_libraries(fsa-search-builder PRIVATE ${libraries})
endif()
