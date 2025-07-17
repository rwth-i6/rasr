add_executable(costa Costa.cc)
add_install_executable(costa)

target_link_libraries(
    costa
    PRIVATE RasrAudio
            RasrBliss
            RasrCore
            RasrFlow
            RasrLm
            RasrSignal
            RasrSpeech
)

if(${MODULE_TENSORFLOW})
    target_link_libraries(costa PRIVATE RasrTensorflow)
endif()
