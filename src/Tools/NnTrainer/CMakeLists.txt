add_executable(nn-trainer NnTrainer.cc)
add_install_executable(nn-trainer)

target_link_libraries(
    nn-trainer
    PRIVATE RasrAudio
            RasrCore
            RasrFlow
            RasrMath
            RasrMm
            RasrNn
            RasrSignal
            RasrSpeech
)

if(${MODULE_TENSORFLOW})
    target_link_libraries(nn-trainer PRIVATE RasrTensorflow)
endif()
