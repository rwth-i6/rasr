add_executable(lm-util LmUtilityTool.cc)
add_install_executable(lm-util)

target_link_libraries(
    lm-util
    PRIVATE RasrBliss
            RasrCore
            RasrFlf
            RasrFlow
            RasrLm
            RasrMath
            RasrMm
            RasrNn
            RasrSignal
            RasrSpeech
)

if(${MODULE_TENSORFLOW})
    target_link_libraries(lm-util PRIVATE RasrTensorflow)
endif()
