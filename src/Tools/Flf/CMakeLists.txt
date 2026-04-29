add_executable(flf-tool FlfTool.cc)
add_install_executable(flf-tool)

target_link_libraries(
    flf-tool
    PRIVATE RasrAm
            RasrAudio
            RasrCore
            RasrFlf
            RasrFlow
            RasrLm
            RasrMath
            RasrMm
            RasrSignal
            RasrSpeech
)
if(${MODULE_NN})
    target_link_libraries(flf-tool PRIVATE RasrNn)
endif()

if(${MODULE_ONNX})
    target_link_libraries(flf-tool PRIVATE RasrOnnx)
endif()

if(${MODULE_TENSORFLOW})
    target_link_libraries(flf-tool PRIVATE RasrTensorflow)
endif()
