find_package(pybind11 REQUIRED)

set(librasr_SOURCES LibRASR.cc PybindModule.cc)

pybind11_add_module(librasr PybindModule.cc LibRASR.cc Search.cc Align.cc)
add_python_dependencies(librasr)

target_link_libraries(
    librasr
    PRIVATE RasrAm
            RasrAudio
            RasrCore
            RasrFlf
            RasrFlow
            RasrLm
            RasrMath
            RasrMm
            RasrNn
            RasrPython
            RasrSignal
            RasrSpeech
)

if(${MODULE_ONNX})
    target_link_libraries(librasr PRIVATE RasrOnnx)
endif()

if(${MODULE_TENSORFLOW})
    target_link_libraries(librasr PRIVATE RasrTensorflow)
endif()
