find_package(pybind11 REQUIRED)

set(librasr_SOURCES LibRASR.cc)

if(${MODULE_PYTHON})
    list(APPEND librasr_SOURCES LabelScorer.cc Lexicon.cc PybindModule.cc
         Search.cc
    )
endif()

pybind11_add_module(librasr ${librasr_SOURCES})

target_link_libraries(
    librasr
    PRIVATE RasrAm
            RasrAudio
            RasrBliss
            RasrCore
            RasrFlf
            RasrFlow
            RasrLm
            RasrMath
            RasrMm
            RasrSignal
            RasrSpeech
)

add_install_library(librasr)

if(${MODULE_PYTHON})
    target_link_libraries(librasr PRIVATE RasrPython)
endif()

if(${MODULE_NN})
    target_link_libraries(librasr PRIVATE RasrNn)
endif()

if(${MODULE_ONNX})
    target_link_libraries(librasr PRIVATE RasrOnnx)
endif()

if(${MODULE_TENSORFLOW})
    target_link_libraries(librasr PRIVATE RasrTensorflow)
endif()
