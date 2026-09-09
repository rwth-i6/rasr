add_executable(feature-extraction FeatureExtraction.cc FeatureExtractor.cc)
add_install_executable(feature-extraction)

target_link_libraries(
    feature-extraction
    PRIVATE RasrAm
            RasrAudio
            RasrCore
            RasrFlow
            RasrLm
            RasrMath
            RasrMm
            RasrSignal
            RasrSpeech
)
if(${MODULE_NN})
    target_link_libraries(feature-extraction PRIVATE RasrNn)
endif()

if(${MODULE_TENSORFLOW})
    target_link_libraries(feature-extraction PRIVATE RasrTensorflow)
endif()
