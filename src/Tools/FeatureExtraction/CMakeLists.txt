add_executable(feature-extraction FeatureExtraction.cc FeatureExtractor.cc)
add_install_executable(feature-extraction)

target_link_libraries(
    feature-extraction
    PRIVATE RasrAm
            RasrAudio
            RasrFlow
            RasrLm
            RasrCore
            RasrMath
            RasrMm
            RasrNn
            RasrSignal
            RasrSpeech
)

if(${MODULE_TENSORFLOW})
    target_link_libraries(feature-extraction PRIVATE RasrTensorflow)
endif()
