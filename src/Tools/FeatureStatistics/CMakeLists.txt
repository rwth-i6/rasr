add_executable(feature-statistics FeatureStatistics.cc)
add_install_executable(feature-statistics)

target_link_libraries(
    feature-statistics
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
    target_link_libraries(feature-statistics PRIVATE RasrTensorflow)
endif()
