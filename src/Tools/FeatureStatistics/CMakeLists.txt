add_executable(feature-statistics FeatureStatistics.cc)
add_install_executable(feature-statistics)

target_link_libraries(
    feature-statistics
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
    target_link_libraries(feature-statistics PRIVATE RasrNn)
endif()

if(${MODULE_TENSORFLOW})
    target_link_libraries(feature-statistics PRIVATE RasrTensorflow)
endif()
