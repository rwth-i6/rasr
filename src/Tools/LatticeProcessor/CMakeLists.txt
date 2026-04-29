add_executable(lattice-processor LatticeProcessor.cc)
add_install_executable(lattice-processor)

target_link_libraries(
    lattice-processor
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
    target_link_libraries(lattice-processor PRIVATE RasrNn)
endif()

if(${MODULE_TENSORFLOW})
    target_link_libraries(lattice-processor PRIVATE RasrTensorflow)
endif()
