add_executable(lattice-processor LatticeProcessor.cc)
add_install_executable(lattice-processor)

target_link_libraries(
    lattice-processor
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
    target_link_libraries(lattice-processor PRIVATE RasrTensorflow)
endif()
