add_executable(lattice-processor LatticeProcessor.cc)
add_install_executable(lattice-processor)

target_link_libraries(lattice-processor PRIVATE
        RasrSpeech
        RasrLattice
        RasrLm
        RasrAm
        RasrMm
        RasrMc
        RasrBliss
        RasrAudio
        RasrSignal
        RasrFlow
        RasrMath
        RasrMathLapack
        RasrCore
        RasrFsa
        RasrFlfCore
        ${SEARCH_LIBS}
)

if (${MODULE_CART})
    target_link_libraries(lattice-processor PRIVATE RasrCart)
endif ()

if (${MODULE_MATH_NR})
    target_link_libraries(lattice-processor PRIVATE RasrMathNr)
endif ()

if (${MODULE_PYTHON})
    target_link_libraries(lattice-processor PRIVATE RasrPython)
    add_python_dependencies(lattice-processor)
endif ()

if (${MODULE_NN})
    target_link_libraries(lattice-processor PRIVATE RasrNn)
endif ()

if (${MODULE_TENSORFLOW})
    target_link_libraries(lattice-processor PRIVATE RasrTensorflow)
    add_tf_dependencies(lattice-processor)
endif ()
