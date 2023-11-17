add_executable(costa Costa.cc)
add_install_executable(costa)

target_link_libraries(costa PRIVATE
        RasrLm
        RasrMc
        RasrBliss
        RasrFsa
        RasrFlfCore
        RasrAudio
        RasrSignal
        RasrSpeech
        RasrAm
        RasrMm
        RasrFlow
        RasrCore
        RasrMath
        RasrMathLapack
        RasrLattice
        ${SEARCH_LIBS}
)

if (${MODULE_CART})
    target_link_libraries(costa PRIVATE RasrCart)
endif ()

if (${MODULE_MATH_NR})
    target_link_libraries(costa PRIVATE RasrMathNr)
endif ()

if (${MODULE_PYTHON})
    target_link_libraries(costa PRIVATE RasrPython)
    add_python_dependencies(costa)
endif ()

if (${MODULE_NN})
    target_link_libraries(costa PRIVATE RasrNn)
endif ()

if (${MODULE_TENSORFLOW})
    add_tf_dependencies(costa)
    target_link_libraries(costa PRIVATE RasrTensorflow)
endif ()
