add_executable(lm-util LmUtilityTool.cc)
add_install_executable(lm-util)

target_link_libraries(lm-util PRIVATE
        RasrFlf
        RasrFlfCore
        RasrSpeech
        RasrAm
        RasrMc
        RasrBliss
        RasrNn
        RasrMm
        RasrSignal
        RasrFlow
        RasrMath
        RasrMathLapack
        ${SEARCH_LIBS}
        RasrLattice
        RasrFsa
        RasrLm
        RasrCore
)

if (${MODULE_FLF_EXT})
    target_link_libraries(lm-util PRIVATE RasrFlfExt)
endif ()

if (${MODULE_CART})
    target_link_libraries(lm-util PRIVATE RasrCart)
endif ()

if (${MODULE_MATH_NR})
    target_link_libraries(lm-util PRIVATE RasrMathNr)
endif ()

if (${MODULE_PYTHON})
    target_link_libraries(lm-util PRIVATE RasrPython)
    add_python_dependencies(lm-util)
endif ()

if (${MODULE_TENSORFLOW})
    target_link_libraries(lm-util PRIVATE RasrTensorflow)
    add_tf_dependencies(lm-util)
endif ()

