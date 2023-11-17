add_executable(fsa Fsa.cc)
add_install_executable(fsa)

target_link_libraries(fsa PRIVATE RasrFsa RasrCore)

if (${MODULE_OPENFST})
    target_link_libraries(fsa PRIVATE RasrOpenFst)
endif ()