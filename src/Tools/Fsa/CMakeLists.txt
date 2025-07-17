add_executable(fsa Fsa.cc)
add_install_executable(fsa)

target_link_libraries(fsa PRIVATE RasrCore RasrFsa)

if(${MODULE_OPENFST})
    target_link_libraries(fsa PRIVATE RasrOpenFst)
endif()
