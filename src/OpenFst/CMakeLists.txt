add_library(RasrOpenFst STATIC Input.cc Module.cc Output.cc SymbolTable.cc)

target_link_libraries(RasrOpenFst PUBLIC RasrCore RasrFsa)

add_rasr_check_executable(check-openfst check.cc)

target_link_rasr_check_libraries(check-openfst PRIVATE RasrOpenFst)
