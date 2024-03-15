set(RasrOpenFst_SOURCES Input.cc Module.cc Output.cc SymbolTable.cc)

add_library(RasrOpenFst STATIC ${RasrOpenFst_SOURCES})

target_link_libraries(RasrOpenFst INTERFACE RasrBliss RasrFsa RasrCore)

add_executable(check-openfst check.cc)
target_link_libraries(check-openfst PRIVATE RasrOpenFst)