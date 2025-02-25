set(RasrLexiconfreeBeamSearch_SOURCES LexiconfreeBeamSearch.cc)

add_library(RasrLexiconfreeBeamSearch STATIC
            ${RasrLexiconfreeBeamSearch_SOURCES})
target_link_libraries(RasrLexiconfreeBeamSearch INTERFACE RasrSearch RasrBliss
                                                          RasrFsa RasrCore)

if(${MODULE_PYTHON})
  add_python_dependencies(RasrLexiconfreeBeamSearch)
endif()
