set(RasrLexiconfreeLabelsyncBeamSearch_SOURCES
    LexiconfreeLabelsyncBeamSearch.cc)

add_library(RasrLexiconfreeLabelsyncBeamSearch STATIC
            ${RasrLexiconfreeLabelsyncBeamSearch_SOURCES})
target_link_libraries(RasrLexiconfreeLabelsyncBeamSearch
                      INTERFACE RasrSearch RasrBliss RasrFsa RasrCore)

if(${MODULE_PYTHON})
  add_python_dependencies(RasrLexiconfreeLabelsyncBeamSearch)
endif()
