set(RasrLexiconfreeGreedySearch_SOURCES LexiconfreeGreedySearch.cc)

add_library(RasrLexiconfreeGreedySearch STATIC
            ${RasrLexiconfreeGreedySearch_SOURCES})
target_link_libraries(RasrLexiconfreeGreedySearch
                      INTERFACE RasrSearch RasrBliss RasrFsa RasrCore)

if(${MODULE_PYTHON})
  add_python_dependencies(RasrLexiconfreeGreedySearch)
endif()
