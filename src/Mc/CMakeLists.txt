set(RasrMm_SOURCES Component.cc)

add_library(RasrMc STATIC ${RasrMm_SOURCES})

target_link_libraries(RasrMc PUBLIC RasrCore)
