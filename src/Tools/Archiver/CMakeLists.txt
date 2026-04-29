add_executable(archiver Archiver.cc)
add_install_executable(archiver)

target_link_libraries(archiver PRIVATE RasrCore RasrFlow RasrMath RasrSpeech)
