set(EXE_NAME "cart-trainer")
add_executable(${EXE_NAME} CartTrainer.cc)
target_link_libraries(${EXE_NAME} PRIVATE RasrCart RasrCore LibXml2::LibXml2 ZLIB::ZLIB Threads::Threads)
add_install_executable(${EXE_NAME})
