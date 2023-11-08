add_executable(acoustic-model-trainer AcousticModelTrainer.cc)

#target_link_libraries(cart-trainer PRIVATE RasrCart RasrCore LibXml2::LibXml2 ZLIB::ZLIB Threads::Threads)
target_link_libraries(acoustic-model-trainer PRIVATE RasrCart RasrCore)