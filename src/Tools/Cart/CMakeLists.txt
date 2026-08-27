add_executable(cart-trainer CartTrainer.cc)
add_install_executable(cart-trainer)

target_link_libraries(cart-trainer PRIVATE RasrCart RasrCore)

add_executable(cart-viewer CartViewer.cc)
add_install_executable(cart-viewer)

target_link_libraries(cart-viewer PRIVATE RasrCart RasrCore)
