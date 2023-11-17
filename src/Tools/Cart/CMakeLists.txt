add_executable(cart-trainer CartTrainer.cc)
target_link_libraries(cart-trainer PRIVATE RasrCart RasrCore)
add_install_executable(cart-trainer)

add_executable(cart-viewer CartViewer.cc)
target_link_libraries(cart-viewer PRIVATE RasrCart RasrCore)
add_install_executable(cart-viewer)
