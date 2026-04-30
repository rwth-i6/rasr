add_executable(xml2line xml2line.c)
add_install_executable(xml2line)
target_link_libraries(xml2line PRIVATE LibXml2::LibXml2)

add_executable(line2xml line2xml.c)
add_install_executable(line2xml)
target_link_libraries(line2xml PRIVATE LibXml2::LibXml2)
