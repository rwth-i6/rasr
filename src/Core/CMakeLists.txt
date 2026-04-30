find_package(BISON REQUIRED)

bison_target(
    ArithmeticExpressionParser
    ${CMAKE_CURRENT_SOURCE_DIR}/ArithmeticExpressionParser.yy
    ${CMAKE_CURRENT_SOURCE_DIR}/ArithmeticExpressionParser.cc
    DEFINES_FILE ${CMAKE_CURRENT_SOURCE_DIR}/ArithmeticExpressionParser.hh
    COMPILE_FLAGS "-d"
)

add_library(
    RasrCore STATIC
    ${BISON_ArithmeticExpressionParser_OUTPUTS}
    Application.cc
    Archive.cc
    Assertions.cc
    BinaryStream.cc
    BinaryTree.cc
    BundleArchive.cc
    CacheManager.cc
    Channel.cc
    Choice.cc
    CodeGenerator.cc
    Component.cc
    CompressedStream.cc
    Configuration.cc
    Debug.cc
    Dependency.cc
    Description.cc
    Directory.cc
    DirectoryArchive.cc
    FileArchive.cc
    FormatSet.cc
    IoUtilities.cc
    MappedArchive.cc
    MD5.cc
    MemoryInfo.cc
    MurmurHash.cc
    Parameter.cc
    ProgressIndicator.cc
    ReferenceCounting.cc
    ResourceUsageInfo.cc
    Statistics.cc
    Status.cc
    StopWatch.cc
    StringExpression.cc
    StringUtilities.cc
    TextStream.cc
    Tokenizer.cc
    Types.cc
    Unicode.cc
    Utility.cc
    Version.cc
    XmlBuilder.cc
    XmlBuilder2.cc
    XmlParser.cc
    XmlStream.cc
)

add_dependencies(RasrCore SourceVersion)
target_include_directories(
    RasrCore PRIVATE ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_BINARY_DIR}/src
)
target_link_libraries(RasrCore PUBLIC RasrSystemDependencies)

add_rasr_check_executable(check-core check.cc)

target_link_rasr_check_libraries(check-core PRIVATE RasrCore RasrMath)

add_rasr_check_executable(check-core-xml check-xml.cc)

target_link_rasr_check_libraries(check-core-xml PRIVATE RasrCore)
