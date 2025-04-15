set(Core_SOURCES
    Application.cc
    Archive.cc
    ArithmeticExpressionParser.cc
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
    MemoryInfo.cc
    MD5.cc
    MurmurHash.cc
    Parameter.cc
    ProgressIndicator.cc
    ReferenceCounting.cc
    ResourceUsageInfo.cc
    Statistics.cc
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
    XmlStream.cc)

find_package(BISON REQUIRED)

bison_target(
  ArithmeticExpressionParser
  ${CMAKE_CURRENT_SOURCE_DIR}/ArithmeticExpressionParser.yy
  ${CMAKE_CURRENT_SOURCE_DIR}/ArithmeticExpressionParser.cc
  DEFINES_FILE ${CMAKE_CURRENT_SOURCE_DIR}/ArithmeticExpressionParser.hh
  COMPILE_FLAGS "-d")
list(APPEND Core_SOURCES ${BISON_ArithmeticExpressionParser_OUTPUTS})

add_library(RasrCore STATIC ${Core_SOURCES})
add_dependencies(RasrCore SourceVersion)

add_executable(check-core check.cc)
target_link_libraries(check-core PRIVATE RasrCore)
