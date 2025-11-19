#ifndef ARCHIVE_IO
#define ARCHIVE_IO

#include <Core/Hash.hh>
#include <Core/MappedArchive.hh>

// -- reader --
template<class T>
Core::MappedArchiveReader& operator>>(Core::MappedArchiveReader& reader, std::unordered_set<T>& target) {
    target.clear();
    std::vector<T> vec;
    reader >> vec;
    target.insert(vec.begin(), vec.end());
    return reader;
}

template<class T>
Core::MappedArchiveReader& operator>>(Core::MappedArchiveReader& reader, std::set<T>& target) {
    target.clear();
    std::vector<T> vec;
    reader >> vec;
    target.insert(vec.begin(), vec.end());
    return reader;
}

template<class T, class T2>
Core::MappedArchiveReader& operator>>(Core::MappedArchiveReader& reader, std::unordered_map<T, T2>& target) {
    target.clear();
    std::vector<std::pair<T, T2>> vec;
    reader >> vec;
    target.insert(vec.begin(), vec.end());
    return reader;
}

template<class T, class T2>
Core::MappedArchiveReader& operator>>(Core::MappedArchiveReader& reader, std::map<T, T2>& target) {
    target.clear();
    std::vector<std::pair<T, T2>> vec;
    reader >> vec;
    target.insert(vec.begin(), vec.end());
    return reader;
}

template<class T, class T2>
Core::MappedArchiveReader& operator>>(Core::MappedArchiveReader& reader, Core::HashMap<T, T2>& target) {
    target.clear();
    std::vector<std::pair<T, T2>> vec;
    reader >> vec;
    target.insert(vec.begin(), vec.end());
    return reader;
}

template<class T, class T2>
Core::MappedArchiveReader& operator>>(Core::MappedArchiveReader& reader, Core::HashMap<T, T2, typename T::Hash>& target) {
    target.clear();
    std::vector<std::pair<T, T2>> vec;
    reader >> vec;
    target.insert(vec.begin(), vec.end());
    return reader;
}

// -- writer --
template<class T>
Core::MappedArchiveWriter& operator<<(Core::MappedArchiveWriter& writer, const std::unordered_set<T>& set) {
    writer << std::vector<T>(set.begin(), set.end());
    return writer;
}

template<class T>
Core::MappedArchiveWriter& operator<<(Core::MappedArchiveWriter& writer, const std::set<T>& set) {
    writer << std::vector<T>(set.begin(), set.end());
    return writer;
}

template<class T, class T2>
Core::MappedArchiveWriter& operator<<(Core::MappedArchiveWriter& writer, const std::unordered_map<T, T2>& map) {
    writer << std::vector<std::pair<T, T2>>(map.begin(), map.end());
    return writer;
}

template<class T, class T2>
Core::MappedArchiveWriter& operator<<(Core::MappedArchiveWriter& writer, const std::map<T, T2>& map) {
    writer << std::vector<std::pair<T, T2>>(map.begin(), map.end());
    return writer;
}

template<class T, class T2>
Core::MappedArchiveWriter& operator<<(Core::MappedArchiveWriter& writer, const Core::HashMap<T, T2>& map) {
    writer << std::vector<std::pair<T, T2>>(map.begin(), map.end());
    return writer;
}

template<class T, class T2>
Core::MappedArchiveWriter& operator<<(Core::MappedArchiveWriter& writer, const Core::HashMap<T, T2, typename T::Hash>& map) {
    writer << std::vector<std::pair<T, T2>>(map.begin(), map.end());
    return writer;
}

#endif
