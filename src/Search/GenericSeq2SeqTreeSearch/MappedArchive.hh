/** Copyright 2020 RWTH Aachen University. All rights reserved.
 *
 *  Licensed under the RWTH ASR License (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.hltpr.rwth-aachen.de/rwth-asr/rwth-asr-license.html
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *  author: Wei Zhou
 */

#include <Core/MappedArchive.hh>

// interfaces for image I/O

template <class T>
Core::MappedArchiveWriter& operator<<(Core::MappedArchiveWriter& writer, const std::unordered_set<T>& source) {
  writer << std::vector<T>(source.begin(), source.end());
  return writer;
}

template <class T>
Core::MappedArchiveReader& operator>>(Core::MappedArchiveReader& reader, std::unordered_set<T>& target) {
  target.clear();
  std::vector<T> vec;
  reader >> vec; 
  target.insert(vec.begin(), vec.end());
  return reader;
}

template <class T, class T2>
Core::MappedArchiveWriter& operator<<(Core::MappedArchiveWriter& writer, const std::unordered_map<T, T2>& source) {
  writer << std::vector<std::pair<T, T2> >(source.begin(), source.end());
  return writer;
}

template <class T, class T2>
Core::MappedArchiveReader& operator>>(Core::MappedArchiveReader& reader, std::unordered_map<T, T2>& target) {
  target.clear();
  std::vector<std::pair<T, T2> > vec;
  reader >> vec; 
  target.insert(vec.begin(), vec.end());
  return reader;
}

template <class T, class T2>
Core::MappedArchiveWriter& operator<<(Core::MappedArchiveWriter& writer, const std::unordered_map<T, std::vector<T2> >& source) {
  std::vector<T> vec1;
  std::vector<std::vector<T2> > vec2;
  for (const auto& kv : source) {
    vec1.push_back(kv.first);
    vec2.push_back(kv.second);
  }
  writer << vec1 << vec2;
  return writer;
}

template <class T, class T2>
Core::MappedArchiveReader& operator>>(Core::MappedArchiveReader& reader, std::unordered_map<T, std::vector<T2> >& target) {
  target.clear();
  std::vector<T> vec1;
  std::vector<std::vector<T2> > vec2;
  reader >> vec1 >> vec2;
  verify(vec1.size() == vec2.size());
  for (u32 idx = 0, size = vec1.size(); idx < size; ++idx)
    target.insert(std::make_pair(vec1[idx], vec2[idx]));
  return reader;
}

template <class T, class T2>
Core::MappedArchiveWriter& operator<<(Core::MappedArchiveWriter& writer, const std::map<T, std::vector<T2>, std::greater<T> >& source) {
  std::vector<T> vec1;
  std::vector<std::vector<T2> > vec2;
  for (const auto& kv : source) {
    vec1.push_back(kv.first);
    vec2.push_back(kv.second);
  }
  writer << vec1 << vec2;
  return writer;
}

template <class T, class T2>
Core::MappedArchiveReader& operator>>(Core::MappedArchiveReader& reader, std::map<T, std::vector<T2>, std::greater<T> >& target) {
  target.clear();
  std::vector<T> vec1;
  std::vector<std::vector<T2> > vec2;
  reader >> vec1 >> vec2;
  verify(vec1.size() == vec2.size());
  for (u32 idx = 0, size = vec1.size(); idx < size; ++idx)
    target.insert(std::make_pair(vec1[idx], vec2[idx]));
  return reader;
}

