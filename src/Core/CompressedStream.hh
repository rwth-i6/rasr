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
 */
#ifndef _CORE_COMPRESSED_STREAM_HH
#define _CORE_COMPRESSED_STREAM_HH

#include <fstream>
#include <iostream>
#include <memory>

namespace Core {

class CompressedInputStream : public std::istream {
private:
    std::unique_ptr<std::filebuf> file_buf_;
    std::streambuf*               buf_;

public:
    CompressedInputStream();
    CompressedInputStream(const std::string& name);
    ~CompressedInputStream() {
        close();
    }

    void open(const std::string& name);
    void close();
    bool isOpen() const {
        return buf_;
    }
};

class CompressedOutputStream : public std::ostream {
private:
    std::unique_ptr<std::filebuf> file_buf_;
    std::streambuf*               buf_;

public:
    CompressedOutputStream();
    CompressedOutputStream(const std::string& name);
    ~CompressedOutputStream() {
        close();
    }

    void open(const std::string& name);
    void close();
    bool isOpen() const {
        return buf_;
    }
};

std::string extendCompressedFilename(const std::string& filename, const std::string& extension);

}  // namespace Core

#endif  // _CORE_COMPRESSED_STREAM_HH
