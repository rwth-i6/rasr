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
#include <Core/CompressedStream.hh>

#include <cstring>

#include <Core/Assertions.hh>
#include <Core/zstr.hh>

using namespace Core;

namespace Core {
}

// ***************************************************************************

CompressedInputStream::CompressedInputStream()
        : std::istream(0),
          file_buf_(nullptr),
          buf_(nullptr) {}

CompressedInputStream::CompressedInputStream(const std::string& name)
        : std::istream(0),
          file_buf_(nullptr),
          buf_(nullptr) {
    open(name);
}

void CompressedInputStream::open(const std::string& name) {
    if (buf_) {
        close();
    }
    if (name == "-") {
        buf_ = std::cin.rdbuf();
    }
    else {
        file_buf_.reset(new std::filebuf());
        if (!file_buf_->open(name, std::ios::in)) {
            setstate(std::ios::failbit);
            file_buf_.reset(nullptr);
            return;
        }
        buf_ = new zstr::istreambuf(file_buf_.get());
    }
    rdbuf(buf_);
}

void CompressedInputStream::close() {
    if (buf_) {
        if (buf_ != std::cin.rdbuf()) {
            delete buf_;
        }
        buf_ = nullptr;
        file_buf_.reset(nullptr);
        rdbuf(0);
    }
}

// ***************************************************************************

CompressedOutputStream::CompressedOutputStream()
        : std::ostream(0),
          file_buf_(nullptr),
          buf_(nullptr) {}

CompressedOutputStream::CompressedOutputStream(const std::string& name)
        : std::ostream(0),
          file_buf_(nullptr),
          buf_(nullptr) {
    open(name);
}

void CompressedOutputStream::open(const std::string& name) {
    if (buf_) {
        close();
    }
    if (name == "-") {
        buf_ = std::cout.rdbuf();
    }
    else if ((name.rfind(".gz") == name.length() - 3) || (name.rfind(".Z") == name.length() - 3)) {
        file_buf_.reset(new std::filebuf());
        if (!file_buf_->open(name, std::ios::out)) {
            setstate(std::ios::failbit);
            return;
        }
        buf_ = new zstr::ostreambuf(file_buf_.get());
    }
    else {
        std::filebuf* buf = new std::filebuf();
        if (!buf->open(name.c_str(), std::ios::out)) {
            setstate(std::ios::failbit);
            delete buf;
            return;
        }
        buf_ = buf;
    }
    rdbuf(buf_);
}

void CompressedOutputStream::close() {
    if (buf_) {
        if (buf_ != std::cout.rdbuf()) {
            delete buf_;
        }
        buf_ = nullptr;
        file_buf_.reset(nullptr);
        rdbuf(0);
    }
}

// ===========================================================================

std::string Core::extendCompressedFilename(const std::string& filename, const std::string& extension) {
    std::string::size_type gzPos  = filename.rfind(".gz");
    std::string::size_type zPos   = filename.rfind(".Z");
    std::string::size_type bz2Pos = filename.rfind(".bz2");
    if (gzPos == filename.length() - 3)
        return std::string(filename, 0, gzPos) + extension + ".gz";
    else if (zPos == filename.length() - 2)
        return std::string(filename, 0, zPos) + extension + ".Z";
    else if (bz2Pos == filename.length() - 3)
        return std::string(filename, 0, bz2Pos) + extension + ".bz2";
    return filename + extension;
}
