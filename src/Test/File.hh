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
#ifndef _TEST_FILE_HH
#define _TEST_FILE_HH

#include <string>

#include <Core/Directory.hh>

namespace Test {

/**
 * Directory containing the checked-in unit-test data files.
 *
 * Defaults to "src/Test/data", i.e. it assumes the test binary is started from
 * the repository root. Override it with the command-line option
 *   --test-data-dir=<path>
 * e.g. "--test-data-dir=data" when running the binary from within src/Test.
 */
std::string dataDir();

/** Path to a file below the unit-test data directory, e.g.
 *  dataFile("arpa_lm/unigram.arpa.gz"). Honors --test-data-dir (see dataDir()). */
std::string dataFile(const std::string& relativePath);

/**
 * Directory for intermediate files.
 * The directory and all included files are deleted when the object is destroyed.
 */
class Directory {
public:
    Directory() {
        create();
    }
    ~Directory() {
        remove();
    }
    const std::string& path() const {
        return path_;
    }

private:
    void        create();
    void        remove();
    std::string path_;
};

/**
 * A Temporary file.
 */
class File {
public:
    File(const Directory& dir, const std::string& name) {
        path_ = Core::joinPaths(dir.path(), name);
    }
    const std::string& path() const {
        return path_;
    }

private:
    void        create(const std::string& dir);
    std::string path_;
};

}  // namespace Test

#endif  // _TEST_FILE_HH
