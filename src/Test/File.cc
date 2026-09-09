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
#include <Test/File.hh>
#include <cstdio>
#include <cstdlib>
#include <unistd.h>

#include <Core/Application.hh>
#include <Core/Parameter.hh>

namespace Test {

namespace {
const Core::ParameterString paramDataDir(
        "test-data-dir",
        "Directory containing the unit-test data files. The default assumes the unit-test binary "
        "is started from the repository root; set this when running it from a different working "
        "directory (e.g. --test-data-dir=data when running from within src/Test).",
        "src/Test/data");
}  // namespace

std::string dataDir() {
    return paramDataDir(Core::Application::us()->getConfiguration());
}

std::string dataFile(const std::string& relativePath) {
    return Core::joinPaths(dataDir(), relativePath);
}

void Directory::create() {
    char*       t = std::getenv("TMPDIR");
    std::string tmpdir;
    if (t)
        tmpdir = t;
    else
        tmpdir = "/tmp";
    const size_t len          = tmpdir.length() + 8;
    char*        dir_template = new char[len];
    snprintf(dir_template, len, "%s/XXXXXX", tmpdir.c_str());
    path_ = ::mkdtemp(dir_template);
    delete[] dir_template;
}

void Directory::remove() {
    Core::removeDirectory(path_);
}

}  // namespace Test
