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
#include "TheanoSegmentOrderingVisitor.hh"

#include <Core/Application.hh>

#include "TheanoCommunicator.hh"

using namespace Bliss;

TheanoSegmentOrderingVisitor::~TheanoSegmentOrderingVisitor() {
}

void TheanoSegmentOrderingVisitor::leaveCorpus(Bliss::Corpus* corpus) {
    curCorpus_.pop_back();
    if (!curCorpus_.empty()) {
        // not the root corpus
        return;
    }

    CustomCorpusGuide corpusGuide(this, /* root */ corpus);
    std::string       name;
    while (TheanoCommunicator::communicator().waitForErrorSignalRequest(name))
        corpusGuide.showSegmentByName(name);
}

SegmentOrderingVisitor* TheanoSegmentOrderingVisitor::copy() {
    Core::Application::us()->error(
            "TheanoSegmentOrderingVisitor: copy not supported (check progress-indication != global)");
    // copy() is currently only used by CorpusDescription::totalSegmentCount().
    // Maybe this could be implemented -- however, not sure if it's worth.
    return 0;
}

void TheanoSegmentOrderingVisitor::setAutoShuffle(bool enabled) {
    if (enabled)
        Core::Application::us()->error("TheanoSegmentOrderingVisitor: auto-shuffle not supported");
}

void TheanoSegmentOrderingVisitor::setSegmentList(const std::string& filename) {
    Core::Application::us()->error(
            "TheanoSegmentOrderingVisitor: segment list not supported: %s", filename.c_str());
}
