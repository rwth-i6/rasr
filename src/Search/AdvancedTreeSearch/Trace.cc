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
#include "Trace.hh"

namespace Search {
void Trace::write(std::ostream& os, Core::Ref<const Bliss::PhonemeInventory> phi) const {
    if (predecessor)
        predecessor->write(os, phi);
    os << "t=" << std::setw(5) << time
       << "    s=" << std::setw(8) << score;

    if (pronunciation) {
        os << "    "
           << std::setw(20) << std::setiosflags(std::ios::left)
           << pronunciation->lemma()->preferredOrthographicForm()
           << "    "
           << "/" << pronunciation->pronunciation()->format(phi) << "/";
    }
    os << std::endl;
}

Trace::Trace(SearchAlgorithm::TimeframeIndex t, ScoreVector s, const Search::TracebackItem::Transit& transit)
        : TracebackItem(0, t, s, transit) {}

Trace::Trace(const Core::Ref<Trace>&               pre,
             const Bliss::LemmaPronunciation*      p,
             SearchAlgorithm::TimeframeIndex       t,
             ScoreVector                           s,
             const Search::TracebackItem::Transit& transit)
        : TracebackItem(p, t, s, transit), predecessor(pre), pruningMark(0) {}

void Trace::getLemmaSequence(std::vector<Bliss::Lemma*>& lemmaSequence) const {
    if (predecessor) {
        predecessor->getLemmaSequence(lemmaSequence);
    }
    if (pronunciation) {
        lemmaSequence.push_back(const_cast<Bliss::Lemma*>(pronunciation->lemma()));
    }
}
}  // namespace Search
