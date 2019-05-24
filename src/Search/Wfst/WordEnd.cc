/** Copyright 2018 RWTH Aachen University. All rights reserved.
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
#include <Search/Wfst/WordEnd.hh>

using namespace Search::Wfst;

bool WordEndDetector::setNonWordPhones(Core::Ref<const Am::AcousticModel> am,
                                       const StateSequenceList&           stateSequences,
                                       const std::vector<std::string>&    phones) {
    Bliss::PhonemeInventoryRef             pi         = am->phonology()->getPhonemeInventory();
    Core::Ref<const Am::AllophoneAlphabet> allophones = am->allophoneAlphabet();
    for (std::vector<std::string>::const_iterator p = phones.begin(); p != phones.end(); ++p) {
        const Bliss::Phoneme* phone = pi->phoneme(*p);
        verify(phone);
        Am::AllophoneIndex ai = allophones->index(
                Am::Allophone(phone->id(), Am::Allophone::isInitialPhone | Am::Allophone::isFinalPhone));
        const Am::Allophone* allophone = allophones->allophone(ai);
        verify(allophone);
        StateSequence states;
        states.createFromAllophone(am, allophone);
        StateSequenceList::const_iterator s = std::find(stateSequences.begin(), stateSequences.end(), states);
        if (s == stateSequences.end()) {
            Core::Application::us()->error("unknown non-word allophone %s (phone %s)",
                                           allophones->toString(*allophone).c_str(), p->c_str());
            return false;
        }
        else {
            nonWordHmms_.insert(&(*s));
        }
    }
    return true;
}

void WordEndDetector::setNonWordModels(const StateSequenceList& stateSequences,
                                       u32                      nNonWordModels) {
    u32 i = 0;
    for (StateSequenceList::const_reverse_iterator s = stateSequences.rbegin();
         s != stateSequences.rend() && i < nNonWordModels; ++s, ++i) {
        nonWordHmms_.insert(&(*s));
    }
}
