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
#ifndef _AM_UTILITIES_HH
#define _AM_UTILITIES_HH

#include <Bliss/Lexicon.hh>
#include <Fsa/Types.hh>

namespace Am {

class LexiconUtilities : public virtual Core::Component {
private:
    Bliss::LexiconRef lexicon_;

    const Bliss::LemmaPronunciation* determineSpecialLemmaPronunciation(const std::string&) const;

public:
    LexiconUtilities(const Core::Configuration&, Bliss::LexiconRef);

    Bliss::Phoneme::Id determineSilencePhoneme() const;
    Fsa::LabelId       determineSilenceLemmaPronunciationId() const;

    const Bliss::LemmaPronunciation* determineSilencePronunciation() const {
        return determineSpecialLemmaPronunciation("silence");
    }
    const Bliss::LemmaPronunciation* determineBlankPronunciation() const {
        return determineSpecialLemmaPronunciation("blank");
    }

    void getInitialAndFinalPhonemes(std::vector<Bliss::Phoneme::Id>& initialPhonemes,
                                    std::vector<Bliss::Phoneme::Id>& finalPhonemes) const;
};

}  // namespace Am

#endif  //_AM_UTILITIES_HH
