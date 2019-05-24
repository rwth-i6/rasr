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
#ifndef _SEARCH_WFST_WORD_END_HH
#define _SEARCH_WFST_WORD_END_HH

#include <Search/Wfst/BookKeeping.hh>
#include <Search/Wfst/StateSequence.hh>

namespace Search {
namespace Wfst {

class WordEndDetector {
public:
    enum WordEndType { WordEndHmm,
                       WordEndOutput };

    WordEndDetector()
            : wordEndType_(WordEndHmm) {}

    void setType(WordEndType t) {
        wordEndType_ = t;
    }
    WordEndType type() const {
        return wordEndType_;
    }

    bool isNonWord(const StateSequence* hmm) const {
        return nonWordHmms_.count(hmm);
    }

    template<class T>
    bool isWordEnd(const T& trace) const {
        if (wordEndType_ == WordEndOutput) {
            return trace.output != OpenFst::Epsilon;
        }
        else {
            return trace.wordEnd && (nonWordHmms_.empty() || !nonWordHmms_.count(trace.input));
        }
    }

    bool isWordEnd(const StateSequence& hmm, OpenFst::Label output) const {
        return (wordEndType_ == WordEndOutput ? output != OpenFst::Epsilon : hmm.isFinal());
    }

    bool setNonWordPhones(Core::Ref<const Am::AcousticModel> am,
                          const StateSequenceList&           stateSequences,
                          const std::vector<std::string>&    phones);
    void setNonWordModels(const StateSequenceList& stateSequences, u32 nNonWordModels);

protected:
    WordEndType                    wordEndType_;
    std::set<const StateSequence*> nonWordHmms_;
};

}  // namespace Wfst
}  // namespace Search

#endif  // _SEARCH_WFST_WORD_END_HH
