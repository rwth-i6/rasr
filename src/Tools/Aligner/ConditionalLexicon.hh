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
#ifndef CONDITIONAL_LEXICON_HH
#define CONDITIONAL_LEXICON_HH

#include <Core/Component.hh>
#include <Core/CompressedStream.hh>
#include <Core/ReferenceCounting.hh>
#include <Fsa/Automaton.hh>
#include <Translation/Common.hh>
#include <Translation/PrefixTree.hh>
#include <vector>

#include <iostream>

namespace Translation {

typedef enum {
    lexiconTypePlain,
    lexiconTypeSri
} lexiconTypes;

const Core::Choice lexiconTypeChoice(
        "plain", lexiconTypePlain,
        "sri", lexiconTypeSri,
        Core::Choice::endMark());

class ConditionalLexicon : public Core::Component, public Core::ReferenceCounted {
public:
    ConditionalLexicon(const Core::Configuration& config)
            : Core::Component(config),
              tokens_(new Fsa::StaticAlphabet()),
              tokenRef_(tokens_) {}

    ConditionalLexicon(const Core::Configuration& config, Fsa::ConstAlphabetRef alphabet)
            : Core::Component(config),
              tokens_(staticCopy(alphabet).get()),
              tokenRef_(tokens_)

    {}

    //! this is deprecated and should just exist as long as i am migrating to the new lexicon
    virtual Translation::Cost getProb(const size_t index, const std::vector<std::string>& key) const = 0;

    virtual Translation::Cost getCost(const size_t index, const std::vector<Fsa::LabelId>& key) const = 0;

    virtual Translation::Cost getReverseCost(const size_t index, const std::vector<Fsa::LabelId>& key) const = 0;

    //! get probability of a lexicon entry or floor if it does not exist
    virtual Translation::Cost getProb(const size_t index, const std::vector<Fsa::LabelId>& key) const = 0;

    //! add value to existing count/prob or create new if it does not exist
    virtual void addValue(const size_t index, const std::vector<Fsa::LabelId>& key, Translation::Cost value) = 0;

    //! add value to existing count/prob or create new if it does not exist
    virtual void addValue(const size_t index, const std::vector<std::string>& key, Translation::Cost value) = 0;

    //! set value of the given entry (overwrite if it exists, create if it doesnt)
    virtual void setValue(const size_t index, const std::vector<Fsa::LabelId>& key, Translation::Cost value) = 0;

    //! set value of the given entry (overwrite if it exists, create if it doesnt)
    virtual void setValue(const size_t index, const std::vector<std::string>& key, Translation::Cost value) = 0;

    //! write lexicon to stream
    virtual void write(std::ostream&) = 0;

    //! normalize
    virtual void normalize(int order) = 0;

protected:
    //! tokens (no distinctions between source or target)
    Fsa::StaticAlphabet*  tokens_;
    Fsa::ConstAlphabetRef tokenRef_;

    //! parameter giving the filename to read from
    //static Core::ParameterString paramFilename_;

public:
    //! read a lexicon from a file stream
    virtual void read(std::istream&) = 0;

    //! read a lexicon from a file stream
    virtual void read() = 0;

    //! return ConstAlphabetRef of the internal token alphabet
    //! for matching against other alphabets
    ConstAlphabetRef getTokenAlphabet() const {
        return Fsa::ConstAlphabetRef(tokens_);
    }
};

typedef Core::Ref<ConditionalLexicon>       ConditionalLexiconRef;
typedef Core::Ref<const ConditionalLexicon> ConstConditionalLexiconRef;
}  // namespace Translation
#endif
