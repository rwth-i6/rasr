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
#ifndef _SEARCH_NON_WORD_TOKENS_HH
#define _SEARCH_NON_WORD_TOKENS_HH

#include <Core/Component.hh>
#include <Am/ClassicStateModel.hh>
#include <Bliss/Lexicon.hh>

namespace Am {
class Allophone;
}

namespace Search { namespace Wfst {

class NonWordTokens : Core::Component
{
private:
    static const Core::ParameterBool paramUseSyntacticTokens;
    static const Core::ParameterBool paramUseSilence;
    static const Core::ParameterStringVector paramLemmas;
    static const Core::ParameterStringVector paramPhones;

public:
    typedef std::map<Bliss::Phoneme::Id, const Am::Allophone*> AllophoneMap;

    NonWordTokens(const Core::Configuration &c, const Bliss::Lexicon &lexicon) :
        Core::Component(c), lexicon_(lexicon), phoneOffset_(0) {}
    ~NonWordTokens();
    bool init();

    const std::vector<const Bliss::LemmaPronunciation*>& lemmaPronunciations() const {
        return lemmaProns_;
    }

    const std::vector<Bliss::Phoneme::Id>& phones() const {
        return phones_;
    }
    bool isNonWordPhone(Bliss::Phoneme::Id phone) const;
    Bliss::Phoneme::Id sourcePhone(Bliss::Phoneme::Id nonWordPhone) const;
    std::string phoneSymbol(Bliss::Phoneme::Id phone) const;

    void createAllophones(Core::Ref<const Am::AllophoneAlphabet> allophoneAlphabet);
    const AllophoneMap& allophones() const {
        return allophones_;
    }
    bool isNonWordAllophone(const Am::Allophone *allophone) const;
    Fsa::LabelId allophoneId(const Am::Allophone *allophone) const;
    const Am::Allophone* allophone(Fsa::LabelId id) const;

    void getEmptySyntacticTokenLemmas(std::vector<const Bliss::Lemma*> &lemmas) const;
    void getEmptySyntacticTokenProns(std::vector<const Bliss::LemmaPronunciation*> &prons) const;

    static const char *phoneSuffix;
private:
    typedef std::map<const Am::Allophone*, Fsa::LabelId> AllophoneIndexMap;
    void setEmptySyntacticTokens();
    void setSilence();
    void setLemmas(const std::vector<std::string> &lemmas);
    void setPhones(const std::vector<std::string> &phones);
    void addPhone(const Bliss::LemmaPronunciation *pron);
    void setNonWordPhones();
    void logSettings() const;
    const Bliss::Lexicon &lexicon_;
    u32 phoneOffset_;
    std::vector<const Bliss::LemmaPronunciation*> lemmaProns_;
    std::vector<Bliss::Phoneme::Id> sourcePhones_, phones_;
    AllophoneMap allophones_;
    AllophoneIndexMap allophoneIndex_;
};

} // namespace Wfst
} // namespace Search

#endif // _SEARCH_NON_WORD_TOKENS_HH
