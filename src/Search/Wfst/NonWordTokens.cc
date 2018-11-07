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
#include <Search/Wfst/NonWordTokens.hh>
#include <Am/ClassicAcousticModel.hh>
#include <Bliss/Fsa.hh>

using namespace Search::Wfst;

const Core::ParameterBool NonWordTokens::paramUseSyntacticTokens(
    "use-syntactic-tokens", "use lemmas with empty syntactic token sequence", false);
const Core::ParameterBool NonWordTokens::paramUseSilence(
    "use-silence", "use only the silence lemma", false);
const Core::ParameterStringVector NonWordTokens::paramLemmas(
    "lemmas", "list of lemmas considered as non word tokens", ",");
const Core::ParameterStringVector NonWordTokens::paramPhones(
    "phones", "list of phones associated with non word tokens", ",");

const char* NonWordTokens::phoneSuffix = "__";

NonWordTokens::~NonWordTokens()
{
    for (AllophoneMap::iterator a = allophones_.begin(); a != allophones_.end(); ++a)
        delete a->second;
}

bool NonWordTokens::init()
{
    std::vector<std::string> lemmas = paramLemmas(config);
    std::vector<std::string> phones = paramPhones(config);
    if (paramUseSyntacticTokens(config)) {
        log("using empty syntactic tokens");
        setEmptySyntacticTokens();
    } else if (paramUseSilence(config)) {
        log("using only silence");
        setSilence();
    } else if (!lemmas.empty()) {
        log("using %d lemmas", static_cast<int>(lemmas.size()));
        setLemmas(lemmas);
    } else if (!phones.empty()) {
        log("using %d phones", static_cast<int>(phones.size()));
        setPhones(phones);
    }
    std::sort(lemmas.begin(), lemmas.end());
    std::sort(phones.begin(), phones.end());
    setNonWordPhones();
    logSettings();
    return true;
}

void NonWordTokens::logSettings() const
{
    std::string lemmaprons;
    for (std::vector<const Bliss::LemmaPronunciation*>::const_iterator lp = lemmaProns_.begin();
            lp != lemmaProns_.end(); ++lp) {
        lemmaprons += lexicon_.lemmaPronunciationAlphabet()->symbol((*lp)->id()) + " ";
    }
    log("%d non-word lemma pronunciations: %s",
        static_cast<int>(lemmaProns_.size()), lemmaprons.c_str());
    std::string phonesymbols;
    for (std::vector<Bliss::Phoneme::Id>::const_iterator p = sourcePhones_.begin();
            p != sourcePhones_.end(); ++p) {
        phonesymbols += lexicon_.phonemeInventory()->phonemeAlphabet()->symbol(*p) + " ";
    }
    log("%d non-word phones: %s",
        static_cast<int>(phones_.size()), phonesymbols.c_str());
}

std::string NonWordTokens::phoneSymbol(Bliss::Phoneme::Id phone) const
{
    s32 id = phone - phoneOffset_;
    verify(id > 0);
    std::string symbol = lexicon_.phonemeInventory()->phonemeAlphabet()->symbol(id);
    return symbol + phoneSuffix;
}

Bliss::Phoneme::Id NonWordTokens::sourcePhone(Bliss::Phoneme::Id nonWordPhone) const
{
    std::vector<Bliss::Phoneme::Id>::const_iterator i = std::find(phones_.begin(), phones_.end(), nonWordPhone);
    if (i == phones_.end()) return Bliss::Phoneme::term;
    u32 d = std::distance(phones_.begin(), i);
    return *(sourcePhones_.begin() + d);
}

bool NonWordTokens::isNonWordPhone(Bliss::Phoneme::Id phone) const
{
    return std::find(phones_.begin(), phones_.end(), phone) != phones_.end();
}

bool NonWordTokens::isNonWordAllophone(const Am::Allophone *allophone) const
{
    return allophoneIndex_.count(allophone);
}

Fsa::LabelId NonWordTokens::allophoneId(const Am::Allophone *allophone) const
{
    AllophoneIndexMap::const_iterator i = allophoneIndex_.find(allophone);
    if (i != allophoneIndex_.end())
        return i->second;
    else
        return Fsa::InvalidLabelId;
}

const Am::Allophone* NonWordTokens::allophone(Fsa::LabelId id) const {
    // @todo: do this more efficiently?
    for (AllophoneIndexMap::const_iterator i = allophoneIndex_.begin();
            i != allophoneIndex_.end(); ++i) {
        if (i->second == id) return i->first;
    }
    return 0;
}

void NonWordTokens::createAllophones(Core::Ref<const Am::AllophoneAlphabet> allophoneAlphabet)

{
    if (!allophones_.empty()) return;
    typedef std::vector<Bliss::Phoneme::Id>::const_iterator Iter;
    Fsa::LabelId index = allophoneAlphabet->nClasses();
    for (Iter p = phones_.begin(); p != phones_.end(); ++p) {
        Am::Allophone *allophone = new Am::Allophone(*p, Am::Allophone::isInitialPhone | Am::Allophone::isFinalPhone);
        allophones_.insert(std::make_pair(*p, allophone));
        allophoneIndex_.insert(std::make_pair(allophone, index));
        ++index;
    }
}

void NonWordTokens::getEmptySyntacticTokenLemmas(std::vector<const Bliss::Lemma*> &lemmas) const
{
    lemmas.clear();
    Bliss::Lexicon::LemmaIterator lemma, lEnd;
    Core::tie(lemma, lEnd) = lexicon_.lemmas();
    for (; lemma != lEnd; ++lemma) {
        if ((*lemma)->syntacticTokenSequence().isEpsilon())
            lemmas.push_back(*lemma);
    }
}

void NonWordTokens::getEmptySyntacticTokenProns(std::vector<const Bliss::LemmaPronunciation*> &prons) const
{
    prons.clear();
    std::vector<const Bliss::Lemma*> lemmas;
    getEmptySyntacticTokenLemmas(lemmas);
    for (std::vector<const Bliss::Lemma*>::const_iterator lemma = lemmas.begin();
            lemma != lemmas.end(); ++lemma) {
        Bliss::Lemma::LemmaPronunciationRange lp = (*lemma)->pronunciations();
        for (Bliss::Lemma::PronunciationIterator p = lp.first; p != lp.second; ++p) {
            prons.push_back(p);
        }
    }
}

void NonWordTokens::setEmptySyntacticTokens()
{
    lemmaProns_.clear();
    getEmptySyntacticTokenProns(lemmaProns_);
    for (std::vector<const Bliss::LemmaPronunciation*>::const_iterator lp = lemmaProns_.begin();
            lp != lemmaProns_.end(); ++lp) {
        addPhone(*lp);
    }
}

void NonWordTokens::setSilence()
{
    const Bliss::Lemma *silLemma = lexicon_.specialLemma("silence");
    verify(silLemma);
    verify(silLemma->nPronunciations() == 1);
    const Bliss::LemmaPronunciation *pron = silLemma->pronunciations().first;
    verify(pron);
    verify(pron->pronunciation()->length() == 1);
    addPhone(pron);
}

void NonWordTokens::setLemmas(const std::vector<std::string> &lemmas)
{
    Core::Ref<const Bliss::LemmaPronunciationAlphabet> lp = lexicon_.lemmaPronunciationAlphabet();
    for (std::vector<std::string>::const_iterator symbol = lemmas.begin(); symbol != lemmas.end(); ++symbol) {
        const Bliss::Lemma *lemma = lexicon_.lemma(*symbol);
        if (!lemma)
            error("unknown lemma symbol '%s'", symbol->c_str());
        else {
            Bliss::Lemma::LemmaPronunciationRange prons = lemma->pronunciations();
            for (Bliss::Lemma::PronunciationIterator p = prons.first; p != prons.second; ++p) {
                lemmaProns_.push_back(p);
                addPhone(p);
            }
        }
    }
}

void NonWordTokens::setPhones(const std::vector<std::string> &phones)
{
    Core::Ref<const Bliss::PhonemeAlphabet> phoneAlphabet =
            lexicon_.phonemeInventory()->phonemeAlphabet();
    for (std::vector<std::string>::const_iterator symbol = phones.begin(); symbol != phones.end(); ++symbol) {
        Fsa::LabelId p = phoneAlphabet->index(*symbol);
        verify(p != Fsa::InvalidLabelId);
        verify(!phoneAlphabet->isDisambiguator(p));
        sourcePhones_.push_back(p);
    }
}

void NonWordTokens::addPhone(const Bliss::LemmaPronunciation *pron)
{
    sourcePhones_.push_back((*pron->pronunciation())[0]);
}

/**
 * set phoneOffset_ and transform sourcePhones_ to phones_
 */
void NonWordTokens::setNonWordPhones()
{
    phoneOffset_ = lexicon_.phonemeInventory()->nPhonemes();
    for (std::vector<Bliss::Phoneme::Id>::const_iterator p = sourcePhones_.begin();
            p != sourcePhones_.end(); ++p) {
        phones_.push_back(*p + phoneOffset_);
    }
}
