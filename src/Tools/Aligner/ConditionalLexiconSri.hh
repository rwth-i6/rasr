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
#ifndef CONDITIONAL_LEXICON_SRI_HH
#define CONDITIONAL_LEXICON_SRI_HH

#include <Core/Component.hh>
#include <Core/CompressedStream.hh>
#include <Core/ReferenceCounting.hh>
#include <Fsa/Automaton.hh>
#include <Translation/Common.hh>
#include <Translation/PrefixTree.hh>
#include <vector>
#include "ConditionalLexicon.hh"

#include <iostream>

#include <Translation/History.hh>
#include <Translation/SRI/Ngram.h>
#include <Translation/SRI/Vocab.h>

#include <ext/hash_map>

namespace Translation {

using __gnu_cxx::hash;
using __gnu_cxx::hash_map;

typedef pair<Fsa::LabelId, Fsa::LabelId> LabelIdPair;

struct hash_LabelIdPair {
    size_t operator()(const LabelIdPair& x) const {
        size_t             return_value = 23;
        hash<unsigned int> f;

        size_t hash_value = f(x.first);
        return_value ^= hash_value;
        return_value += (hash_value << (return_value & 7));
        return_value ^= ((x.first) << ((return_value >> 3) & 15));
        hash_value = f(x.second);
        return_value ^= hash_value;
        return_value += (hash_value << (return_value & 7));
        return_value ^= ((x.second) << ((return_value >> 3) & 15));

        return return_value;
    }
};

struct equal_LabelIdPair {
    size_t operator()(const LabelIdPair& x, const LabelIdPair& y) const {
        return (x.first == y.first && x.second == y.second);
    }
};

class ConditionalLexiconSri : public ConditionalLexicon {
public:
    ConditionalLexiconSri(const Core::Configuration& config)
            : ConditionalLexicon(config),
              lexiconFilename_(paramFilename_(config)),
              sriTupleVocabulary_(),
              srilm_(sriTupleVocabulary_),
              userLmOrder_(paramLmOrder_(config)),
              vocabMap() {
        std::cerr << "lexicon filename: " << lexiconFilename_ << std::endl;
        std::cerr << "reading ..." << std::endl;
        this->read();
        floor_ = 1e-99;
        VocabIndexHistory::setSentinel(Vocab_None);
        //! \todo sentence begin symbol is hardcoded
        fsaSentenceBeginSymbol_ = tokens_->addSymbol("<s>");
        VocabIndexHistory::setHashParameters(51283, 3051491, 32);
    }

    ConditionalLexiconSri(const Core::Configuration& config, Fsa::ConstAlphabetRef alphabet)
            : ConditionalLexicon(config, alphabet),
              lexiconFilename_(paramFilename_(config)),
              sriTupleVocabulary_(),
              srilm_(sriTupleVocabulary_),
              userLmOrder_(paramLmOrder_(config)),
              vocabMap() {}
    //! this is deprecated and should just exist as long as i am migrating to the new lexicon
    virtual Translation::Cost getProb(const size_t index, const std::vector<std::string>& key) const {
        std::cerr << __FUNCTION__ << " to be implemented " << std::endl;
        return 0;
    }

    virtual Translation::Cost getCost(const size_t index, const std::vector<Fsa::LabelId>& key) const {
        VocabIndex        tupleIndex(mapToTupleIndex_(key[0], key[1]));
        VocabIndexHistory tupleHistory(lmOrder_ - 1);

        if (key.size() > 2) {
            std::vector<VocabIndex> sriKey;
            size_t                  keyIndex(2);
            while (keyIndex < key.size()) {
                if (key[keyIndex] == fsaSentenceBeginSymbol_ || key[keyIndex + 1] == fsaSentenceBeginSymbol_) {
                    sriKey.push_back(sentenceBeginIndex_);
                    break;
                }
                else {
                    sriKey.push_back(mapToTupleIndex_(key[keyIndex], key[keyIndex + 1]));
                }
                keyIndex += 2;
            }
            tupleHistory.expand(sriKey);
        }
        else {
            tupleHistory.expandEmpty();
        }

        return -srilm_.wordProb(tupleIndex, tupleHistory.rbegin());
    }

    virtual Translation::Cost getReverseCost(const size_t index, const std::vector<Fsa::LabelId>& key) const {
        std::cerr << __FUNCTION__ << " to be implemented " << std::endl;
        return 99;
    }

    //! get probability of a lexicon entry or floor if it does not exist
    virtual Translation::Cost getProb(const size_t index, const std::vector<Fsa::LabelId>& key) const {
        return ::pow(10, getCost(index, key));
    }

    //! add value to existing count/prob or create new if it does not exist
    virtual void addValue(const size_t index, const std::vector<Fsa::LabelId>& key, Translation::Cost value) {
        std::cerr << __FUNCTION__ << " to be implemented " << std::endl;
    }

    //! add value to existing count/prob or create new if it does not exist
    virtual void addValue(const size_t index, const std::vector<std::string>& key, Translation::Cost value) {
        std::cerr << __FUNCTION__ << " to be implemented " << std::endl;
    }

    //! set value of the given entry (overwrite if it exists, create if it doesnt)
    virtual void setValue(const size_t index, const std::vector<Fsa::LabelId>& key, Translation::Cost value) {
        std::cerr << __FUNCTION__ << " to be implemented " << std::endl;
    }

    //! set value of the given entry (overwrite if it exists, create if it doesnt)
    virtual void setValue(const size_t index, const std::vector<std::string>& key, Translation::Cost value) {
        std::cerr << __FUNCTION__ << " to be implemented " << std::endl;
    }

    //! write lexicon to stream
    virtual void write(std::ostream&) {};

    //! normalize
    virtual void normalize(int order) {};

private:
    VocabIndex mapToTupleIndexString_(Fsa::LabelId s, Fsa::LabelId t) const {
        std::string tupleSymbol = tokens_->symbol(s) + "|" + tokens_->symbol(t);
        return sriTupleVocabulary_.getIndex(VocabString(tupleSymbol.c_str()));
    }

    VocabIndex mapToTupleIndex_(Fsa::LabelId s, Fsa::LabelId t) const {
        std::string tupleSymbol = tokens_->symbol(s) + "|" + tokens_->symbol(t);

        LabelIdPair                  biSymbol(s, t);
        VocabMapType::const_iterator i = vocabMap.find(biSymbol);
        if (i != vocabMap.end()) {
            return i->second;
        }
        else {
            return unknownIndex_;
        }
    }

    typedef Translation::SimplePrefixTree<Fsa::LabelId, Translation::Cost> Lexicon;
    typedef Core::Ref<Lexicon>                                             LexiconRef;

    typedef MallocOptimizedHistory<VocabIndex> VocabIndexHistory;

    //! holds lexica for the different types of transitions (if neccessary)
    std::vector<LexiconRef> lexica_;

    //! floor probability for values that are not in the lexicon
    Translation::Cost floor_;

    //! parameter giving the filename to read from
    static Core::ParameterString paramFilename_;

    //! filename as a string. read from the correspondinf parameter
    const std::string lexiconFilename_;

    //! paramter for overriding the language model order given in the file
    static Core::ParameterInt paramLmOrder_;

    mutable Vocab sriTupleVocabulary_;  // Mutable in order to avoid a compiler warning.
    mutable Ngram srilm_;               // Mutable in order to avoid a compiler warning. It is however possible that the sri toolkit does change the internal state.
    VocabIndex    unknownIndex_;
    VocabIndex    sentenceBeginIndex_;
    VocabIndex    sentenceEndIndex_;
    VocabIndex*   history_;

    Fsa::LabelId fsaSentenceBeginSymbol_;

    size_t lmOrder_;
    size_t userLmOrder_;

    typedef hash_map<const LabelIdPair, VocabIndex, hash_LabelIdPair, equal_LabelIdPair> VocabMapType;
    VocabMapType                                                                         vocabMap;

public:
    //! read a lexicon from a file stream
    virtual void read(std::istream&) {};

    //! read a lexicon from a file given as component parameter
    virtual void read();
};
}  // namespace Translation
#endif
