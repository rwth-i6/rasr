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
#ifndef TRANSLATION_LEXICON
#define TRANSLATION_LEXICON

#include <Core/CompressedStream.hh>
#include <Core/Vector.hh>
#include <Fsa/Automaton.hh>

#include <ext/hash_map>
#include <map>
#include <vector>

using __gnu_cxx::hash;
using __gnu_cxx::hash_map;

namespace Fsa {

typedef Vector<LabelId> LabelIdVector;

struct hash_LabelIdVector {
    size_t operator()(const LabelIdVector* x) const {
        size_t             return_value = 1;
        hash<unsigned int> f;
        for (LabelIdVector::const_iterator i = x->begin(); i != x->end(); ++i) {
            size_t hash_value = f(*i);
            return_value ^= hash_value;
            return_value += (hash_value << (return_value & 7));
            return_value ^= ((*i) << ((return_value >> 3) & 15));
        }
        return return_value;
    }
};

struct equal_LabelIdVector {
    size_t operator()(const LabelIdVector* x, const LabelIdVector* y) const {
        bool return_value = 1;
        if (x->size() != y->size())
            return false;
        LabelIdVector::const_iterator i = x->begin();
        LabelIdVector::const_iterator j = y->begin();
        while (i != x->end() && return_value) {
            return_value *= (*i) == (*j);
            ++i;
            ++j;
        }
        return return_value;
    }
};

class TranslationLexicon {
private:
    typedef hash_map<const LabelIdVector*, f32, hash_LabelIdVector, equal_LabelIdVector> lexiconType;
    // ATTENTION: memory leak here
    lexiconType lexicon_;
    const f32   floor_;
    // no distinction between input and output tokens;
    // the user has to be aware of this
    StaticAlphabet *sourceTokens_, *targetTokens_;

public:
    TranslationLexicon(const std::string& file, f32 floor = 99)
            : floor_(floor),
              sourceTokens_(new StaticAlphabet()),
              targetTokens_(new StaticAlphabet()) {
        Core::CompressedInputStream i(file);
        std::cerr << "reading lexicon from file " << file << std::endl;
        u32 counter(0);

        while (i) {
            f32         prob;
            std::string sourceWord, targetWord;
            i >> prob >> sourceWord >> targetWord;
            counter++;

            LabelIdVector* lexEntry = new LabelIdVector;
            // ATTENTION: memory leak here
            lexEntry->push_back(sourceTokens_->addSymbol(sourceWord));
            lexEntry->push_back(targetTokens_->addSymbol(targetWord));

            lexicon_[lexEntry] = prob;

            if ((counter % 1000000) == 0) {
                std::cerr << "[" << counter << "]" << std::endl;
            }
            else if ((counter % 100000) == 0) {
                std::cerr << ".";
                std::cerr.flush();
            }
        }
        std::cerr << "read " << counter << " entries" << std::endl;
    }

    f32 getProb(const LabelIdVector& x) const {
        lexiconType::const_iterator fi = lexicon_.find(&x);
        if (fi != lexicon_.end())
            return fi->second;
        else
            return floor_;
    }

    f32 getProb(const std::vector<std::string>& x) const {
        LabelIdVector y;
        y.push_back(sourceTokens_->addSymbol(x[0]));
        y.push_back(targetTokens_->addSymbol(x[1]));
        lexiconType::const_iterator fi = lexicon_.find(&y);
        if (fi != lexicon_.end())
            return fi->second;
        else
            return floor_;
    }
};  // namespace Fsa
}  // namespace Fsa

#endif
