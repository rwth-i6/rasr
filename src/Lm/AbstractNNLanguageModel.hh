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
#ifndef _LM_ABSTRACT_NN_LANGUAGE_MODEL_HH
#define _LM_ABSTRACT_NN_LANGUAGE_MODEL_HH

#include "LanguageModel.hh"
#include "NNHistoryManager.hh"

namespace Lm {

struct NNCacheWithStats : public NNCacheBase {
    mutable std::vector<bool> output_used;
};

class AbstractNNLanguageModel : public LanguageModel {
public:
    typedef LanguageModel Precursor;

    static Core::ParameterBool   paramCollectStatistics;
    static Core::ParameterString paramVocabularyFile;
    static Core::ParameterString paramVocabUnknownWord;

    AbstractNNLanguageModel(Core::Configuration const& c, Bliss::LexiconRef l);
    virtual ~AbstractNNLanguageModel();

protected:
    bool        collect_statistics_;
    std::string vocab_file_;
    std::string unknown_word_;

    Bliss::LexiconRef   lexicon_;
    size_t              num_outputs_;
    std::vector<size_t> lexicon_mapping_;
    std::vector<u32>    usage_histogram_;

    void loadVocabulary();
    void useOutput(NNCacheWithStats const& cache, size_t idx) const;
    void onRelease(HistoryHandle handle);
    void logStatistics() const;
};

// inline implementations
inline void AbstractNNLanguageModel::useOutput(NNCacheWithStats const& cache, size_t idx) const {
    if (collect_statistics_) {
        if (cache.output_used.empty()) {
            cache.output_used.resize(num_outputs_);
        }
        cache.output_used[idx] = true;
    }
}

} // namespace Lm

#endif // _LM_ABSTRACT_NN_LANGUAGE_MODEL_HH

