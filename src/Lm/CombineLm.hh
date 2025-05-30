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
#ifndef _LM_COMBINE_LM_HH
#define _LM_COMBINE_LM_HH

#include <memory>
#include <vector>

#include "HistoryManager.hh"
#include "LanguageModel.hh"
#include "ScaledLanguageModel.hh"
#include "SearchSpaceAwareLanguageModel.hh"

namespace Lm {

class CombineLanguageModel : public LanguageModel, public SearchSpaceAwareLanguageModel {
public:
    typedef LanguageModel Precursor;

    static Core::ParameterInt   paramNumLms;
    static Core::ParameterBool  paramLinearCombination;
    static Core::ParameterInt   paramLookaheadLM;
    static Core::ParameterInt   paramRecombinationLM;
    static Core::ParameterFloat paramSkipThreshold;

    CombineLanguageModel(Core::Configuration const& c, Bliss::LexiconRef l);
    CombineLanguageModel(Core::Configuration const& c, Bliss::LexiconRef l, std::vector<Core::Ref<ScaledLanguageModel>> const& subLms);

    virtual ~CombineLanguageModel();

    virtual Lm::Score sentenceBeginScore() const;
    virtual void      getDependencies(Core::DependencySet& dependencies) const;

    virtual History                        startHistory() const;
    virtual History                        extendedHistory(History const& history, Token w) const;
    virtual History                        reducedHistory(History const& history, u32 limit) const;
    virtual Score                          score(History const& history, Token w) const;
    virtual Score                          sentenceEndScore(const History& history) const;
    virtual Core::Ref<const LanguageModel> lookaheadLanguageModel() const;
    virtual Core::Ref<const LanguageModel> recombinationLanguageModel() const;
    virtual bool                           setSegment(Bliss::SpeechSegment const* s);

    virtual void startFrame(Search::TimeframeIndex time) const;
    virtual void setInfo(History const& hist, SearchSpaceInformation const& info) const;

private:
    std::vector<Core::Ref<ScaledLanguageModel>>       lms_;
    std::vector<Core::Ref<const LanguageModel>>       unscaled_lms_;
    std::vector<SearchSpaceAwareLanguageModel const*> ssa_lms_;
    std::vector<Score>                                skip_thresholds_;

    bool linear_combination_;
    int  lookahead_lm_;
    int  recombination_lm_;
};

}  // namespace Lm

#endif /* _LM_COMBINE_LM_HH */
