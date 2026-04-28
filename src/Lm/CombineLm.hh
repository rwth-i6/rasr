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
    virtual ~CombineLanguageModel();

    virtual Lm::Score sentenceBeginScore() const;
    virtual void      getDependencies(Core::DependencySet& dependencies) const;

    virtual History     startHistory() const;
    virtual History     extendedHistory(History const& history, Token w) const;
    virtual History     reducedHistory(History const& history, u32 limit) const;
    virtual History     reduceHistoryByN(History const&, u32 n) const;
    virtual std::string formatHistory(const History&) const;
    virtual Score       score(const History& history, Token w) const;
    virtual Score       sentenceEndScore(const History& history) const;

    virtual void getBatch(const History& h, const CompiledBatchRequest* cbr, std::vector<f32>& result) const;
    virtual void cacheBatch(const History& h, const CompiledBatchRequest* cbr, u32 size) const;

    virtual bool              fixedHistory(s32 limit) const;
    virtual bool              isSparse(const History& h) const;
    virtual HistorySuccessors getHistorySuccessors(const History& h) const;
    virtual Score             getBackOffScore(const History& h) const;

    virtual Core::Ref<const LanguageModel> lookaheadLanguageModel() const;
    virtual Core::Ref<const LanguageModel> recombinationLanguageModel() const;

    virtual void setSegment(Bliss::SpeechSegment const* s);

    virtual void startFrame(Search::TimeframeIndex time) const;
    virtual void setInfo(History const& hist, SearchSpaceInformation const& info) const;

protected:
    typedef std::unordered_map<Bliss::Token::Id, Score> TokenScoreMap;

    template<bool linear>
    HistorySuccessors getCombinedHistorySuccessors(const History& h) const;

    // also support partial LMs combined scoring
    template<bool linear>
    Score score_(const History& h, Token w, const std::vector<u32>& lmIds) const;

    template<bool linear>
    void cacheBatch_(const History& h, const CompiledBatchRequest* cbr, u32 size) const;

    bool matchCacheHistory(const History& h) const;

private:
    std::vector<Core::Ref<ScaledLanguageModel>>       lms_;
    std::vector<Core::Ref<const LanguageModel>>       unscaled_lms_;
    std::vector<SearchSpaceAwareLanguageModel const*> ssa_lms_;
    std::vector<Score>                                skip_thresholds_;

    bool linear_combination_;
    int  lookahead_lm_;
    int  recombination_lm_;

    std::vector<u32> lmIds_;

    // cached scores for partial sparse lookahead (so far only single history cache: unigram)
    mutable std::vector<History> cacheHist_;
    mutable std::vector<Score>   cacheScores_;
    // lexicon tokenId to requests mapping
    mutable std::vector<std::vector<u32>> token2Requests_;

    mutable u32 staticRequestSize_;

    std::vector<History>          staticCacheHist_;
    std::vector<Score>            staticCacheScores_;
    std::vector<std::vector<u32>> staticToken2Requests_;
};

}  // namespace Lm

#endif /* _LM_COMBINE_LM_HH */
