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
#include "CombineLm.hh"

#include <Core/MurmurHash.hh>
#include <Math/Utilities.hh>
#include "Module.hh"

namespace {
class CombineHistoryManager : public Lm::HistoryManager {
public:
    CombineHistoryManager(size_t numLms)
            : Lm::HistoryManager(),
              numLms_(numLms) {
    }
    virtual ~CombineHistoryManager() = default;

    virtual Lm::HistoryHandle acquire(Lm::HistoryHandle handle) {
        Lm::History const* prev_hist = reinterpret_cast<Lm::History const*>(handle);
        Lm::History*       new_hist  = new Lm::History[numLms_];
        for (size_t i = 0ul; i < numLms_; i++) {
            new_hist[i] = prev_hist[i];
        }
        return reinterpret_cast<Lm::HistoryHandle>(new_hist);
    }

    virtual void release(Lm::HistoryHandle handle) {
        Lm::History const* hist = reinterpret_cast<Lm::History const*>(handle);
        delete[] hist;
    }

    virtual Lm::HistoryHash hashKey(Lm::HistoryHandle handle) const {
        Lm::History const*           hist = reinterpret_cast<Lm::History const*>(handle);
        std::vector<Lm::HistoryHash> hashes(numLms_);
        for (size_t i = 0ul; i < numLms_; i++) {
            hashes[i] = hist[i].hashKey();
        }
        return Core::MurmurHash3_x64_64(reinterpret_cast<void const*>(hashes.data()), hashes.size() * sizeof(Lm::HistoryHash), 0x305ff0a7);
    }

    virtual bool isEquivalent(Lm::HistoryHandle lhs, Lm::HistoryHandle rhs) const {
        Lm::History const* lhs_hist = reinterpret_cast<Lm::History const*>(lhs);
        Lm::History const* rhs_hist = reinterpret_cast<Lm::History const*>(rhs);
        for (size_t i = 0ul; i < numLms_; i++) {
            if (not(lhs_hist[i] == rhs_hist[i])) {
                return false;
            }
        }
        return true;
    }

    virtual std::string format(Lm::HistoryHandle handle) const {
        Lm::History const* hist = reinterpret_cast<Lm::History const*>(handle);
        std::stringstream  ss;
        ss << "CombinedHistory<";
        for (size_t i = 0ul; i < numLms_; i++) {
            ss << " h" << i << ": " << hist[i].format();
        }
        ss << " >";
        return ss.str();
    }

private:
    size_t numLms_;
};
}  // namespace

namespace Lm {

Core::ParameterInt CombineLanguageModel::paramNumLms(
        "num-lms", "number of language models to combine", 1, 1);
Core::ParameterBool CombineLanguageModel::paramLinearCombination(
        "linear-combination", "if true linear combination instead of log-linear combination is used", false);
Core::ParameterInt CombineLanguageModel::paramLookaheadLM(
        "lookahead-lm", "index of the sub-lm to be used for lookahead, use 0 for the combine-lm itself", 0, 0);
Core::ParameterInt CombineLanguageModel::paramRecombinationLM(
        "recombination-lm", "index of the sub-lm to be used for recombination, use 0 for the combine-lm itself", 0, 0);
Core::ParameterFloat CombineLanguageModel::paramSkipThreshold(
        "skip-threshold", "if this LM's (unscaled) score is greater than this threshold successive LMs are not evaluated", std::numeric_limits<Score>::max());

CombineLanguageModel::CombineLanguageModel(Core::Configuration const& c, Bliss::LexiconRef l)
        : Core::Component(c),
          CombineLanguageModel::Precursor(c, l),
          lms_(),
          unscaled_lms_(),
          linear_combination_(paramLinearCombination(c)),
          lookahead_lm_(paramLookaheadLM(config)),
          recombination_lm_(paramRecombinationLM(config)),
          staticRequestSize_(0) {
    size_t num_lms = paramNumLms(c);
    for (size_t i = 0ul; i < num_lms; i++) {
        Core::Configuration sub_config = select(std::string("lm-") + std::to_string(i + 1));
        lms_.push_back(Module::instance().createScaledLanguageModel(sub_config, l));
        unscaled_lms_.push_back(lms_.back()->unscaled());
        ssa_lms_.push_back(dynamic_cast<SearchSpaceAwareLanguageModel const*>(unscaled_lms_.back().get()));
        skip_thresholds_.push_back(paramSkipThreshold(sub_config));
        lmIds_.push_back(i);
    }
    historyManager_ = new CombineHistoryManager(num_lms);
}

CombineLanguageModel::CombineLanguageModel(Core::Configuration const& c, Bliss::LexiconRef l, std::vector<Core::Ref<ScaledLanguageModel>> const& subLms)
        : Core::Component(c), CombineLanguageModel::Precursor(c, l), lms_(), unscaled_lms_(), linear_combination_(paramLinearCombination(c)), lookahead_lm_(paramLookaheadLM(config)), recombination_lm_(paramRecombinationLM(config)) {
    size_t num_lms = subLms.size();
    for (auto const& subLm : subLms) {
        lms_.push_back(subLm);
        unscaled_lms_.push_back(lms_.back()->unscaled());
        ssa_lms_.push_back(dynamic_cast<SearchSpaceAwareLanguageModel const*>(unscaled_lms_.back().get()));
        skip_thresholds_.push_back(paramSkipThreshold(subLm->getConfiguration()));
    }
    historyManager_ = new CombineHistoryManager(num_lms);
}

CombineLanguageModel::~CombineLanguageModel() {
}

Score CombineLanguageModel::sentenceBeginScore() const {
    if (linear_combination_) {
        Score s(std::numeric_limits<Score>::infinity());
        for (size_t i = 0ul; i < lms_.size(); i++) {
            s = Math::scoreSum(s, unscaled_lms_[i]->sentenceBeginScore() - std::log(lms_[i]->scale()));
        }
        return s;
    }
    else {
        Score s(0.0);
        for (size_t i = 0ul; i < lms_.size(); i++) {
            s += lms_[i]->sentenceBeginScore();
        }
        return s;
    }
}

void CombineLanguageModel::getDependencies(Core::DependencySet& dependencies) const {
    for (size_t i = 0ul; i < lms_.size(); i++) {
        lms_[i]->getDependencies(dependencies);
    }
}

History CombineLanguageModel::startHistory() const {
    History* hist = new History[lms_.size()];
    for (size_t i = 0ul; i < lms_.size(); i++) {
        hist[i] = lms_[i]->startHistory();
    }
    History h = this->history(hist);
    delete[] hist;
    return h;
}

History CombineLanguageModel::extendedHistory(History const& history, Token w) const {
    require(history.isManagedBy(historyManager_));
    History const* prev_hist = reinterpret_cast<History const*>(history.handle());
    History*       new_hist  = new History[lms_.size()];
    for (size_t i = 0ul; i < lms_.size(); i++) {
        new_hist[i] = lms_[i]->extendedHistory(prev_hist[i], w);
    }
    History h = this->history(new_hist);
    delete[] new_hist;
    return h;
}

History CombineLanguageModel::reducedHistory(History const& history, u32 limit) const {
    require(history.isManagedBy(historyManager_));
    History const* prev_hist = reinterpret_cast<History const*>(history.handle());
    History*       new_hist  = new History[lms_.size()];
    for (size_t i = 0ul; i < lms_.size(); i++) {
        new_hist[i] = lms_[i]->reducedHistory(prev_hist[i], limit);
    }
    History h = this->history(new_hist);
    delete[] new_hist;
    return h;
}

History CombineLanguageModel::reduceHistoryByN(History const& history, u32 n) const {
    require(history.isManagedBy(historyManager_));
    History const* prev_hist = reinterpret_cast<History const*>(history.handle());
    History*       new_hist  = new History[lms_.size()];
    for (size_t i = 0ul; i < lms_.size(); i++) {
        new_hist[i] = lms_[i]->reduceHistoryByN(prev_hist[i], n);
    }
    History h = this->history(new_hist);
    delete[] new_hist;
    return h;
}

std::string CombineLanguageModel::formatHistory(const History& h) const {
    const History* hist = reinterpret_cast<const History*>(h.handle());

    std::stringstream ss;
    ss << "CombinedHistory<";
    for (size_t i = 0ul; i < lms_.size(); i++) {
        ss << " h" << i << ": " << unscaled_lms_[i]->formatHistory(hist[i]);
    }
    ss << " >";
    return ss.str();
}

Score CombineLanguageModel::score(const History& history, Token w) const {
    require(history.isManagedBy(historyManager_));
    if (linear_combination_) {
        return score_<true>(history, w, lmIds_);
    }
    else {
        return score_<false>(history, w, lmIds_);
    }
}

Score CombineLanguageModel::sentenceEndScore(const History& history) const {
    require(history.isManagedBy(historyManager_));
    History const* hist = reinterpret_cast<History const*>(history.handle());
    if (linear_combination_) {
        Score s(std::numeric_limits<Score>::infinity());
        for (size_t i = 0ul; i < lms_.size(); i++) {
            s = Math::scoreSum(s, unscaled_lms_[i]->sentenceEndScore(hist[i]) - std::log(lms_[i]->scale()));
        }
        return s;
    }
    else {
        Score s(0.0);
        for (size_t i = 0ul; i < lms_.size(); i++) {
            s += lms_[i]->sentenceEndScore(hist[i]);
        }
        return s;
    }
}

void CombineLanguageModel::getBatch(const History& h, const CompiledBatchRequest* cbr, std::vector<f32>& result) const {
    if (cacheHist_.empty() || cacheScores_.empty() || !matchCacheHistory(h)) {
        Precursor::getBatch(h, cbr, result);
        return;
    }

    // apply update on partial sparse LMs' tokens only, others are cached and operated in same scheme
    require(h.isManagedBy(historyManager_));
    const History*                 hist    = reinterpret_cast<const History*>(h.handle());
    const NonCompiledBatchRequest* ncbr    = required_cast(const NonCompiledBatchRequest*, cbr);
    const BatchRequest&            request = ncbr->request;

    std::unordered_set<u32> tokens;
    Score                   backoff = 0;
    if (linear_combination_) {
        backoff = std::numeric_limits<Score>::infinity();
    }
    for (u32 i = 0; i < lms_.size(); ++i) {
        if (cacheHist_[i].isValid()) {
            continue;
        }
        HistorySuccessors subSuccessors = unscaled_lms_[i]->getHistorySuccessors(hist[i]);
        for (const WordScore& ws : subSuccessors) {
            tokens.insert(ws.token());
        }
        if (linear_combination_) {
            backoff = Math::scoreSum(backoff, subSuccessors.backOffScore - std::log(lms_[i]->scale()));
        }
        else {
            backoff += subSuccessors.backOffScore * lms_[i]->scale();
        }
    }

    // non-existing tokens' scores based on cached scores and backoff
    verify(result.size() == cacheScores_.size());
    if (linear_combination_) {
        result = cacheScores_;  // assume 0-prob. here
    }
    else {
        std::transform(cacheScores_.begin(), cacheScores_.end(), result.begin(), std::bind(std::plus<f32>(), std::placeholders::_1, backoff * ncbr->scale()));
    }

    // full combined score for these existing tokens (Note: further simplified to first token only)
    for (std::unordered_set<u32>::const_iterator tokId = tokens.begin(); tokId != tokens.end(); ++tokId) {
        std::vector<u32>& rqsts    = token2Requests_.at(*tokId);
        Score             tokScore = score(h, request[rqsts.front()].tokens[0]) * ncbr->scale();
        for (std::vector<u32>::const_iterator reqId = rqsts.begin(); reqId != rqsts.end(); ++reqId) {
            const Request& r   = request[*reqId];
            Score          sco = tokScore + r.offset;
            if (result[r.target] > sco) {
                result[r.target] = sco;
            }
        }
    }
}

void CombineLanguageModel::cacheBatch(const History& h, const CompiledBatchRequest* cbr, u32 size) const {
    verify(h.isValid());
    if (linear_combination_) {
        cacheBatch_<true>(h, cbr, size);
    }
    else {
        cacheBatch_<false>(h, cbr, size);
    }
}

bool CombineLanguageModel::fixedHistory(s32 limit) const {
    for (u32 i = 0; i < lms_.size(); ++i) {
        if (!unscaled_lms_[i]->fixedHistory(limit)) {
            return false;
        }
    }
    return true;
}

bool CombineLanguageModel::isSparse(const History& h) const {
    // combineLM itself is used for lookahead: only true if all subLMs are sparse
    if (!h.isValid()) {
        for (u32 i = 0; i < lms_.size(); ++i) {
            if (!lms_[i]->isSparse(h)) {
                return false;
            }
        }
        return true;
    }

    require(h.isManagedBy(historyManager_));
    const History* hist = reinterpret_cast<const History*>(h.handle());
    for (u32 i = 0; i < lms_.size(); ++i) {
        if (!lms_[i]->isSparse(hist[i])) {
            return false;
        }
    }
    return true;
}

HistorySuccessors CombineLanguageModel::getHistorySuccessors(const History& h) const {
    if (linear_combination_) {
        return getCombinedHistorySuccessors<true>(h);
    }
    else {
        return getCombinedHistorySuccessors<false>(h);
    }
}

Score CombineLanguageModel::getBackOffScore(const History& h) const {
    require(h.isManagedBy(historyManager_));
    const History* hist    = reinterpret_cast<const History*>(h.handle());
    Score          backoff = linear_combination_ ? std::numeric_limits<Score>::infinity() : 0;

    for (u32 i = 0; i < lms_.size(); ++i) {
        if (linear_combination_) {
            backoff = Math::scoreSum(backoff, unscaled_lms_[i]->getBackOffScore(hist[i]) - std::log(lms_[i]->scale()));
        }
        else {
            backoff += unscaled_lms_[i]->getBackOffScore(hist[i]) * lms_[i]->scale();
        }
    }
    return backoff;
}

Core::Ref<const LanguageModel> CombineLanguageModel::lookaheadLanguageModel() const {
    if (lookahead_lm_ > 0) {
        require_le(static_cast<unsigned>(lookahead_lm_), unscaled_lms_.size());
        return unscaled_lms_[lookahead_lm_ - 1];
    }
    return Core::Ref<LanguageModel>();
}

Core::Ref<const LanguageModel> CombineLanguageModel::recombinationLanguageModel() const {
    if (recombination_lm_ > 0) {
        require_le(static_cast<unsigned>(recombination_lm_), unscaled_lms_.size());
        return unscaled_lms_[recombination_lm_ - 1];
    }
    return Core::Ref<LanguageModel>();
}

bool CombineLanguageModel::setSegment(Bliss::SpeechSegment const* s) {
    bool changed = false;
    for (size_t i = 0ul; i < lms_.size(); i++) {
        changed |= lms_[i]->setSegment(s);
    }
    return changed;
}

void CombineLanguageModel::startFrame(Search::TimeframeIndex time) const {
    for (auto lm : ssa_lms_) {
        if (lm) {
            lm->startFrame(time);
        }
    }
}

void CombineLanguageModel::setInfo(History const& hist, SearchSpaceInformation const& info) const {
    History const* comb_hist = reinterpret_cast<History const*>(hist.handle());
    for (size_t i = 0ul; i < ssa_lms_.size(); i++) {
        if (ssa_lms_[i]) {
            ssa_lms_[i]->setInfo(comb_hist[i], info);
        }
    }
}

// combine sparse scores with closest behavior as actual scoring
// one HistorySuccessors for each subLM where tokens are not requested to be the same
// combined HistorySuccessors is a super-set of all sub HistorySuccessors with score combined
// in the same way as scoring (use backoff score if a token does not exist)
// TODO test efficiency and maybe improve
template<bool linear>
HistorySuccessors CombineLanguageModel::getCombinedHistorySuccessors(const History& h) const {
    require(h.isManagedBy(historyManager_));
    const History* hist = reinterpret_cast<const History*>(h.handle());

    TokenScoreMap              combineSuccessors;
    std::set<Bliss::Token::Id> combineTokens;
    Score                      backoff = 0;
    if (linear) {
        backoff = std::numeric_limits<Score>::infinity();
    }

    for (u32 i = 0; i < lms_.size(); ++i) {
        HistorySuccessors          subSuccessors = unscaled_lms_[i]->getHistorySuccessors(hist[i]);
        std::set<Bliss::Token::Id> subTokens;
        for (const WordScore& ws : subSuccessors) {
            subTokens.insert(ws.token());
            TokenScoreMap::iterator iter = combineSuccessors.insert(std::make_pair(ws.token(), backoff)).first;
            if (linear) {
                iter->second = Math::scoreSum(iter->second, ws.score() - std::log(lms_[i]->scale()));
            }
            else {
                iter->second += ws.score() * lms_[i]->scale();
            }
        }

        if (combineTokens.empty()) {
            combineTokens.swap(subTokens);
        }
        else if (subTokens.empty()) {
            for (TokenScoreMap::iterator iter = combineSuccessors.begin(); iter != combineSuccessors.end(); ++iter) {
                if (linear) {
                    iter->second = Math::scoreSum(iter->second, subSuccessors.backOffScore - std::log(lms_[i]->scale()));
                }
                else {
                    iter->second += subSuccessors.backOffScore * lms_[i]->scale();
                }
            }
        }
        else {
            std::set<Bliss::Token::Id> missTokens;
            std::set_difference(combineTokens.begin(), combineTokens.end(), subTokens.begin(), subTokens.end(), std::inserter(missTokens, missTokens.begin()));
            for (std::set<Bliss::Token::Id>::const_iterator it = missTokens.begin(); it != missTokens.end(); ++it) {
                Score& s = combineSuccessors[*it];
                if (linear) {
                    s = Math::scoreSum(s, subSuccessors.backOffScore - std::log(lms_[i]->scale()));
                }
                else {
                    s += subSuccessors.backOffScore * lms_[i]->scale();
                }
            }
            if (subTokens.size() > missTokens.size()) {
                subTokens.insert(missTokens.begin(), missTokens.end());
                combineTokens.swap(subTokens);
            }
            else {
                missTokens.insert(subTokens.begin(), subTokens.end());
                combineTokens.swap(missTokens);
            }
        }

        if (linear) {
            backoff = Math::scoreSum(backoff, subSuccessors.backOffScore - std::log(lms_[i]->scale()));
        }
        else {
            backoff += subSuccessors.backOffScore * lms_[i]->scale();
        }
    }

    HistorySuccessors res;
    res.backOffScore = backoff;
    res.reserve(combineSuccessors.size());
    for (TokenScoreMap::const_iterator iter = combineSuccessors.begin(); iter != combineSuccessors.end(); ++iter) {
        res.emplace_back(iter->first, iter->second);
    }
    return res;
}

template<bool linear>
Score CombineLanguageModel::score_(const History& history, Token w, const std::vector<u32>& lmIds) const {
    History const* hist = reinterpret_cast<History const*>(history.handle());

    Score prev_score     = 0.0;
    bool  override_score = false;
    Score comb_score     = linear ? std::numeric_limits<Score>::infinity() : 0.0;

    for (std::vector<u32>::const_iterator it = lmIds.begin(); it != lmIds.end(); ++it) {
        Score raw_score = 0.0;
        if (!override_score) {
            raw_score  = unscaled_lms_[*it]->score(hist[*it], w);
            prev_score = raw_score;
            override_score |= raw_score >= skip_thresholds_[*it];
        }
        else {
            raw_score = prev_score;
            if (unscaled_lms_[*it]->scoreCached(history, w)) {
                raw_score = unscaled_lms_[*it]->score(hist[*it], w);
            }
        }
        if (linear) {
            comb_score = Math::scoreSum(comb_score, raw_score - std::log(lms_[*it]->scale()));
        }
        else {
            comb_score += raw_score * lms_[*it]->scale();
        }
    }
    return comb_score;
}

template<bool linear>
void CombineLanguageModel::cacheBatch_(const History& h, const CompiledBatchRequest* cbr, u32 size) const {
    cacheHist_.clear();
    cacheScores_.clear();
    verify(matchCacheHistory(h));
    // partial non-sparse LMs to be cached
    std::vector<u32> cacheLmIds;
    for (u32 i = 0; i < lms_.size(); ++i) {
        if (cacheHist_[i].isValid()) {
            cacheLmIds.push_back(i);
        }
    }
    if (cacheLmIds.empty() || cacheLmIds.size() == lms_.size()) {
        cacheHist_.clear();
        return;
    }

    // cached LMs combined scoring + token to request mapping
    const NonCompiledBatchRequest* ncbr    = required_cast(const NonCompiledBatchRequest*, cbr);
    const BatchRequest&            request = ncbr->request;
    cacheScores_.resize(size, Core::Type<Score>::max);

    u32 startIdx = 0;
    if (token2Requests_.empty() && staticToken2Requests_.empty()) {
        staticRequestSize_ = request.size();
    }
    else if (!staticToken2Requests_.empty()) {
        verify(staticRequestSize_ > 0 && request.size() >= staticRequestSize_);
        token2Requests_ = staticToken2Requests_;
        startIdx        = staticRequestSize_;
    }
    token2Requests_.resize(lexicon()->nSyntacticTokens());

    for (u32 idx = 0; idx < request.size(); ++idx) {
        const Request& r   = request[idx];
        Score          sco = 0.0;
        if (r.tokens.length() >= 1) {
            // first token only: mostly should be just single mapping
            if (idx >= startIdx) {
                token2Requests_.at(r.tokens[0]->id()).push_back(idx);
            }
            sco += score_<linear>(h, r.tokens[0], cacheLmIds);
            if (r.tokens.length() > 1) {
                History hh = extendedHistory(h, r.tokens[0]);
                for (u32 ti = 1;; ++ti) {
                    Token st = r.tokens[ti];
                    sco += score_<linear>(hh, st, cacheLmIds);
                    if (ti + 1 >= r.tokens.length()) {
                        break;
                    }
                    hh = extendedHistory(hh, st);
                }
            }
        }
        sco *= ncbr->scale();
        sco += r.offset;
        if (cacheScores_[r.target] > sco) {
            cacheScores_[r.target] = sco;
        }
    }
}

bool CombineLanguageModel::matchCacheHistory(const History& h) const {
    const History* hist = reinterpret_cast<const History*>(h.handle());
    if (cacheHist_.empty()) {
        for (u32 i = 0; i < lms_.size(); ++i) {
            if (unscaled_lms_[i]->isSparse(hist[i])) {
                cacheHist_.emplace_back();
            }
            else {
                cacheHist_.emplace_back(hist[i]);
            }
        }
    }
    else {
        for (u32 i = 0; i < lms_.size(); ++i) {
            if (!unscaled_lms_[i]->isSparse(hist[i]) && !(hist[i] == cacheHist_[i])) {
                return false;
            }
        }
    }
    return true;
}

}  // namespace Lm
