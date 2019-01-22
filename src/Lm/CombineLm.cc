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
#include "CombineLm.hh"

#include <Core/MurmurHash.hh>
#include <Math/Utilities.hh>
#include "Module.hh"

namespace {
    class CombineHistoryManager : public Lm::HistoryManager {
    public:
        CombineHistoryManager(size_t numLms) : Lm::HistoryManager(), numLms_(numLms) {
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
            Lm::History const* hist = reinterpret_cast<Lm::History const*>(handle);
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
                if (not (lhs_hist[i] == rhs_hist[i])) {
                    return false;
                }
            }
            return true;
        }

        virtual std::string format(Lm::HistoryHandle handle) const {
            Lm::History const* hist = reinterpret_cast<Lm::History const*>(handle);
            std::stringstream ss;
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
}

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
                                          : Core::Component(c), CombineLanguageModel::Precursor(c, l),
                                            lms_(), unscaled_lms_(), linear_combination_(paramLinearCombination(c)),
                                            lookahead_lm_(paramLookaheadLM(config)), recombination_lm_(paramRecombinationLM(config)) {
    size_t num_lms = paramNumLms(c);
    for (size_t i = 0ul; i < num_lms; i++) {
        Core::Configuration sub_config = select(std::string("lm-") + std::to_string(i+1));
        lms_.push_back(Module::instance().createScaledLanguageModel(sub_config, l));
        unscaled_lms_.push_back(lms_.back()->unscaled());
        ssa_lms_.push_back(dynamic_cast<SearchSpaceAwareLanguageModel const*>(unscaled_lms_.back().get()));
        skip_thresholds_.push_back(paramSkipThreshold(sub_config));
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
    History* new_hist = new History[lms_.size()];
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
    History* new_hist = new History[lms_.size()];
    for (size_t i = 0ul; i < lms_.size(); i++) {
        new_hist[i] = lms_[i]->reducedHistory(prev_hist[i], limit);
    }
    History h = this->history(new_hist);
    delete[] new_hist;
    return h;
}

Score CombineLanguageModel::score(History const& history, Token w) const {
    require(history.isManagedBy(historyManager_));
    History const* hist = reinterpret_cast<History const*>(history.handle());
    Score prev_score = 0.0;
    bool  override_score = false;
    if (linear_combination_) {
        Score s(std::numeric_limits<Score>::infinity());
        for (size_t i = 0ul; i < lms_.size(); i++) {
            Score raw_score = 0.0;
            if (not override_score) {
                raw_score = unscaled_lms_[i]->score(hist[i], w);
                prev_score = raw_score;
                override_score |= raw_score >= skip_thresholds_[i];
            }
            else {
                raw_score = prev_score;
                if (unscaled_lms_[i]->scoreCached(history, w)) {
                    raw_score = unscaled_lms_[i]->score(hist[i], w);
                }
            }
            s = Math::scoreSum(s, raw_score - std::log(lms_[i]->scale()));
        }
        return s;
    }
    else {
        Score s(0.0);
        for (size_t i = 0ul; i < lms_.size(); i++) {
            Score raw_score = unscaled_lms_[i]->score(hist[i], w);
            if (not override_score) {
                raw_score = unscaled_lms_[i]->score(hist[i], w);
                prev_score = raw_score;
                override_score |= raw_score >= skip_thresholds_[i];
            }
            else {
                raw_score = prev_score;
                if (unscaled_lms_[i]->scoreCached(history, w)) {
                    raw_score = unscaled_lms_[i]->score(hist[i], w);
                }
            }
            s += raw_score * lms_[i]->scale();
        }
        return s;
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

Core::Ref<const LanguageModel> CombineLanguageModel::lookaheadLanguageModel() const {
    if (lookahead_lm_ > 0) {
        require_le(static_cast<unsigned>(lookahead_lm_), unscaled_lms_.size());
        return unscaled_lms_[lookahead_lm_-1];
    }
    return Core::Ref<LanguageModel>();
}

Core::Ref<const LanguageModel> CombineLanguageModel::recombinationLanguageModel() const {
    if (recombination_lm_ > 0) {
        require_le(static_cast<unsigned>(recombination_lm_), unscaled_lms_.size());
        return unscaled_lms_[recombination_lm_-1];
    }
    return Core::Ref<LanguageModel>();
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

} // namespace Lm
