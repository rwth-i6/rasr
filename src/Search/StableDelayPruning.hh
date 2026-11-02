/** Copyright 2026 RWTH Aachen University. All rights reserved.
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

#ifndef SEARCH_STABLE_DELAY_PRUNING_HH
#define SEARCH_STABLE_DELAY_PRUNING_HH

#include <functional>
#include <vector>

#include <Bliss/Lexicon.hh>
#include <Search/Traceback.hh>
#include <Search/Types.hh>

namespace Search {

/*
 * Maximum-stable-delay pruning, see section "Maximum-stable-delay pruning" in doc/search_v2.rst.
 * Only traces that aren't shared by all hypotheses are modified, so the stable prefix stays append-only.
 */
class StableDelayPruning {
public:
    struct Hypothesis {
        Core::Ref<LatticeTrace>* trace;        // May be replaced when splitting a pause
        bool                     traceIsOpen;  // `trace` is the current item, which may still be replaced by a copy with a later end time
        Bliss::Lemma const*      openPause;    // Pause lemma the hypothesis is currently in, nullptr if none
        Score                    score;
    };

    using IsPause = std::function<bool(Bliss::Lemma const*)>;

    StableDelayPruning(TimeframeIndex cutoff, IsPause isPause)
            : cutoff_(cutoff), isPause_(std::move(isPause)) {}

    // Sets `keep` per hypothesis. Returns false if there is no reference and the delay is exceeded.
    bool apply(std::vector<Hypothesis>& hyps, std::vector<bool>& keep) const {
        keep.assign(hyps.size(), false);

        Hypothesis const* best      = nullptr;
        Hypothesis const* reference = nullptr;
        for (auto const& hyp : hyps) {
            if (not best or hyp.score < best->score) {
                best = &hyp;
            }
            if ((not reference or hyp.score < reference->score) and (finished(hyp)->time >= cutoff_ or hyp.openPause)) {
                reference = &hyp;
            }
        }
        if (not reference) {
            if (best) {
                keepThrough(hyps, keep, finished(*best));
            }
            return false;
        }

        LatticeTrace* root = finished(*reference);
        if (root->time < cutoff_) {
            // The reference is in a pause that started before the cutoff
            splitPause(hyps, keep, root, reference->openPause->pronunciations().first);
            return true;
        }
        while (root->predecessor and root->predecessor->time >= cutoff_) {
            root = root->predecessor.get();
        }
        if (isPause(*root) and root->predecessor) {
            splitPause(hyps, keep, root->predecessor.get(), root->pronunciation);
        }
        else {
            keepThrough(hyps, keep, root);
        }
        return true;
    }

private:
    TimeframeIndex cutoff_;
    IsPause        isPause_;

    static LatticeTrace* finished(Hypothesis const& hyp) {
        return hyp.traceIsOpen ? (*hyp.trace)->predecessor.get() : hyp.trace->get();
    }

    bool isPause(LatticeTrace const& trace) const {
        return trace.pronunciation and isPause_(trace.pronunciation->lemma());
    }

    static void keepThrough(std::vector<Hypothesis> const& hyps, std::vector<bool>& keep, LatticeTrace const* root) {
        for (size_t i = 0ul; i < hyps.size(); ++i) {
            LatticeTrace const* curr = hyps[i].trace->get();
            while (curr and curr != root and curr->time > root->time) {
                curr = curr->predecessor.get();
            }
            keep[i] = curr == root;
        }
    }

    // Keeps the hypotheses that continue `start` with the pause `pron` until the cutoff and splits the pause at the cutoff
    void splitPause(std::vector<Hypothesis>& hyps, std::vector<bool>& keep, LatticeTrace* start, Bliss::LemmaPronunciation const* pron) const {
        Bliss::Lemma const* pause = pron->lemma();

        // First pause trace ending at the cutoff or later (nullptr if still in the pause) and the trace following it
        struct Split {
            LatticeTrace* spanning  = nullptr;
            LatticeTrace* successor = nullptr;
        };
        std::vector<Split> splits(hyps.size());

        for (size_t i = 0ul; i < hyps.size(); ++i) {
            auto const&   hyp       = hyps[i];
            LatticeTrace* successor = hyp.traceIsOpen ? hyp.trace->get() : nullptr;
            LatticeTrace* curr      = finished(hyp);
            bool          onlyPause = true;
            Split         split;
            while (curr and curr != start and curr->time > start->time) {
                if (not curr->pronunciation or curr->pronunciation->lemma() != pause) {
                    onlyPause = false;
                    split     = {};
                }
                else if (curr->time >= cutoff_) {
                    split = {curr, successor};
                }
                successor = curr;
                curr      = curr->predecessor.get();
            }
            if (curr == start and (split.spanning or (onlyPause and hyp.openPause == pause))) {
                keep[i]   = true;
                splits[i] = split.spanning ? split : Split{nullptr, hyp.traceIsOpen ? hyp.trace->get() : nullptr};
            }
        }

        // Nothing to split if all kept hypotheses share the same pause trace already
        LatticeTrace* common = nullptr;
        bool          shared = true;
        for (size_t i = 0ul; i < hyps.size() and shared; ++i) {
            if (keep[i]) {
                shared = splits[i].spanning and (not common or splits[i].spanning == common);
                common = splits[i].spanning;
            }
        }
        if (shared) {
            return;
        }

        // The part before the cutoff has no score of its own
        auto before = Core::ref(new LatticeTrace(Core::Ref<LatticeTrace>(start), pron, cutoff_, start->score, {}));
        for (size_t i = 0ul; i < hyps.size(); ++i) {
            if (not keep[i]) {
                continue;
            }
            auto [spanning, successor] = splits[i];
            if (spanning and spanning->time > cutoff_) {
                spanning->predecessor = before;
            }
            else if (successor) {
                successor->predecessor = before;
            }
            else {
                *hyps[i].trace = before;
            }
        }
    }
};

}  // namespace Search

#endif  // SEARCH_STABLE_DELAY_PRUNING_HH
