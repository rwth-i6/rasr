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
#include "SmoothedAccuracy.hh"
#include "Arithmetic.hh"
#include "Compose.hh"
#include <Fsa/Arithmetic.hh>
#include <Fsa/Basic.hh>
#include <Fsa/Cache.hh>
#include <Fsa/Compose.hh>
#include <Fsa/Determinize.hh>
#include <Fsa/Levenshtein.hh>
#include <Fsa/Minimize.hh>
#include <Fsa/Project.hh>
#include <Fsa/RemoveEpsilons.hh>
#include <Fsa/Static.hh>
#include <Core/Vector.hh>
#include <Core/Assertions.hh>
#include <Core/Hash.hh>
#include <Core/Types.hh>
#include <Speech/PhonemeSequenceAlignmentGenerator.hh>
#include <Speech/AuxiliarySegmentwiseTrainer.hh>


namespace Lattice {

    /**
     * SmoothedFrameStateAccuracyAutomaton
     */
    class SmoothedFrameStateAccuracyAutomaton : public ModifyWordLattice
    {
    private:
        typedef std::unordered_map<Fsa::LabelId, f64> States;
        typedef Core::Vector<States> ActiveStates;
    protected:
        Core::Ref<const Bliss::LemmaPronunciationAlphabet> alphabet_;
        ActiveStates stateIds_;
        Speech::AlignmentGeneratorRef alignmentGenerator_;
        SmoothingFunction &smoothing_;
    private:
        class LatticeToActiveStates : public DfsState
        {
        private:
            SmoothedFrameStateAccuracyAutomaton &parent_;
        public:
            LatticeToActiveStates(ConstWordLatticeRef, SmoothedFrameStateAccuracyAutomaton &parent);
            virtual void discoverState(Fsa::ConstStateRef sp);
            virtual void finish();
        };
    protected:
        Fsa::LabelId label(Fsa::LabelId e) const {
            return alignmentGenerator_->acousticModel()->emissionIndex(e);
        }
        f32 accuracy(const States &refs, const Fsa::LabelId &h) const;
    public:
        SmoothedFrameStateAccuracyAutomaton(
            ConstWordLatticeRef, ConstWordLatticeRef, Speech::AlignmentGeneratorRef, SmoothingFunction &);
        virtual std::string describe() const {
            return Core::form("smoothed-frame-state-accuracy(%s,%s)", fsa_->describe().c_str(), smoothing_.name().c_str());
        }
        virtual void modifyState(Fsa::State *sp) const;
    };

    SmoothedFrameStateAccuracyAutomaton::LatticeToActiveStates::LatticeToActiveStates(
        ConstWordLatticeRef correct, SmoothedFrameStateAccuracyAutomaton &parent) :
        DfsState(correct),
        parent_(parent)
    {}

    void SmoothedFrameStateAccuracyAutomaton::LatticeToActiveStates::discoverState(Fsa::ConstStateRef sp)
    {
        const Speech::TimeframeIndex startTime = wordBoundaries_->time(sp->id());
        for (Fsa::State::const_iterator a = sp->begin(); a != sp->end(); ++ a) {
            const Bliss::LemmaPronunciation *pronunciation = parent_.alphabet_->lemmaPronunciation(a->input());
            if (!pronunciation) continue;
            const Speech::TimeframeIndex endTime =
                wordBoundaries_->time(fsa_->getState(a->target())->id());
            Bliss::Coarticulated<Bliss::LemmaPronunciation> coarticulatedPronunciation(
                *pronunciation, wordBoundaries_->transit(sp->id()).final,
                wordBoundaries_->transit(fsa_->getState(a->target())->id()).initial);
            const Speech::Alignment *alignment =
                parent_.alignmentGenerator_->getAlignment(coarticulatedPronunciation, startTime, endTime);
            for (std::vector<Speech::AlignmentItem>::const_iterator al = alignment->begin(); al != alignment->end(); ++ al) {
                parent_.stateIds_.grow(al->time, States());
                States &stateIds = parent_.stateIds_[al->time];
                if (stateIds.find(parent_.label(al->emission)) == stateIds.end()) {
                    stateIds[parent_.label(al->emission)] = f32(a->weight());
                } else {
                    stateIds[parent_.label(al->emission)] += f32(a->weight());
                }
            }
        }
    }

    void SmoothedFrameStateAccuracyAutomaton::LatticeToActiveStates::finish()
    {
        for (Speech::TimeframeIndex t = 0; t < parent_.stateIds_.size(); ++ t) {
            States &stateIds = parent_.stateIds_[t];
            f64 sum = 0;
            for (States::iterator it = stateIds.begin(); it != stateIds.end(); ++ it) {
                sum += it->second;
            }
            for (States::iterator it = stateIds.begin(); it != stateIds.end(); ++ it) {
                it->second = sum;
            }
            parent_.smoothing_.updateStatistics(sum);
        }
    }

    SmoothedFrameStateAccuracyAutomaton::SmoothedFrameStateAccuracyAutomaton(
        ConstWordLatticeRef lattice,
        ConstWordLatticeRef correct,
        Speech::AlignmentGeneratorRef alignmentGenerator,
        SmoothingFunction &smoothing)
        :
        ModifyWordLattice(lattice),
        alphabet_(required_cast(const Bliss::LemmaPronunciationAlphabet*,
                                lattice->part(0)->getInputAlphabet().get())),
        alignmentGenerator_(alignmentGenerator),
        smoothing_(smoothing)
   {
        LatticeToActiveStates s(correct, *this);
        s.dfs();
    }

    void SmoothedFrameStateAccuracyAutomaton::modifyState(Fsa::State *sp) const
    {
        const Speech::TimeframeIndex startTime = wordBoundaries_->time(sp->id());
        for (Fsa::State::iterator a = sp->begin(); a != sp->end(); ++ a) {
            const Bliss::LemmaPronunciation *pronunciation = alphabet_->lemmaPronunciation(a->input());
            f32 weight = 0;
            if (pronunciation) {
                const Speech::TimeframeIndex endTime =
                    wordBoundaries_->time(fsa_->getState(a->target())->id());
                Bliss::Coarticulated<Bliss::LemmaPronunciation> coarticulatedPronunciation(
                    *pronunciation, wordBoundaries_->transit(sp->id()).final,
                    wordBoundaries_->transit(fsa_->getState(a->target())->id()).initial);
                const Speech::Alignment *alignment =
                    alignmentGenerator_->getAlignment(coarticulatedPronunciation, startTime, endTime);
                for (std::vector<Speech::AlignmentItem>::const_iterator al = alignment->begin(); al != alignment->end(); ++ al) {
                    weight += accuracy(stateIds_[al->time], label(al->emission));
                }
            }
            a->weight_ = Fsa::Weight(weight);
        }
    }

    f32 SmoothedFrameStateAccuracyAutomaton::accuracy(const States &refs, const Fsa::LabelId &h) const
    {
        States::const_iterator it = refs.find(h);
        return it != refs.end() ? smoothing_.dfx(it->second) : 0;
    }

    ConstWordLatticeRef getSmoothedFrameStateAccuracy(
        ConstWordLatticeRef lattice,
        ConstWordLatticeRef correct,
        Speech::AlignmentGeneratorRef alignmentGenerator,
        SmoothingFunction &smoothing)
    {
        Core::Ref<SmoothedFrameStateAccuracyAutomaton> a(
            new SmoothedFrameStateAccuracyAutomaton(lattice, correct, alignmentGenerator, smoothing));
        Core::Ref<WordLattice> result(new WordLattice);
        result->setWordBoundaries(a->wordBoundaries());
        result->setFsa(Fsa::cache(a), WordLattice::accuracyFsa);
        return result;
    }

} // namespace Lattice
