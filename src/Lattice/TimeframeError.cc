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
#include "TimeframeError.hh"
#include <Bliss/Lexicon.hh>
#include <Fsa/Cache.hh>
#include "Basic.hh"
#include "Posterior.hh"

namespace Lattice {

/**
 * Mapping of label ids: base class implementing trivial mapping.
 */
class Mapping {
private:
    const ShortPauses& shortPauses_;

protected:
    bool isShortPause(Fsa::LabelId id) const {
        return shortPauses_.find(id) != shortPauses_.end();
    }

public:
    Mapping(const ShortPauses& shortPauses)
            : shortPauses_(shortPauses) {}
    virtual ~Mapping() {}

    virtual Fsa::LabelId map(Fsa::LabelId pronId) const {
        return isShortPause(pronId) ? Fsa::Epsilon : pronId;
    }

    static Mapping* createMapping(
            bool useLemmata, const ShortPauses&, ConstWordLatticeRef);
};

/**
 * map to lemma ids
 */
class LemmaMapping : public Mapping {
private:
    Core::Ref<const Bliss::LemmaPronunciationAlphabet> alphabet_;

public:
    LemmaMapping(const ShortPauses& shortPauses, Core::Ref<const Bliss::LemmaPronunciationAlphabet> alphabet)
            : Mapping(shortPauses), alphabet_(alphabet) {}
    virtual ~LemmaMapping() {}

    virtual Fsa::LabelId map(Fsa::LabelId pronId) const {
        const Bliss::LemmaPronunciation* lp = alphabet_->lemmaPronunciation(pronId);
        return (lp && !isShortPause(lp->lemma()->id())) ? lp->lemma()->id() : Fsa::Epsilon;
    }
};

Mapping* Mapping::createMapping(
        bool                useLemmata,
        const ShortPauses&  shortPauses,
        ConstWordLatticeRef lattice) {
    if (useLemmata) {
        const Bliss::LemmaPronunciationAlphabet* alphabet =
                required_cast(const Bliss::LemmaPronunciationAlphabet*,
                              lattice->mainPart()->getInputAlphabet().get());
        return new LemmaMapping(shortPauses,
                                Core::Ref<const Bliss::LemmaPronunciationAlphabet>(alphabet));
    }
    else {
        return new Mapping(shortPauses);
    }
}

/**
 *  Calculate 1 - p_t(w_n|x_1,...,x_T) for each time frame and word index.
 */
class TimeframeErrorDfsState : public DfsState {
private:
    class AccumulatorMap : public std::unordered_map<Fsa::LabelId, Fsa::Accumulator*> {
    public:
        ~AccumulatorMap() {
            for (iterator it = begin(); it != end(); ++it) {
                delete it->second;
            }
        }
    };
    typedef std::vector<AccumulatorMap> Accumulators;

public:
    class ScoreMap : public std::unordered_map<Fsa::LabelId, f32> {
    public:
        f32 score(Fsa::LabelId id) const {
            ScoreMap::const_iterator it = find(id);
            verify(it != end());
            return it->second;
        }
    };
    typedef Core::Vector<ScoreMap> Scores;

private:
    Accumulators   accumulators_;
    Scores*        scores_;
    const Mapping& mapping_;
    f32            tolerance_;

public:
    TimeframeErrorDfsState(ConstWordLatticeRef lattice, const Mapping& mapping)
            : DfsState(lattice),
              accumulators_(lattice->maximumTime()),
              scores_(0),
              mapping_(mapping),
              tolerance_(-0.001) {}
    virtual ~TimeframeErrorDfsState() {}

    void discoverState(Fsa::ConstStateRef sp) {
        const Speech::TimeframeIndex start = wordBoundaries_->time(sp->id());
        verify(start != Speech::InvalidTimeframeIndex);
        for (Fsa::State::const_iterator a = sp->begin(); a != sp->end(); ++a) {
            const Speech::TimeframeIndex end = wordBoundaries_->time(fsa_->getState(a->target())->id());
            verify(end != Speech::InvalidTimeframeIndex);
            const Fsa::LabelId id = mapping_.map(a->input());
            verify(end <= accumulators_.size());
            for (Speech::TimeframeIndex time = start; time < end; ++time) {
                AccumulatorMap& acc = accumulators_[time];
                if (acc.find(id) == acc.end()) {
                    acc[id] = fsa_->semiring()->getCollector(a->weight());
                }
                else {
                    acc[id]->feed(a->weight());
                }
            }
        }
    }

    void finish() {
        verify(scores_);
        scores_->resize(accumulators_.size());
        for (Speech::TimeframeIndex time = 0; time < accumulators_.size(); ++time) {
            AccumulatorMap& acc = accumulators_[time];
            ScoreMap&       sco = (*scores_)[time];
            sco.clear();
            for (AccumulatorMap::const_iterator a = acc.begin(); a != acc.end(); ++a) {
                f32 p = f32(a->second->get());
                verify(p > tolerance_);
                sco[a->first] = Fsa::Weight(1 - exp(-p));
            }
        }
    }

    void getScores(Scores* scores) {
        require(scores);
        scores_ = scores;
        dfs();
        scores_ = 0;
    }
};

/**
 * timeframe error automaton: base class
 */
class TimeframeErrorAutomaton : public ModifyWordLattice {
    typedef TimeframeErrorDfsState::Scores Scores;

private:
    Mapping* mapping_;

protected:
    Scores scores_;

protected:
    virtual Fsa::Weight score(Speech::TimeframeIndex start, Speech::TimeframeIndex end, Fsa::LabelId id) const = 0;

public:
    TimeframeErrorAutomaton(ConstWordLatticeRef, const ShortPauses&, bool useLemmata);
    virtual ~TimeframeErrorAutomaton() {
        delete mapping_;
    }

    std::string describe() const {
        return "timeframe-error";
    }
    void modifyState(Fsa::State*) const;
};

TimeframeErrorAutomaton::TimeframeErrorAutomaton(ConstWordLatticeRef lattice,
                                                 const ShortPauses&  shortPauses,
                                                 bool                useLemmata)
        : ModifyWordLattice(lattice),
          mapping_(Mapping::createMapping(useLemmata, shortPauses, lattice)) {
    setProperties(Fsa::PropertySortedByWeight, Fsa::PropertyNone);
    TimeframeErrorDfsState s(lattice, *mapping_);
    s.getScores(&scores_);
}

void TimeframeErrorAutomaton::modifyState(Fsa::State* sp) const {
    const Speech::TimeframeIndex start = wordBoundaries_->time(sp->id());
    for (Fsa::State::iterator a = sp->begin(); a != sp->end(); ++a) {
        const Speech::TimeframeIndex end = wordBoundaries_->time(fsa_->getState(a->target())->id());
        a->weight_                       = score(start, end, mapping_->map(a->input()));
    }
}

/**
 * sum scoring
 */
class SumTimeframeErrorAutomaton : public TimeframeErrorAutomaton {
    typedef TimeframeErrorAutomaton Precursor;

private:
    f32 normalization_;

protected:
    virtual Fsa::Weight score(Speech::TimeframeIndex start, Speech::TimeframeIndex end, Fsa::LabelId id) const {
        f32 sum = 0;
        for (Speech::TimeframeIndex time = start; time < end; ++time) {
            sum += scores_[time].score(id);
        }
        f32 den = 1 + normalization_ * (end - start - 1);
        return Fsa::Weight(sum / den);
    }

public:
    SumTimeframeErrorAutomaton(ConstWordLatticeRef lattice,
                               const ShortPauses&  shortPauses,
                               bool                useLemmata,
                               f32                 normalization)
            : Precursor(lattice, shortPauses, useLemmata),
              normalization_(normalization) {}
    virtual ~SumTimeframeErrorAutomaton() {}
};

/**
 * maximum scoring
 */
class MaximumTimeframeErrorAutomaton : public TimeframeErrorAutomaton {
    typedef TimeframeErrorAutomaton Precursor;

protected:
    virtual Fsa::Weight score(Speech::TimeframeIndex start, Speech::TimeframeIndex end, Fsa::LabelId id) const {
        f32 maximum = Core::Type<f32>::min;
        for (Speech::TimeframeIndex time = start; time < end; ++time) {
            maximum = std::max(maximum, scores_[time].score(id));
        }
        return Fsa::Weight(maximum);
    }

public:
    MaximumTimeframeErrorAutomaton(ConstWordLatticeRef lattice,
                                   const ShortPauses&  shortPauses,
                                   bool                useLemmata)
            : Precursor(lattice, shortPauses, useLemmata) {}
    virtual ~MaximumTimeframeErrorAutomaton() {}
};

ConstWordLatticeRef getSumTimeframeErrors(ConstWordLatticeRef total,
                                          const ShortPauses&  shortPauses,
                                          bool                useLemmata,
                                          f32                 normalization) {
    total = changeSemiring(total, Fsa::LogSemiring);
    // remove epsilons before posterior calculation?
    Core::Ref<TimeframeErrorAutomaton> tfe(
            new SumTimeframeErrorAutomaton(
                    posterior(total), shortPauses, useLemmata, normalization));
    Core::Ref<WordLattice> result(new WordLattice);
    result->setWordBoundaries(tfe->wordBoundaries());
    result->setFsa(Fsa::cache(tfe), "timeframe");
    return result;
}

ConstWordLatticeRef getMaximumTimeframeErrors(ConstWordLatticeRef total,
                                              const ShortPauses&  shortPauses,
                                              bool                useLemmata) {
    total = changeSemiring(total, Fsa::LogSemiring);
    // remove epsilons before posterior calculation?
    Core::Ref<TimeframeErrorAutomaton> tfe(new MaximumTimeframeErrorAutomaton(posterior(total), shortPauses, useLemmata));
    Core::Ref<WordLattice>             result(new WordLattice);
    result->setWordBoundaries(tfe->wordBoundaries());
    result->setFsa(tfe, "timeframe");
    return result;
}

/**
 *  WordTimeframeAccuracyDfsState
 */
class WordTimeframeAccuracyDfsState : public DfsState {
public:
    typedef Core::Vector<std::unordered_set<Fsa::LabelId>> References;

private:
    References*    references_;
    const Mapping& mapping_;

public:
    WordTimeframeAccuracyDfsState(ConstWordLatticeRef lattice, const Mapping& mapping)
            : DfsState(lattice),
              references_(0),
              mapping_(mapping) {}
    virtual ~WordTimeframeAccuracyDfsState() {}

    void discoverState(Fsa::ConstStateRef sp) {
        const Speech::TimeframeIndex start = wordBoundaries_->time(sp->id());
        verify(start != Speech::InvalidTimeframeIndex);
        for (Fsa::State::const_iterator a = sp->begin(); a != sp->end(); ++a) {
            const Speech::TimeframeIndex end = wordBoundaries_->time(fsa_->getState(a->target())->id());
            Fsa::LabelId                 id  = mapping_.map(a->input());
            verify(end != Speech::InvalidTimeframeIndex);
            references_->grow(end);
            for (Speech::TimeframeIndex time = start; time < end; ++time) {
                (*references_)[time].insert(id);
            }
        }
    }

    void setReferences(References* references) {
        verify(references);
        references_ = references;
        references_->clear();
        dfs();
    }
};

/**
 * WordTimeframeAccuracyAutomaton
 */
class WordTimeframeAccuracyAutomaton : public ModifyWordLattice {
private:
    Mapping*                                  mapping_;
    WordTimeframeAccuracyDfsState::References references_;
    f32                                       normalization_;

public:
    WordTimeframeAccuracyAutomaton(
            ConstWordLatticeRef, ConstWordLatticeRef,
            const ShortPauses&, bool useLemmata, f32 normalization);
    virtual ~WordTimeframeAccuracyAutomaton() {
        delete mapping_;
    }

    std::string describe() const {
        return "frame-word-accuracy";
    }
    void modifyState(Fsa::State*) const;
};

WordTimeframeAccuracyAutomaton::WordTimeframeAccuracyAutomaton(ConstWordLatticeRef lattice,
                                                               ConstWordLatticeRef correct,
                                                               const ShortPauses&  shortPauses,
                                                               bool                useLemmata,
                                                               f32                 normalization)
        : ModifyWordLattice(lattice),
          mapping_(Mapping::createMapping(useLemmata, shortPauses, lattice)),
          normalization_(normalization) {
    setProperties(Fsa::PropertySortedByWeight, Fsa::PropertyNone);
    WordTimeframeAccuracyDfsState s(correct, *mapping_);
    s.setReferences(&references_);
}

void WordTimeframeAccuracyAutomaton::modifyState(Fsa::State* sp) const {
    const Speech::TimeframeIndex start = wordBoundaries_->time(sp->id());
    for (Fsa::State::iterator a = sp->begin(); a != sp->end(); ++a) {
        const Speech::TimeframeIndex end = wordBoundaries_->time(fsa_->getState(a->target())->id());
        const Fsa::LabelId           id  = mapping_->map(a->input());
        f32                          sum = 0;
        for (Speech::TimeframeIndex time = start; time < end; ++time) {
            if (references_[time].find(id) != references_[time].end()) {
                ++sum;
            }
        }
        f32 den    = 1 + normalization_ * (end - start - 1);
        a->weight_ = Fsa::Weight(sum / den);
    }
}

ConstWordLatticeRef getWordTimeframeAccuracy(ConstWordLatticeRef lattice,
                                             ConstWordLatticeRef correct,
                                             const ShortPauses&  shortPauses,
                                             bool                useLemmata,
                                             f32                 normalization) {
    Core::Ref<WordTimeframeAccuracyAutomaton> accuracy(
            new WordTimeframeAccuracyAutomaton(
                    lattice, correct, shortPauses, useLemmata, normalization));
    Core::Ref<WordLattice> result(new WordLattice);
    result->setFsa(accuracy, accuracy->describe());
    result->setWordBoundaries(accuracy->wordBoundaries());
    return result;
}

}  // namespace Lattice
