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
#include "LatticeRescorerAutomaton.hh"
#include <Fsa/Best.hh>

using namespace Speech;

/*
 * LatticeRescorerAutomaton: base class
 */
LatticeRescorerAutomaton::LatticeRescorerAutomaton(Lattice::ConstWordLatticeRef lattice)
        : Precursor(lattice) {
    setProperties(Fsa::PropertySortedByWeight, Fsa::PropertyNone);
}

void LatticeRescorerAutomaton::modifyState(Fsa::State* sp) const {
    if (sp->isFinal())
        sp->weight_ = semiring()->one();
    for (Fsa::State::iterator a = sp->begin(); a != sp->end(); ++a)
        a->weight_ = score(sp->id(), *a);
}

/*
 * LatticeRescorerAutomaton: with cache
 */
CachedLatticeRescorerAutomaton::CachedLatticeRescorerAutomaton(Lattice::ConstWordLatticeRef lattice)
        : Precursor(lattice) {}

void CachedLatticeRescorerAutomaton::modifyState(Fsa::State* sp) const {
    if (sp->isFinal())
        sp->weight_ = semiring()->one();
    for (Fsa::State::iterator a = sp->begin(); a != sp->end(); ++a) {
        const Key key(
                a->input(),
                (*wordBoundaries_)[sp->id()],
                (*wordBoundaries_)[fsa_->getState(a->target())->id()]);
        Scores::const_iterator it = cache_.find(key.str);
        if (it == cache_.end())
            cache_.insert(std::make_pair(key.str, score(sp->id(), *a)));
        a->weight_ = cache_[key.str];
    }
}

/*
 * EmissionLatticeRescorerAutomaton
 */
EmissionLatticeRescorerAutomaton::EmissionLatticeRescorerAutomaton(
        Lattice::ConstWordLatticeRef lattice,
        AlignmentGeneratorRef        alignmentGenerator,
        ConstSegmentwiseFeaturesRef  features,
        Core::Ref<Am::AcousticModel> acousticModel)
        : Precursor(lattice),
          alignmentGenerator_(alignmentGenerator),
          features_(features),
          acousticModel_(acousticModel) {
    require(alignmentGenerator_ && acousticModel_);
}

Fsa::Weight EmissionLatticeRescorerAutomaton::score(Fsa::StateId s, const Fsa::Arc& a) const {
    const Bliss::LemmaPronunciationAlphabet* alphabet      = required_cast(const Bliss::LemmaPronunciationAlphabet*, fsa_->getInputAlphabet().get());
    const Bliss::LemmaPronunciation*         pronunciation = alphabet->lemmaPronunciation(a.input());
    const TimeframeIndex                     begtime       = wordBoundaries_->time(s);
    if (pronunciation && begtime != InvalidTimeframeIndex) {
        Bliss::Coarticulated<Bliss::LemmaPronunciation> coarticulatedPronunciation(*pronunciation, wordBoundaries_->transit(s).final,
                                                                                   wordBoundaries_->transit(fsa_->getState(a.target())->id()).initial);
        const TimeframeIndex                            endtime = wordBoundaries_->time(fsa_->getState(a.target())->id());
        return _score(coarticulatedPronunciation, begtime, endtime);
    }
    else {
        return fsa_->semiring()->one();
    }
}

Fsa::Weight EmissionLatticeRescorerAutomaton::_score(const Bliss::Coarticulated<Bliss::LemmaPronunciation>& coarticulatedPronunciation,
                                                     TimeframeIndex begtime, TimeframeIndex endtime) const {
    if (begtime < endtime) {
        f32              score     = fsa_->semiring()->one();
        const Alignment* alignment = alignmentGenerator_->getAlignment(coarticulatedPronunciation, begtime, endtime);
        for (std::vector<AlignmentItem>::const_iterator al = alignment->begin(); al != alignment->end(); ++al) {
            Mm::FeatureScorer::Scorer scorer = acousticModel_->featureScorer()->getScorer((*features_)[al->time]);
            score += scorer->score(acousticModel_->emissionIndex(al->emission));
        }
        return Fsa::Weight(score);
    }
    else {
        Core::Application::us()->warning("score 0 assigned to arc with begin time ")
                << begtime << " , end time " << endtime << " and label id " << coarticulatedPronunciation.object().id();
        return fsa_->semiring()->one();
    }
}

/*
 * TdpLatticeRescorerAutomaton
 */
TdpLatticeRescorerAutomaton::TdpLatticeRescorerAutomaton(Lattice::ConstWordLatticeRef       lattice,
                                                         AlignmentGeneratorRef              alignmentGenerator,
                                                         AllophoneStateGraphBuilder*        allophoneStateGraphBuilder,
                                                         Core::Ref<const Am::AcousticModel> acousticModel)
        : Precursor(lattice),
          alignmentGenerator_(alignmentGenerator),
          allophoneStateGraphBuilder_(allophoneStateGraphBuilder),
          acousticModel_(acousticModel),
          alphabet_(required_cast(const Bliss::LemmaPronunciationAlphabet*, fsa_->getInputAlphabet().get())) {
    require(alignmentGenerator_ && allophoneStateGraphBuilder_ && acousticModel_);
    require(alphabet_);
}

Fsa::Weight TdpLatticeRescorerAutomaton::score(Fsa::StateId s, const Fsa::Arc& a) const {
    const Bliss::LemmaPronunciation* pronunciation =
            alphabet_->lemmaPronunciation(a.input());
    const TimeframeIndex begtime = wordBoundaries_->time(s);
    if (pronunciation && begtime != InvalidTimeframeIndex) {
        const TimeframeIndex                            endtime = wordBoundaries_->time(fsa_->getState(a.target())->id());
        Bliss::Coarticulated<Bliss::LemmaPronunciation> coarticulatedPronunciation(*pronunciation, wordBoundaries_->transit(s).final,
                                                                                   wordBoundaries_->transit(fsa_->getState(a.target())->id()).initial);
        return _score(coarticulatedPronunciation, begtime, endtime);
    }
    else {
        return fsa_->semiring()->one();
    }
}

Fsa::Weight TdpLatticeRescorerAutomaton::_score(
        const Bliss::Coarticulated<Bliss::LemmaPronunciation>& coarticulatedPronunciation,
        TimeframeIndex begtime, TimeframeIndex endtime) const {
    if (begtime < endtime) {
        const Alignment* alignment = alignmentGenerator_->getAlignment(coarticulatedPronunciation, begtime, endtime);
        Fsa::Weight      score     = Fsa::bestscore(Fsa::staticCopy(
                allophoneStateGraphBuilder_->build(*alignment,
                                                            Bliss::Coarticulated<Bliss::Pronunciation>(*coarticulatedPronunciation.object().pronunciation(),
                                                                                                       coarticulatedPronunciation.leftContext(),
                                                                                                       coarticulatedPronunciation.rightContext()))));
        if (fsa_->semiring()->compare(score, fsa_->semiring()->invalid()) == 0) {
            score = Fsa::Weight(1e9);  // fsa_->semiring()->zero();
        }
        return score;
    }
    else {
        Core::Application::us()->warning("score 0 assigned to arc with begin time ")
                << begtime << " , end time " << endtime << " and label id " << coarticulatedPronunciation.object().id();
        return fsa_->semiring()->one();
    }
}

/*
 * CombinedAcousticLatticeRescorerAutomaton
 */
CombinedAcousticLatticeRescorerAutomaton::CombinedAcousticLatticeRescorerAutomaton(Lattice::ConstWordLatticeRef lattice,
                                                                                   AlignmentGeneratorRef        alignmentGenerator,
                                                                                   Core::Ref<Am::AcousticModel> acousticModel,
                                                                                   ConstSegmentwiseFeaturesRef  features,
                                                                                   AllophoneStateGraphBuilder*  allophoneStateGraphBuilder)
        : Precursor(lattice),
          emissionRescorer_(new EmissionLatticeRescorerAutomaton(lattice, alignmentGenerator, features, acousticModel)),
          tdpRescorer_(new TdpLatticeRescorerAutomaton(lattice, alignmentGenerator, allophoneStateGraphBuilder, acousticModel)) {}

Fsa::Weight CombinedAcousticLatticeRescorerAutomaton::score(Fsa::StateId s, const Fsa::Arc& a) const {
    const Bliss::LemmaPronunciationAlphabet* alphabet      = required_cast(const Bliss::LemmaPronunciationAlphabet*, fsa_->getInputAlphabet().get());
    const Bliss::LemmaPronunciation*         pronunciation = alphabet->lemmaPronunciation(a.input());
    const TimeframeIndex                     begtime       = wordBoundaries_->time(s);
    if (pronunciation && begtime != InvalidTimeframeIndex) {
        Bliss::Coarticulated<Bliss::LemmaPronunciation> coarticulatedPronunciation(*pronunciation, wordBoundaries_->transit(s).final,
                                                                                   wordBoundaries_->transit(fsa_->getState(a.target())->id()).initial);
        const TimeframeIndex                            endtime = wordBoundaries_->time(fsa_->getState(a.target())->id());
        return _score(coarticulatedPronunciation, begtime, endtime);
    }
    else {
        return fsa_->semiring()->one();
    }
}

Fsa::Weight CombinedAcousticLatticeRescorerAutomaton::_score(const Bliss::Coarticulated<Bliss::LemmaPronunciation>& coarticulatedPronunciation,
                                                             TimeframeIndex begtime, TimeframeIndex endtime) const {
    return fsa_->semiring()->extend(emissionRescorer_->_score(coarticulatedPronunciation, begtime, endtime),
                                    tdpRescorer_->_score(coarticulatedPronunciation, begtime, endtime));
}

/*
 * CombinedAcousticSummedPronunciationLatticeRescorerAutomaton
 */
CombinedAcousticSummedPronunciationLatticeRescorerAutomaton::CombinedAcousticSummedPronunciationLatticeRescorerAutomaton(
        Lattice::ConstWordLatticeRef lattice,
        AlignmentGeneratorRef        alignmentGenerator,
        Core::Ref<Am::AcousticModel> acousticModel,
        ConstSegmentwiseFeaturesRef  features,
        AllophoneStateGraphBuilder*  allophoneStateGraphBuilder)
        : Precursor(lattice, alignmentGenerator, acousticModel, features, allophoneStateGraphBuilder) {}

Fsa::Weight CombinedAcousticSummedPronunciationLatticeRescorerAutomaton::score(Fsa::StateId s, const Fsa::Arc& a) const {
    Fsa::Weight                              resultScore   = fsa_->semiring()->one();
    const Bliss::LemmaPronunciationAlphabet* alphabet      = required_cast(const Bliss::LemmaPronunciationAlphabet*, fsa_->getInputAlphabet().get());
    const Bliss::LemmaPronunciation*         pronunciation = alphabet->lemmaPronunciation(a.input());
    const TimeframeIndex                     begtime       = wordBoundaries_->time(s);
    if (pronunciation && begtime != InvalidTimeframeIndex) {
        const TimeframeIndex                endtime = wordBoundaries_->time(fsa_->getState(a.target())->id());
        Bliss::Lemma::PronunciationIterator p, p_end;
        for (Core::tie(p, p_end) = pronunciation->lemma()->pronunciations(); p != p_end; ++p) {
            Bliss::Coarticulated<Bliss::LemmaPronunciation> coarticulatedPronunciation(*p, wordBoundaries_->transit(s).final,
                                                                                       wordBoundaries_->transit(fsa_->getState(a.target())->id()).initial);

            resultScore = fsa_->semiring()->collect(Precursor::_score(coarticulatedPronunciation, begtime, endtime), resultScore);
        }
    }
    return resultScore;
}

/*
 * AlignmentLatticeRescorerAutomaton
 */
AlignmentLatticeRescorerAutomaton::AlignmentLatticeRescorerAutomaton(Lattice::ConstWordLatticeRef lattice,
                                                                     AlignmentGeneratorRef        alignmentGenerator)
        : Precursor(lattice),
          alignmentGenerator_(alignmentGenerator) {
    require(alignmentGenerator_);
}

Fsa::Weight AlignmentLatticeRescorerAutomaton::score(Fsa::StateId s, const Fsa::Arc& a) const {
    const Bliss::LemmaPronunciationAlphabet* alphabet      = required_cast(const Bliss::LemmaPronunciationAlphabet*, fsa_->getInputAlphabet().get());
    const Bliss::LemmaPronunciation*         pronunciation = alphabet->lemmaPronunciation(a.input());
    const TimeframeIndex                     begtime       = wordBoundaries_->time(s);
    if (pronunciation && begtime != InvalidTimeframeIndex) {
        // fsa_->getState(a.target())->id(): guarantee that wordBoundary at a.target() exists
        Bliss::Coarticulated<Bliss::LemmaPronunciation> coarticulatedPronunciation(*pronunciation, wordBoundaries_->transit(s).final,
                                                                                   wordBoundaries_->transit(fsa_->getState(a.target())->id()).initial);
        const TimeframeIndex                            endtime = wordBoundaries_->time(fsa_->getState(a.target())->id());
        return _score(coarticulatedPronunciation, begtime, endtime);
    }
    else {
        return fsa_->semiring()->one();
    }
}

Fsa::Weight AlignmentLatticeRescorerAutomaton::_score(const Bliss::Coarticulated<Bliss::LemmaPronunciation>& coarticulatedPronunciation,
                                                      TimeframeIndex begtime, TimeframeIndex endtime) const {
    if (begtime < endtime) {
        return Fsa::Weight(alignmentGenerator_->alignmentScore(coarticulatedPronunciation, begtime, endtime));
    }
    else {
        Core::Application::us()->warning("score 0 assigned to arc with begin time ")
                << begtime << " , end time " << endtime << " and label id " << coarticulatedPronunciation.object().id();
        return fsa_->semiring()->one();
    }
}

/*
 * PronunciationLatticeRescorerAutomaton
 */
PronunciationLatticeRescorerAutomaton::PronunciationLatticeRescorerAutomaton(Lattice::ConstWordLatticeRef lattice, f32 pronunciationScale)
        : LatticeRescorerAutomaton(lattice),
          alphabet_(required_cast(const Bliss::LemmaPronunciationAlphabet*, fsa_->getInputAlphabet().get())),
          pronunciationScale_(pronunciationScale) {}

Fsa::Weight PronunciationLatticeRescorerAutomaton::score(Fsa::StateId s, const Fsa::Arc& a) const {
    const Bliss::LemmaPronunciation* pronunciation = alphabet_->lemmaPronunciation(a.input());
    if (pronunciation) {
        return Fsa::Weight(pronunciationScale_ * pronunciation->pronunciationScore());
    }
    return fsa_->semiring()->one();
}

/*
 * LmLatticeRescorerAutomaton
 */
LmLatticeRescorerAutomaton::LmLatticeRescorerAutomaton(Lattice::ConstWordLatticeRef             lattice,
                                                       Core::Ref<const Lm::ScaledLanguageModel> languageModel,
                                                       f32                                      pronunciationScale)
        : LatticeRescorerAutomaton(lattice),
          languageModel_(languageModel),
          alphabet_(required_cast(const Bliss::LemmaPronunciationAlphabet*, fsa_->getInputAlphabet().get())),
          pronunciationScale_(pronunciationScale) {}

Fsa::Weight LmLatticeRescorerAutomaton::score(Fsa::StateId s, const Fsa::Arc& a) const {
    if (s == fsa_->initialStateId()) {
        histories_.grow(s);
        histories_[s] = languageModel_->startHistory();
    }
    require(histories_[s].isValid());
    Lm::History                      hist  = histories_[s];
    Lm::Score                        score = 0;
    const Bliss::LemmaPronunciation* lp    = alphabet_->lemmaPronunciation(a.input());
    if (lp) {
        Lm::addLemmaPronunciationScore(languageModel_, lp, pronunciationScale_, languageModel_->scale(), hist, score);
    }
    /*! \warning sentence end score has to be added manually */
    if (fsa_->getState(a.target())->isFinal()) {
        score += languageModel_->sentenceEndScore(hist);
        hist = languageModel_->startHistory();
    }

    histories_.grow(a.target());
    if (!histories_[a.target()].isValid()) {
        histories_[a.target()] = hist;
    }
    if (!(hist == histories_[a.target()])) {
        languageModel_->error() << "Mismatch between lattice and language model: "
                                   "ambiguous history at state '"
                                << a.target() << "'.\n"
                                                 "Possible causes: 1) lattice is time-conditioned,\n"
                                                 "2) lattice has been generated by using another language model.";
    }
    return Fsa::Weight(score);
}
