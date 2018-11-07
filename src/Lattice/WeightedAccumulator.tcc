#include <Mm/AssigningFeatureScorer.hh>

using namespace Lattice;

/**
 * WeightedCachedAcousticAccumulator
 */
template <class Trainer>
WeightedCachedAcousticAccumulator<Trainer>::WeightedCachedAcousticAccumulator(
    typename Precursor::ConstSegmentwiseFeaturesRef features,
    typename Precursor::AlignmentGeneratorRef alignmentGenerator,
    Trainer *trainer,
    Mm::Weight weightThreshold,
    Core::Ref<const Am::AcousticModel> acousticModel,
    const Confidences &confidences)
    :
    Precursor(features, alignmentGenerator, trainer, weightThreshold, acousticModel),
    confidences_(confidences)
{}

template <class Trainer>
void WeightedCachedAcousticAccumulator<Trainer>::process(
    typename Precursor::TimeframeIndex t,
    Mm::MixtureIndex m,
    Mm::Weight w)
{
    Precursor::process(t, m, w * confidences_[t]);
}


/**
 * CachedAcousticSparseAccumulator
 */
template <class Trainer>
CachedAcousticSparseAccumulator<Trainer>::CachedAcousticSparseAccumulator(
    typename Precursor::ConstSegmentwiseFeaturesRef features,
    typename Precursor::AlignmentGeneratorRef alignmentGenerator,
    Trainer *trainer,
    Mm::Weight weightThreshold,
    Core::Ref<const Am::AcousticModel> acousticModel)
    :
    Precursor(features, alignmentGenerator, trainer, weightThreshold, acousticModel)
{}

template<class Trainer>
void CachedAcousticSparseAccumulator<Trainer>::finish()
{
    Precursor::finish();
    for (Collector::const_iterator c = this->collector_.begin(); c != this->collector_.end(); ++ c) {
        Core::Ref<const Speech::Feature> af = (*this->accumulationFeatures_)[c->first.t];
        const Sparse::SpeechFeature *asf = dynamic_cast<const Sparse::SpeechFeature*>(af.get());
        require(!asf or (asf->nSparseStreams() == 0));
    }
}


/**
 * DensityCachedAcousticAccumulator
 */
template <class Trainer>
DensityCachedAcousticAccumulator<Trainer>::DensityCachedAcousticAccumulator(
    typename Precursor::ConstSegmentwiseFeaturesRef features,
    typename Precursor::AlignmentGeneratorRef alignmentGenerator,
    Trainer *trainer,
    Mm::Weight weightThreshold,
    Core::Ref<const Am::AcousticModel> acousticModel)
    :
    Precursor(features, alignmentGenerator, trainer, weightThreshold, acousticModel)
{}

template <class Trainer>
void DensityCachedAcousticAccumulator<Trainer>::finish()
{
    for (Collector::const_iterator c = this->collector_.begin(); c != this->collector_.end(); ++ c) {
        posteriorFeatureScorer_->setFilter(c->first.m);
        Core::Ref<PosteriorScorer> scorer(
            required_cast(
            PosteriorScorer*,
             posteriorFeatureScorer_->getAssigningScorer(
             (*this->features_)[c->first.t]).get()));
        Core::Ref<const Speech::Feature> af((*this->accumulationFeatures_)[c->first.t]);
        const Sparse::SpeechFeature *asf = dynamic_cast<const Sparse::SpeechFeature*>(af.get());
        if (posteriorFeatureScorer_->useViterbi()) {
            const PosteriorsAndDensities &p = scorer->posteriorsAndDensities();
            this->accumulate(af->mainStream(), p.begin()->first, c->second);
            if (asf and asf->nSparseStreams()) {
                this->accumulate(asf->sparseStream(0), p.begin()->first, c->second);
            }
        } else {
            PosteriorsAndDensities p = scorer->posteriorsAndDensities();
            this->accumulate(af->mainStream(), p * c->second);
            require(!asf or (asf->nSparseStreams() == 0));
        }
    }
}

/**
 * TdpAccumulator
 */
template <class Trainer>
TdpAccumulator<Trainer>::TdpAccumulator(
    typename Precursor::ConstSegmentwiseFeaturesRef features,
    typename Precursor::AlignmentGeneratorRef alignmentGenerator,
    Trainer *trainer,
    Mm::Weight weightThreshold,
    Core::Ref<const Am::AcousticModel> acousticModel)
    :
    Precursor(features, alignmentGenerator, trainer, weightThreshold, acousticModel)
{}

template <class Trainer>
void TdpAccumulator<Trainer>::discoverState(Fsa::ConstStateRef sp)
{
    for (Fsa::State::const_iterator a = sp->begin(); a != sp->end(); ++ a) {
        const typename Precursor::Alignment *alignment = this->getAlignment(sp, *a);
        if (alignment) {
            f32 weight = f32(a->weight());
            if (weight > this->weightThreshold_) {
                std::vector<Speech::AlignmentItem>::const_iterator al = alignment->begin();
                require(al != alignment->end());
                const TransitionFeatures::IndexedFeatures &features =
                    transitions_->getIndexedFeatures(TransitionFeatures::Transition(-1, al->emission));
                for (TransitionFeatures::IndexedFeatures::const_iterator it = features.begin(); it != features.end(); ++ it) {
                    accumulate(it->second, it->first, weight);
                }
                for (; al != alignment->end(); ++ al) {
                    typename Precursor::Alignment::const_iterator nextAl = al + 1;
                    const TransitionFeatures::IndexedFeatures &features =
                    transitions_->getIndexedFeatures(
                        TransitionFeatures::Transition(
                            al->emission, nextAl != alignment->end() ? nextAl->emission : -1));
                    for (TransitionFeatures::IndexedFeatures::const_iterator it = features.begin(); it != features.end(); ++ it) {
                        this->accumulate(it->second, it->first, weight);
                    }
                }
            }
        }
    }
}

/**
 * LmAccumulator
 */
template <class Trainer>
LmAccumulator<Trainer>::LmAccumulator(
    Trainer *trainer,
    Mm::Weight weightThreshold,
    Core::Ref<const Lm::LanguageModel> languageModel)
    :
    Precursor(trainer, weightThreshold),
    languageModel_(languageModel)
{}

template <class Trainer>
void LmAccumulator<Trainer>::discoverState(Fsa::ConstStateRef sp)
{
    if (sp->id() == this->fsa_->initialStateId()) {
        histories_.grow(sp->id());
        histories_[sp->id()] = languageModel_->startHistory();
    }
    require(histories_[sp->id()].isValid());
    for (Fsa::State::const_iterator a = sp->begin(); a != sp->end(); ++ a) {
        const f32 weight = f32(a->weight());
        const Bliss::LemmaPronunciation *lp = alphabet_->lemmaPronunciation(a->input());
        Lm::History hist = histories_[sp->id()];
        if (lp) {
            const Bliss::SyntacticTokenSequence tokenSequence(lp->lemma()->syntacticTokenSequence());
            for (u32 ti = 0; ti < tokenSequence.length(); ++ ti) {
                const Bliss::SyntacticToken *st = tokenSequence[ti];
                if (weight > this->weightThreshold_) {
                    accumulate(mgrams_->getFeatures(hist, st), 0, weight);
                }
                hist = languageModel_->extendedHistory(hist, st);
            }
        }
        if (this->fsa_->getState(a->target())->isFinal()) {
            if (weight > this->weightThreshold_) {
                accumulate(mgrams_->getFeatures(hist, languageModel_->sentenceEndToken()), 0, weight);
            }
            hist = languageModel_->startHistory();
        }

        histories_.grow(a->target());
        if (!histories_[a->target()].isValid()) {
           histories_[a->target()] = hist;
        }
        if (!(hist == histories_[a->target()])) {
           languageModel_->error() <<
           "Mismatch between lattice and language model: "            \
           "ambiguous history at state '" << a->target() << "' ('" <<
           languageModel_->formatHistory(hist) << "' vs. '" <<
           languageModel_->formatHistory(histories_[a->target()]) << "').\n" \
           "Possible causes: 1) lattice is time-conditioned,\n"	     \
           "2) lattice has been generated by using another language model.";
        }
    }
}

template <class Trainer>
void LmAccumulator<Trainer>::setFsa(Fsa::ConstAutomatonRef fsa)
{
    Precursor::setFsa(fsa);
    alphabet_ = required_cast(const Bliss::LemmaPronunciationAlphabet*, this->fsa_->getInputAlphabet().get());
}

/**
 * WeightedDensityCachedAcousticAccumulator
 */
template <class Trainer>
WeightedDensityCachedAcousticAccumulator<Trainer>::WeightedDensityCachedAcousticAccumulator(
    typename Precursor::ConstSegmentwiseFeaturesRef features,
    typename Precursor::AlignmentGeneratorRef alignmentGenerator,
    Trainer *trainer,
    Mm::Weight weightThreshold,
    Core::Ref<const Am::AcousticModel> acousticModel,
    const Confidences &confidences)
    :
    Precursor(features, alignmentGenerator, trainer, weightThreshold, acousticModel),
    confidences_(confidences)
{}

template <class Trainer>
void WeightedDensityCachedAcousticAccumulator<Trainer>::process(
    typename Precursor::TimeframeIndex t,
    Mm::MixtureIndex m,
    Mm::Weight w)
{
    Precursor::process(t, m, w * confidences_[t]);
}


