#include <Core/XmlStream.hh>
#include <Flow/Data.hh>
#include <Flow/DataAdaptor.hh>
#include <Fsa/Best.hh>
#include <Fsa/Sort.hh>
#include <Fsa/Static.hh>
#include <Mm/FeatureScorer.hh>
#include <Mm/Module.hh>
#include <Speech/AlignedFeatureCache.hh>
#include <Speech/Alignment.hh>
#include <Speech/PhonemeSequenceAlignmentGenerator.hh>

#include <Flf/FlfCore/Basic.hh>
#include <Flf/FlfCore/Traverse.hh>
#include <Flf/FlfExt/AcousticAlignment.hh>
#include <Flf/Lexicon.hh>
#include <Flf/RescoreInternal.hh>
#include <Flf/SegmentwiseSpeechProcessor.hh>
#include <Flf/TimeframeConfusionNetwork.hh>

#include <Flf/Best.hh>
#include <Flf/Copy.hh>

#include <Flow/Attributes.hh>
#include <Flow/Cache.hh>

namespace Flf {

// -------------------------------------------------------------------------
StateAlignment::StateAlignment()
        : lemmaPron(0),
          alignment(0),
          emissionScore(Semiring::Max),
          transitionScore(Semiring::Max) {}
// -------------------------------------------------------------------------

// -------------------------------------------------------------------------
typedef Core::Ref<Mm::FeatureScorer>                    FeatureScorerRef;
typedef Bliss::Coarticulated<Bliss::Pronunciation>      CoarticulatedPronunciation;
typedef Bliss::Coarticulated<Bliss::LemmaPronunciation> CoarticulatedLemmaPronunciation;

struct AlignedCoarticulatedLemmaPronunciation {
    struct Key {
        struct Hash;
        struct Equal;

        const CoarticulatedLemmaPronunciation coLp;
        const Fsa::LabelId                    id;
        const Time                            beginTime, endTime;

        Key(
                const CoarticulatedLemmaPronunciation& coLp, Time beginTime, Time endTime)
                : coLp(coLp), id(coLp.object().id()), beginTime(beginTime), endTime(endTime) {}
    };
};
struct AlignedCoarticulatedLemmaPronunciation::Key::Hash {
    size_t operator()(const AlignedCoarticulatedLemmaPronunciation::Key& k) const {
        return (k.id & 0x0FFF) | ((k.beginTime & 0x03FF) << 12) | ((k.endTime & 0x03FF) << 22);
    }
};
struct AlignedCoarticulatedLemmaPronunciation::Key::Equal {
    bool operator()(const AlignedCoarticulatedLemmaPronunciation::Key& k1, const AlignedCoarticulatedLemmaPronunciation::Key& k2) const {
        return (k1.id == k2.id) && (k1.beginTime == k2.beginTime) && (k1.endTime == k2.endTime) && (k1.coLp.leftContext() == k2.coLp.leftContext()) && (k1.coLp.rightContext() == k2.coLp.rightContext());
#if 0  // not supported by the acoustic lattice rescoring
                && (k1.coLp.leftBoundary() == k2.coLp.leftBoundary())
                && (k1.coLp.rightBoundary() == k2.coLp.rightBoundary());
#endif
    }
};

typedef std::unordered_map<AlignedCoarticulatedLemmaPronunciation::Key,
                           StateAlignment,
                           AlignedCoarticulatedLemmaPronunciation::Key::Hash,
                           AlignedCoarticulatedLemmaPronunciation::Key::Equal>
        StateAlignmentMap;
typedef std::unordered_map<AlignedCoarticulatedLemmaPronunciation::Key,
                           ConstSubWordAlignmentRef,
                           AlignedCoarticulatedLemmaPronunciation::Key::Hash,
                           AlignedCoarticulatedLemmaPronunciation::Key::Equal>
        SubWordAlignmentMap;
// -------------------------------------------------------------------------

// -------------------------------------------------------------------------
class AcousticPhonemeSequenceAligner : public Speech::PhonemeSequenceAlignmentGenerator, public SegmentwiseSpeechProcessor {
    typedef Speech::PhonemeSequenceAlignmentGenerator Precursor;

private:
    bool              computeEmissionAndTransitionScore_;
    StateAlignmentMap cache_;

protected:
    Score emissionScore(const CoarticulatedLemmaPronunciation& coLp, const Speech::Alignment* alignment) {
        verify_(alignment);
        Am::AcousticModel&                  acousticModel = const_cast<Am::AcousticModel&>(*Precursor::acousticModel());
        Speech::ConstSegmentwiseFeaturesRef featuresRef   = features();
        Mm::Score                           score         = 0.0;
        for (Speech::Alignment::const_iterator itA = alignment->begin(), endA = alignment->end(); itA != endA; ++itA) {
            Mm::FeatureScorer::Scorer scorer = acousticModel.featureScorer()->getScorer((*featuresRef)[itA->time]);
            score += scorer->score(acousticModel.emissionIndex(itA->emission));
        }
        return score;
    }

    Score transitionScore(const CoarticulatedLemmaPronunciation& coLp, const Speech::Alignment* alignment) {
        verify_(alignment);
        Fsa::Weight weight = Fsa::bestscore(Fsa::staticCopy(
                Precursor::allophoneStateGraphBuilder()->build(
                        *alignment,
                        CoarticulatedPronunciation(
                                *coLp.object().pronunciation(),
                                coLp.leftContext(),
                                coLp.rightContext()))));
#if 0  // not supported by the acoustic lattice rescoring
                                coLp.leftBoundary(),
                                coLp.rightBoundary()))));
#endif
        return Score(weight);
    }

protected:
    virtual void process(const FeatureList& features) {
        Speech::SegmentwiseFeaturesRef segmentwiseFeatures = Speech::SegmentwiseFeaturesRef(new Speech::SegmentwiseFeatures);
        for (FeatureList::const_iterator itFeature = features.begin(), endFeature = features.end();
             itFeature != endFeature; ++itFeature)
            segmentwiseFeatures->feed(*itFeature);
        setFeatures(segmentwiseFeatures);
    }

public:
    AcousticPhonemeSequenceAligner(const Core::Configuration& config, ModelCombinationRef mc, bool computeEmissionAndTransitionScore)
            : Speech::PhonemeSequenceAlignmentGenerator(config, mc),
              SegmentwiseSpeechProcessor(config, mc),
              computeEmissionAndTransitionScore_(computeEmissionAndTransitionScore) {
    }
    virtual ~AcousticPhonemeSequenceAligner() {}

    void align(const Bliss::SpeechSegment* segment) {
        cache_.clear();
        Precursor::setSpeechSegment(const_cast<Bliss::SpeechSegment*>(segment));
        SegmentwiseSpeechProcessor::processSegment(segment);
    }

    const StateAlignment& alignment(const CoarticulatedLemmaPronunciation& coLp, Time beginTime, Time endTime) {
        std::pair<StateAlignmentMap::iterator, bool> itCache = cache_.insert(
                std::make_pair(AlignedCoarticulatedLemmaPronunciation::Key(coLp, beginTime, endTime), StateAlignment()));
        StateAlignment& sa = itCache.first->second;
        if (itCache.second) {
            sa.lemmaPron = Lexicon::us()->lemmaPronunciationAlphabet()->lemmaPronunciation(coLp.object().id());
            sa.alignment = Precursor::getAlignment(coLp, beginTime, endTime);
            if (computeEmissionAndTransitionScore_ && sa.alignment) {
                sa.emissionScore   = emissionScore(coLp, sa.alignment);
                sa.transitionScore = transitionScore(coLp, sa.alignment);
            }
        }
        return sa;
    }
};
// -------------------------------------------------------------------------

// -------------------------------------------------------------------------
Time SubWordAlignment::duration() const {
    Time d = 0;
    for (const_iterator it = begin(); it != end(); ++it)
        d += it->duration;
    return d;
}
// -------------------------------------------------------------------------

// -------------------------------------------------------------------------
LatticeAlignment::LatticeAlignment(ConstLatticeRef l, AcousticPhonemeSequenceAligner* aligner, const LabelMapList& subwordMaps)
        : l_(l), isLemma_(false), aligner_(aligner), subwordMaps_(subwordMaps), size_(aligner->features()->size()) {
    switch (Lexicon::us()->alphabetId(l_->getInputAlphabet())) {
        case Lexicon::LemmaAlphabetId:
            isLemma_ = true;
            break;
        case Lexicon::LemmaPronunciationAlphabetId:
            isLemma_ = false;
            break;
        default:
            Core::Application::us()->criticalError(
                    "LatticeAlignment: Alphabet \"%s\" is not supported; lemma or lemma-pronunciation alphabet required.",
                    Lexicon::us()->alphabetName(Lexicon::us()->alphabetId(l_->getInputAlphabet())).c_str());
    }
}

LatticeAlignment::~LatticeAlignment() {}

void LatticeAlignment::setNonWordLemmaPronunciations(const Lexicon::ConstLemmaPronunciationPtrList& nonWordLemmaProns) {
    nonWordLemmaProns_ = nonWordLemmaProns;
}

const AcousticPhonemeSequenceAligner* LatticeAlignment::aligner() const {
    return aligner_;
}

StateAlignment LatticeAlignment::stateAlignment(const State* sp, State::const_iterator a) const {
    StateAlignment  sa;
    const Boundary &leftBoundary = l_->getBoundaries()->get(sp->id()), &rightBoundary = l_->getBoundaries()->get(a->target());
    Time            startTime = leftBoundary.time(), endTime = rightBoundary.time();
    /*
     * Depending on the feature extraction or re-scoring algorithms
     * the feature sequence can be shorter than the time covered by the lattice.
     * Adjust the arc ending time to the number of features,
     * in case the overlap is zero we return no alignment, but zero score
     */
    if (Time(size()) < endTime) {
        if (Time(size()) <= startTime) {
            Core::Application::us()->warning(
                    "Arc \"%s\"[%d,%d) is not in [0,%d); discard.",
                    l_->getInputAlphabet()->symbol(a->input()).c_str(),
                    startTime, endTime, size());
            return sa;
        }
        else {
            if (endTime - size() > 1)  // that case just happens too often
                Core::Application::us()->warning(
                        "Arc \"%s\"[%d,%d) is not in [0,%d); align [%d,%d).",
                        l_->getInputAlphabet()->symbol(a->input()).c_str(),
                        startTime, endTime, size(), startTime, size());
            endTime = size();
        }
    }
    /*
     * Alignment of an arc of length 0 will always fail,
     * in this case we return no alignment, but zero score;
     * verify we have "positive" length
     */
    if (startTime == endTime) {
        if (a->input() != Fsa::Epsilon)
            Core::Application::us()->warning(
                    "Arc \"%s\"[%d,%d) has no length.",
                    l_->getInputAlphabet()->symbol(a->input()).c_str(),
                    startTime, endTime);
        return sa;
    }
    else
        verify(startTime < endTime);
    /*
     * We can handle three types of arcs:
     * 1. epsilon arcs:
     *    often they are just a substitute for silence/noise/non-word-arcs,
     *    so, we try all possible non-word arcs in order ot find a matching one
     * 2. lemma (aka word):
     *    we don't have a pronunciation given,
     *    so, we try all pronunciations of the lemma and return
     *    the alignment of the lowest-scoring lemma-pronunciation
     * 3. lemma-pronunciation:
     *    the idela case, we simply align the pronunciation
     */
    if (a->input() == Fsa::Epsilon) {
        Score bestScore = Core::Type<Score>::max;
        for (Lexicon::ConstLemmaPronunciationPtrList::const_iterator itLp = nonWordLemmaProns_.begin(); itLp != nonWordLemmaProns_.end(); ++itLp) {
            const Bliss::LemmaPronunciation* tmpLp = *itLp;
            CoarticulatedLemmaPronunciation  coLp(*tmpLp, Bliss::Phoneme::term, Bliss::Phoneme::term);

            const StateAlignment& tmpSa = aligner_->alignment(coLp, startTime, endTime);
            if (tmpSa.alignment && (tmpSa.alignment->score() < bestScore)) {
                sa        = tmpSa;
                bestScore = tmpSa.alignment->score();
            }
        }
    }
    else if (isLemma_) {
        Score               bestScore = Core::Type<Score>::max;
        const Bliss::Lemma* l         = Lexicon::us()->lemmaAlphabet()->lemma(a->input());
        verify(l);
        Bliss::Lemma::LemmaPronunciationRange lpRange = l->pronunciations();
        if (lpRange.first == lpRange.second)
            Core::Application::us()->warning(
                    "\"%s\" has no pronunciation.",
                    Lexicon::us()->lemmaAlphabet()->symbol(a->input()).c_str());
        for (; lpRange.first != lpRange.second; ++lpRange.first) {
            const Bliss::LemmaPronunciation* tmpLp = lpRange.first;
            CoarticulatedLemmaPronunciation  coLp(
                    *tmpLp,
                    Bliss::Phoneme::term,   // leftBoundary.transit().final,
                    Bliss::Phoneme::term);  // rightBoundary.transit().initial,
#if 0                                        // not supported by the acoustic lattice rescoring
                    (leftBoundary.transit().boundary == AcrossWordBoundary) ? Am::Phonology::isInitialPhone : Am::Phonology::isWithinPhone,
                    (rightBoundary.transit().boundary == AcrossWordBoundary) ? Am::Phonology::isFinalPhone : Am::Phonology::isWithinPhone);
#endif
            const StateAlignment& tmpSa = aligner_->alignment(coLp, startTime, endTime);
            if (tmpSa.alignment && (tmpSa.alignment->score() < bestScore)) {
                sa        = tmpSa;
                bestScore = tmpSa.alignment->score();
            }
        }
    }
    else {
        const Bliss::LemmaPronunciation* tmpLp = Lexicon::us()->lemmaPronunciationAlphabet()->lemmaPronunciation(a->input());
        CoarticulatedLemmaPronunciation  coLp(
                *tmpLp,
                leftBoundary.transit().final,
                rightBoundary.transit().initial);
#if 0  // not supported by the acoustic lattice rescoring
                (leftBoundary.transit().boundary == AcrossWordBoundary) ? Am::Phonology::isInitialPhone : Am::Phonology::isWithinPhone,
                (rightBoundary.transit().boundary == AcrossWordBoundary) ? Am::Phonology::isFinalPhone : Am::Phonology::isWithinPhone);
#endif
        sa = aligner_->alignment(coLp, startTime, endTime);
    }
    return sa;
}

Time LatticeAlignment::duration(const State* sp, State::const_iterator a) const {
    return l_->getBoundaries()->get(a->target()).time() - l_->getBoundaries()->get(sp->id()).time();
}

ConstSubWordAlignmentRef LatticeAlignment::phonemeAlignment(const State* sp, State::const_iterator a) const {
    const StateAlignment sa = stateAlignment(sp, a);
    if (sa.alignment) {
        verify(sa.lemmaPron);
        Am::ConstAllophoneStateAlphabetRef allophoneStateAlphabet = aligner_->acousticModel()->allophoneStateAlphabet();
        SubWordAlignment*                  pa                     = new SubWordAlignment(sa.lemmaPron->id());
        Fsa::LabelId                       lastPhonemeId          = Fsa::InvalidLabelId;
        Am::AllophoneState                 lastAlloState;
        Time                               begin = 0, end = 0, pendingDuration = duration(sp, a);
        for (Speech::Alignment::const_iterator it = sa.alignment->begin(), it_end = sa.alignment->end(); it != it_end; ++it) {
            const Am::AllophoneState alloState = allophoneStateAlphabet->allophoneState(it->emission);
            const Fsa::LabelId       phonemeId = alloState.allophone()->central();
            if ((phonemeId != lastPhonemeId) || (alloState.state() < lastAlloState.state())) {
                if (lastPhonemeId != Fsa::InvalidLabelId) {
                    const Time phonemeDuration = end - begin + 1;
                    pa->push_back(SubWord(
                            lastPhonemeId, phonemeDuration,
                            (lastAlloState.allophone()->history().empty() ? Bliss::Phoneme::term : lastAlloState.allophone()->history()[0]),
                            (lastAlloState.allophone()->future().empty() ? Bliss::Phoneme::term : lastAlloState.allophone()->future()[0])));

                    pendingDuration -= phonemeDuration;
                    verify(pendingDuration > 0);
                }
                lastPhonemeId = phonemeId;
                begin = end = it->time;
            }
            else {
                verify(end < it->time);
                end = it->time;
            }
            lastAlloState = alloState;
        }
        verify(end - begin + 1 <= pendingDuration);
        pa->push_back(SubWord(
                lastPhonemeId, pendingDuration,
                (lastAlloState.allophone()->history().empty() ? Bliss::Phoneme::term : lastAlloState.allophone()->history()[0]),
                (lastAlloState.allophone()->future().empty() ? Bliss::Phoneme::term : lastAlloState.allophone()->future()[0])));

        ConstSubWordAlignmentRef phonemeAlignment = ConstSubWordAlignmentRef(pa);
        return phonemeAlignment;
    }
    else
        return ConstSubWordAlignmentRef();
}

ConstSubWordAlignmentRef LatticeAlignment::subwordAlignment(const State* sp, State::const_iterator a, u32 i) const {
    verify_(i < subwordMaps_.size());
    if (a->input() == Fsa::Epsilon)
        return ConstSubWordAlignmentRef();
    ConstSubWordAlignmentRef subwordAlignment;
    ConstSubWordAlignmentRef pa;
    Fsa::LabelId             lpLabel = Fsa::InvalidLabelId;
    if (isLemma_) {
        pa = phonemeAlignment(sp, a);
        if (!pa)
            return ConstSubWordAlignmentRef();
        lpLabel = pa->label();
    }
    else
        lpLabel = a->input();
    const LabelMap::Mapping& mapping = (*subwordMaps_[i])[lpLabel];

    switch (mapping.size()) {
        case 0:
            subwordAlignment = ConstSubWordAlignmentRef(new SubWordAlignment(lpLabel, SubWord(Fsa::Epsilon, duration(sp, a))));
            break;
        case 1:
            subwordAlignment = ConstSubWordAlignmentRef(new SubWordAlignment(lpLabel, SubWord(mapping.front().label, duration(sp, a))));
            break;
        default: {
            if (!isLemma_)
                pa = phonemeAlignment(sp, a);
            if (!pa)
                return ConstSubWordAlignmentRef();
            Lexicon::LemmaPronunciationAlphabetRef lemmaPronunciationAlphabet = Lexicon::us()->lemmaPronunciationAlphabet();
            SubWordAlignment*                      swa                        = new SubWordAlignment(lpLabel);
            SubWordAlignment::const_iterator       itPa = pa->begin(), endPa = pa->end(), itSubBeginPa = pa->begin(), itSubEndPa = pa->begin();
            for (LabelMap::Mapping::const_iterator itSubWord = mapping.begin(), endSubWord = mapping.end(); itSubWord != endSubWord; ++itSubWord) {
                Fsa::LabelId                subLpLabel           = itSubWord->label;
                Time                        subwordDuration      = 0;
                const Bliss::Pronunciation& subWordPronunciation = *lemmaPronunciationAlphabet->lemmaPronunciation(subLpLabel)->pronunciation();
                for (const Bliss::Phoneme::Id* itPhoneme = subWordPronunciation.phonemes(); *itPhoneme != Bliss::Phoneme::term; ++itPhoneme)
                    if ((itPa != endPa) && (Fsa::LabelId(*itPhoneme) == itPa->label)) {
                        subwordDuration += itPa->duration;
                        itSubEndPa = itPa;
                        ++itPa;
                    }
                swa->push_back(SubWord(subLpLabel, subwordDuration, itSubBeginPa->leftContext, itSubEndPa->rightContext));
                itSubBeginPa = itSubEndPa = itPa;

                if (subwordDuration == 0)
                    Core::Application::us()->error(
                            "Failed to align sub word \"%s\" to arc \"%s\"[%d,%d]; sub word has length zero.",
                            Lexicon::us()->lemmaPronunciationAlphabet()->symbol(subLpLabel).c_str(),
                            l_->inputAlphabet()->symbol(a->input()).c_str(),
                            l_->getBoundaries()->get(sp->id()).time(), l_->getBoundaries()->get(a->target()).time());
            }
            if (itPa != endPa)
                Core::Application::us()->error(
                        "Failed to align sub words to arc \"%s\"[%d,%d]; pending phonemes detected.",
                        l_->inputAlphabet()->symbol(a->input()).c_str(),
                        l_->getBoundaries()->get(sp->id()).time(), l_->getBoundaries()->get(a->target()).time());
            verify_(mapping.size() == swa->size());
            subwordAlignment = ConstSubWordAlignmentRef(swa);
        }
    }
    verify(subwordAlignment);
    return subwordAlignment;
}

namespace {
struct PhonemeArcAligner {
    const LatticeAlignment& latticeAlignment;
    PhonemeArcAligner(const LatticeAlignment& latticeAlignment)
            : latticeAlignment(latticeAlignment) {}
    Fsa::ConstAlphabetRef alphabet() const {
        return Lexicon::us()->phonemeInventory()->phonemeAlphabet();
    }
    ConstSubWordAlignmentRef operator()(ConstStateRef sr, State::const_iterator a) const {
        return latticeAlignment.phonemeAlignment(sr.get(), a);
    }
};

struct SubwordArcAligner {
    const LatticeAlignment& latticeAlignment;
    u32                     i_;
    SubwordArcAligner(const LatticeAlignment& latticeAlignment, u32 i)
            : latticeAlignment(latticeAlignment), i_(i) {}
    Fsa::ConstAlphabetRef alphabet() const {
        return Lexicon::us()->lemmaPronunciationAlphabet();
    }
    ConstSubWordAlignmentRef operator()(ConstStateRef sr, State::const_iterator a) const {
        return latticeAlignment.subwordAlignment(sr.get(), a, i_);
    }
};

template<class ArcAligner>
class LatticeFromLatticeAlignmentBuilder : public TraverseState {
private:
    ArcAligner         arcAligner_;
    StaticLattice*     s_;
    StaticBoundaries*  b_;
    ConstSemiringRef   semiring_;
    ConstBoundariesRef boundaries_;

    Core::Vector<Fsa::StateId> sidMap_;
    Fsa::StateId               nextSid_;

protected:
    ScoresRef partialWeight(ScoresRef weight, f32 d) {
        ScoresRef partialWeight = semiring_->clone(weight);
        for (Scores::iterator itScore = semiring_->begin(partialWeight), endScore = semiring_->end(partialWeight);
             itScore != endScore; ++itScore)
            *itScore *= d;
        return partialWeight;
    }

    virtual void exploreState(ConstStateRef sr) {
        verify(sr->id() < sidMap_.size());
        Fsa::StateId newSid = sidMap_[sr->id()];
        State*       sp     = new State(newSid);
        s_->setState(sp);
        const Boundary& boundary = boundaries_->get(sr->id());
        b_->set(newSid, boundary);
        Time t = boundary.time();
        if (sr->isFinal())
            sp->setFinal(sr->weight());
        for (State::const_iterator a = sr->begin(), a_end = sr->end(); a != a_end; ++a) {
            Fsa::StateId targetSid = a->target(), newTargetSid;
            sidMap_.grow(targetSid, Fsa::InvalidStateId);
            newTargetSid = sidMap_[targetSid];
            if (newTargetSid == Fsa::InvalidStateId)
                newTargetSid = sidMap_[targetSid] = nextSid_++;
            if (a->input() == Fsa::Epsilon) {
                sp->newArc(newTargetSid, a->weight(), Fsa::Epsilon, Fsa::Epsilon);
            }
            else {
                ConstSubWordAlignmentRef alignment = arcAligner_(sr, a);
                if (alignment) {
                    if (alignment->size() > 1) {
                        f32    arcDuration = f32(alignment->duration());
                        Time   currentT    = t;
                        State* currentSp   = sp;
                        currentSp->newArc(nextSid_, partialWeight(a->weight(), f32(alignment->front().duration) / arcDuration), alignment->front().label, a->input());
                        for (u32 i = 1; i < alignment->size() - 1; ++i) {
                            const SubWord &prevAi = (*alignment)[i - 1], &ai = (*alignment)[i];
                            currentSp = new State(nextSid_++);
                            s_->setState(currentSp);
                            currentT += prevAi.duration;
                            b_->set(currentSp->id(), Boundary(currentT, Boundary::Transit(ai.leftContext, prevAi.rightContext, WithinWordBoundary)));
                            currentSp->newArc(nextSid_, partialWeight(a->weight(), f32(ai.duration) / arcDuration), ai.label, Fsa::Epsilon);
                        }
                        const SubWord &prevAi = (*alignment)[alignment->size() - 2], &ai = (*alignment)[alignment->size() - 1];
                        currentSp = new State(nextSid_++);
                        s_->setState(currentSp);
                        currentT += prevAi.duration;
                        b_->set(currentSp->id(), Boundary(currentT, Boundary::Transit(ai.leftContext, prevAi.rightContext, WithinWordBoundary)));
                        currentSp->newArc(newTargetSid, partialWeight(a->weight(), f32(alignment->back().duration) / arcDuration), alignment->back().label, Fsa::Epsilon);
                    }
                    else {
                        verify(alignment->size() == 1);
                        sp->newArc(newTargetSid, a->weight(), alignment->front().label, a->input());
                    }
                }
                else
                    Core::Application::us()->warning(
                            "No subword alignment available for arc \"%s\"[%d,%d]; discard arc.",
                            l->inputAlphabet()->symbol(a->input()).c_str(),
                            t, l->getBoundaries()->get(a->target()).time());
            }
        }
    }

    LatticeFromLatticeAlignmentBuilder(ConstLatticeRef l, const ArcAligner& arcAligner, StaticLattice* s, StaticBoundaries* b)
            : TraverseState(l), arcAligner_(arcAligner), s_(s), b_(b) {
        semiring_               = l->semiring();
        boundaries_             = l->getBoundaries();
        Fsa::StateId initialSid = l->initialStateId();
        sidMap_.grow(initialSid, Fsa::InvalidStateId);
        sidMap_[initialSid] = 0;
        s_->setInitialStateId(0);
        nextSid_ = 1;
        traverse();
    }

public:
    static ConstLatticeRef build(ConstLatticeRef l, const ArcAligner& arcAligner);
};

template<class ArcAligner>
ConstLatticeRef LatticeFromLatticeAlignmentBuilder<ArcAligner>::build(ConstLatticeRef l, const ArcAligner& arcAligner) {
    StaticBoundariesRef b = StaticBoundariesRef(new StaticBoundaries);
    StaticLatticeRef    s = StaticLatticeRef(new StaticLattice(Fsa::TypeTransducer));
    s->setInputAlphabet(arcAligner.alphabet());
    s->setOutputAlphabet(l->inputAlphabet());
    s->setProperties(l->knownProperties(), l->properties());
    s->setSemiring(l->semiring());
    s->setBoundaries(b);
    s->setDescription(
            Core::form("acoustic-alignment(%s)", l->describe().c_str()));
    LatticeFromLatticeAlignmentBuilder<ArcAligner> builder(l, arcAligner, s.get(), b.get());
    return s;
}

class PosteriorCnFromLatticeAlignmentBuilder : public TraverseState {
protected:
    const Boundaries& boundaries;
    const FwdBwd&     fb;
    PosteriorCn&      cn;

protected:
    Probability collect(Probability score1, Probability score2) {
        return std::min(score1, score2) -
               ::log1p(::exp(std::min(score1, score2) - std::max(score1, score2)));
    }

    void add(Time t, Fsa::LabelId label, Score score) {
        verify_(t < cn.size());
        PosteriorCn::Slot&          pdf = cn[t];
        PosteriorCn::Arc            cnArc(label, score);
        PosteriorCn::Slot::iterator pdfIt = std::lower_bound(pdf.begin(), pdf.end(), cnArc);
        if ((pdfIt == pdf.end()) || (pdfIt->label != cnArc.label))
            pdf.insert(pdfIt, cnArc);
        else
            pdfIt->score = collect(pdfIt->score, cnArc.score);
    }

    void finalize() {
        PosteriorCn::Arc cnEpsArc(Fsa::Epsilon, Semiring::Invalid);
        for (u32 t = 0; t < cn.size(); ++t) {
            PosteriorCn::Slot& pdf = cn[t];
            Probability        sum = 0.0;
            for (PosteriorCn::Slot::iterator itPdf = pdf.begin(); itPdf != pdf.end(); ++itPdf)
                sum += (itPdf->score = ::exp(-itPdf->score));
            if (sum < 0.99) {
                verify(pdf.empty() || (pdf.front().label != Fsa::Epsilon));
                pdf.insert(pdf.begin(), cnEpsArc);
                pdf.front().score = 1.0 - sum;
            }
            else if (sum > 1.01)
                Core::Application::us()->warning("Sum of time frame %d is %.2f not in ~1.0", t, sum);
        }
    }

    PosteriorCnFromLatticeAlignmentBuilder(ConstLatticeRef l, const FwdBwd& fb, PosteriorCn& cn)
            : TraverseState(l),
              boundaries(*l->getBoundaries()),
              fb(fb),
              cn(cn) {}
};

class AllophoneStatePosteriorCnFromLatticeAlignmentBuilder : public PosteriorCnFromLatticeAlignmentBuilder {
    typedef PosteriorCnFromLatticeAlignmentBuilder Precursor;

private:
    const LatticeAlignment& latticeAlignment_;

protected:
    virtual void exploreState(ConstStateRef sr) {
        Time t_begin = boundaries.get(sr->id()).time();
        if (cn.size() <= t_begin)
            return;
        for (State::const_iterator a = sr->begin(), a_end = sr->end(); a != a_end; ++a) {
            if ((Fsa::FirstLabelId <= a->input()) && (a->input() <= Fsa::LastLabelId)) {
                Time                     t_end     = std::min(boundaries.get(a->target()).time(), Time(cn.size()));
                const Speech::Alignment* alignment = latticeAlignment_.stateAlignment(sr.get(), a).alignment;
                if (alignment) {
                    Probability score = fb.arc(sr, a).score();
                    // cn.grow(t_end - 1);
                    verify((t_begin <= alignment->front().time) && (alignment->back().time < t_end));
                    for (Speech::Alignment::const_iterator it = alignment->begin(), it_end = alignment->end(); it != it_end; ++it)
                        add(it->time, it->emission, score);
                }
                else
                    Core::Application::us()->warning(
                            "No alignment available for arc \"%s\"[%d,%d); ignore arc.",
                            l->inputAlphabet()->symbol(a->input()).c_str(),
                            t_begin, t_end);
            }
        }
    }

    AllophoneStatePosteriorCnFromLatticeAlignmentBuilder(ConstLatticeRef l, const FwdBwd& fb, const LatticeAlignment& latticeAlignment, PosteriorCn& cn)
            : Precursor(l, fb, cn),
              latticeAlignment_(latticeAlignment) {
        traverse();
        finalize();
    }

public:
    static ConstPosteriorCnRef build(ConstLatticeRef l, const FwdBwd& fb, const LatticeAlignment& latticeAlignment);
};
ConstPosteriorCnRef AllophoneStatePosteriorCnFromLatticeAlignmentBuilder::build(ConstLatticeRef l, const FwdBwd& fb, const LatticeAlignment& latticeAlignment) {
    PosteriorCn* cn = new PosteriorCn;
    cn->alphabet    = latticeAlignment.aligner()->acousticModel()->allophoneStateAlphabet();
    cn->resize(latticeAlignment.size());
    AllophoneStatePosteriorCnFromLatticeAlignmentBuilder builder(l, fb, latticeAlignment, *cn);
    return ConstPosteriorCnRef(cn);
}

template<class ArcAligner>
class SubwordPosteriorCnFromLatticeAlignmentBuilder : public PosteriorCnFromLatticeAlignmentBuilder {
    typedef PosteriorCnFromLatticeAlignmentBuilder Precursor;

private:
    ArcAligner arcAligner_;

protected:
    virtual void exploreState(ConstStateRef sr) {
        Time t_begin = boundaries.get(sr->id()).time();
        if (cn.size() <= t_begin)
            return;
        for (State::const_iterator a = sr->begin(), a_end = sr->end(); a != a_end; ++a) {
            if ((Fsa::FirstLabelId <= a->input()) && (a->input() <= Fsa::LastLabelId)) {
                Time                     t_end     = std::min(boundaries.get(a->target()).time(), Time(cn.size()));
                ConstSubWordAlignmentRef alignment = arcAligner_(sr, a);
                if (alignment) {
                    Probability score = fb.arc(sr, a).score();
                    // cn.grow(t_end - 1);
                    Time t = t_begin;
                    for (SubWordAlignment::const_iterator itSw = alignment->begin(), endSw = alignment->end(); itSw != endSw; ++itSw) {
                        Fsa::LabelId label = itSw->label;
                        verify(t + itSw->duration <= t_end);
                        for (Time swt_end = t + itSw->duration; t != swt_end; ++t)
                            add(t, label, score);
                    }
                }
                else
                    Core::Application::us()->warning(
                            "No alignment available for arc \"%s\"[%d,%d); ignore arc.",
                            l->inputAlphabet()->symbol(a->input()).c_str(),
                            t_begin, t_end);
            }
        }
    }

    SubwordPosteriorCnFromLatticeAlignmentBuilder(ConstLatticeRef l, const FwdBwd& fb, const ArcAligner& arcAligner, PosteriorCn& cn)
            : Precursor(l, fb, cn),
              arcAligner_(arcAligner) {
        traverse();
        finalize();
    }

public:
    static ConstPosteriorCnRef build(ConstLatticeRef l, const FwdBwd& fb, const ArcAligner& arcAligner);
};
template<class ArcAligner>
ConstPosteriorCnRef SubwordPosteriorCnFromLatticeAlignmentBuilder<ArcAligner>::build(ConstLatticeRef l, const FwdBwd& fb, const ArcAligner& arcAligner) {
    PosteriorCn* cn = new PosteriorCn;
    cn->alphabet    = arcAligner.alphabet();
    cn->resize(arcAligner.latticeAlignment.size());
    SubwordPosteriorCnFromLatticeAlignmentBuilder<ArcAligner> builder(l, fb, arcAligner, *cn);
    return ConstPosteriorCnRef(cn);
}
}  // namespace

ConstLatticeRef LatticeAlignment::phonemeLattice() const {
    return LatticeFromLatticeAlignmentBuilder<PhonemeArcAligner>::build(l_, PhonemeArcAligner(*this));
}

ConstLatticeRef LatticeAlignment::subwordLattice(u32 i) const {
    verify(i < subwordMaps_.size());
    return LatticeFromLatticeAlignmentBuilder<SubwordArcAligner>::build(l_, SubwordArcAligner(*this, i));
}

ConstPosteriorCnRef LatticeAlignment::framePosteriorCn(ConstFwdBwdRef fb) const {
    return AllophoneStatePosteriorCnFromLatticeAlignmentBuilder::build(l_, *fb, *this);
}

ConstPosteriorCnRef LatticeAlignment::phonemeFramePosteriorCn(ConstFwdBwdRef fb) const {
    return SubwordPosteriorCnFromLatticeAlignmentBuilder<PhonemeArcAligner>::build(l_, *fb, PhonemeArcAligner(*this));
}

ConstPosteriorCnRef LatticeAlignment::subwordFramePosteriorCn(ConstFwdBwdRef fb, u32 i) const {
    verify(i < subwordMaps_.size());
    return SubwordPosteriorCnFromLatticeAlignmentBuilder<SubwordArcAligner>::build(l_, *fb, SubwordArcAligner(*this, i));
}
// -------------------------------------------------------------------------

// -------------------------------------------------------------------------
LatticeAlignmentBuilder::LatticeAlignmentBuilder(const Core::Configuration& config, AcousticPhonemeSequenceAligner* aligner, const LabelMapList& subwordMaps)
        : Core::Component(config), aligner_(aligner), subwordMaps_(subwordMaps) {
    if (aligner_->acousticModel()->hmmTopologySet()->getDefault().nPhoneStates() < 3)
        Core::Application::us()->warning("Default HMM has less than 3 states; proper function of phoneme/sub-word-unit alignment cannot be guaranted.");
    for (LabelMapList::const_iterator itMap = subwordMaps.begin(); itMap != subwordMaps.end(); ++itMap) {
        verify(*itMap);
        if ((Lexicon::us()->alphabetId((*itMap)->from) != Lexicon::LemmaPronunciationAlphabetId) ||
            (Lexicon::us()->alphabetId((*itMap)->to) != Lexicon::LemmaPronunciationAlphabetId))
            Core::Application::us()->criticalError(
                    "Subword map must have lemma pronunciations as input and output; input is \"%s\", output is \"%s\".",
                    Lexicon::us()->alphabetName(Lexicon::us()->alphabetId((*itMap)->from)).c_str(),
                    Lexicon::us()->alphabetName(Lexicon::us()->alphabetId((*itMap)->to)).c_str());
    }
    nonWordLemmaProns_           = Lexicon::us()->nonWordLemmaPronunciations();
    Core::Component::Message msg = log();
    if (nonWordLemmaProns_.empty()) {
        msg << "No non-word pronunciations found.";
    }
    else {
        Core::Ref<const Bliss::PhonemeInventory> phonemeInventory = Lexicon::us()->phonemeInventory();
        msg << "Non-word pronunciations:\n";
        for (Lexicon::ConstLemmaPronunciationPtrList::const_iterator itLp = nonWordLemmaProns_.begin(); itLp != nonWordLemmaProns_.end(); ++itLp) {
            const Bliss::LemmaPronunciation& lp = **itLp;
            msg << "    " << lp.lemma()->name().str() << "   /" << lp.pronunciation()->format(phonemeInventory) << "/\n";
        }
    }
}

LatticeAlignmentBuilder::~LatticeAlignmentBuilder() {
    delete aligner_;
}

ConstLatticeAlignmentRef LatticeAlignmentBuilder::build(ConstLatticeRef l, const Bliss::SpeechSegment* segment) {
    aligner_->align(segment);
    LatticeAlignment* latticeAlignment = new LatticeAlignment(l, aligner_, subwordMaps_);
    latticeAlignment->setNonWordLemmaPronunciations(nonWordLemmaProns_);
    return ConstLatticeAlignmentRef(latticeAlignment);
}

LatticeAlignmentBuilderRef LatticeAlignmentBuilder::create(const Core::Configuration& config, const LabelMapList& subwordMaps, bool computeEmissionAndTransitionScore) {
    ModelCombinationRef             mc      = getModelCombination(config, getAm(Core::Configuration(config, "acoustic-model")));
    AcousticPhonemeSequenceAligner* aligner = new AcousticPhonemeSequenceAligner(config, mc, computeEmissionAndTransitionScore);
    return LatticeAlignmentBuilderRef(new LatticeAlignmentBuilder(config, aligner, subwordMaps));
}
// -------------------------------------------------------------------------

// -------------------------------------------------------------------------
class ArcAlignmentNode : public Node {
public:
    static const Core::ParameterBool paramProjectInput;

private:
    LatticeAlignmentBuilderRef builder_;
    bool                       projectInput_;
    ConstLatticeAlignmentRef   latticeAlignment_;

public:
    ArcAlignmentNode(const std::string& name, const Core::Configuration& config)
            : Node(name, config) {}
    virtual ~ArcAlignmentNode() {}

    virtual void init(const std::vector<std::string>& arguments) {
        if (!connected(0) || !connected(1))
            criticalError("Incoming lattice at port 0 and speech segment at port 1 expected.");
        LabelMapList          subwordMaps;
        Fsa::ConstAlphabetRef lpAlphabet = Lexicon::us()->alphabet(Lexicon::LemmaPronunciationAlphabetId);
        for (u32 i = 1;; ++i) {
            LabelMapRef map = LabelMap::load(select(Core::form("subword-map-%d", i)), lpAlphabet);
            if (map)
                subwordMaps.push_back(map);
            else
                break;
        }
        builder_      = LatticeAlignmentBuilder::create(config, subwordMaps);
        projectInput_ = paramProjectInput(config);
    }

    virtual ConstLatticeRef sendLattice(Port to) {
        if (!latticeAlignment_) {
            ConstLatticeRef             l       = requestLattice(0);
            const Bliss::SpeechSegment* segment = static_cast<const Bliss::SpeechSegment*>(requestData(1));
            latticeAlignment_                   = builder_->build(l, segment);
        }
        ConstLatticeRef l;
        if (to == 0)
            l = latticeAlignment_->phonemeLattice();
        else
            l = latticeAlignment_->subwordLattice(to - 1);
        if (projectInput_)
            l = projectInput(l);
        return l;
    }

    virtual void sync() {
        latticeAlignment_.reset();
    }
};
const Core::ParameterBool ArcAlignmentNode::paramProjectInput(
        "project-input",
        "make lattice an acceptor by mapping input to output",
        false);
NodeRef createArcAlignmentNode(const std::string& name, const Core::Configuration& config) {
    return NodeRef(new ArcAlignmentNode(name, config));
}
// -------------------------------------------------------------------------

// -------------------------------------------------------------------------
class ExtendByAcousticScoreLattice : public RescoreLattice {
    typedef RescoreLattice Precursor;

private:
    ConstLatticeAlignmentRef latticeAlignment_;
    ScoreId                  id_;
    Score                    scale_;
    Score                    maxScore_;
    bool                     scoreEpsArcs_;

public:
    ExtendByAcousticScoreLattice(
            ConstLatticeRef          l,
            ConstLatticeAlignmentRef latticeAlignment,
            ScoreId                  id,
            Score                    scale,
            Score                    maxScore,
            bool                     scoreEpsArcs,
            RescoreMode              rescoreMode)
            : Precursor(l, rescoreMode), latticeAlignment_(latticeAlignment), id_(id), scale_(scale), maxScore_(maxScore), scoreEpsArcs_(scoreEpsArcs) {
        require(latticeAlignment);
    }
    virtual ~ExtendByAcousticScoreLattice() {}

    virtual void rescore(State* sp) const {
        if (getBoundaries()->get(sp->id()).time() >= Time(latticeAlignment_->size()))
            return;
        for (State::iterator a = sp->begin(); a != sp->end(); ++a) {
            if (scoreEpsArcs_ || (a->input() != Fsa::Epsilon)) {
                const StateAlignment sa    = latticeAlignment_->stateAlignment(sp, a);
                const Score          score = (sa.alignment) ? std::min(sa.alignment->score(), maxScore_) : maxScore_;
                a->weight_->add(id_, score);
            }
        }
    }

    virtual std::string describe() const {
        return Core::form("extendByAcoustic(%s,dim=%zu)", fsa_->describe().c_str(), id_);
    }
};
ConstLatticeRef extendByAcousticScore(
        ConstLatticeRef          l,
        ConstLatticeAlignmentRef latticeAlignment,
        ScoreId id, Score scale,
        Score       maxScore,
        bool        scoreEpsArcs,
        RescoreMode rescoreMode) {
    if (!l)
        return ConstLatticeRef();
    return ConstLatticeRef(new ExtendByAcousticScoreLattice(l, latticeAlignment, id, scale, maxScore, scoreEpsArcs, rescoreMode));
}

class ExtendByAcousticScoreNode : public RescoreSingleDimensionNode {
    typedef RescoreSingleDimensionNode Precursor;

public:
    static const Core::ParameterFloat paramMaxScore;
    static const Core::ParameterBool  paramScoreEps;

private:
    LatticeAlignmentBuilderRef builder_;
    ConstLatticeAlignmentRef   latticeAlignment_;
    Score                      maxScore_;
    bool                       scoreEpsArcs_;

protected:
    virtual ConstLatticeRef rescore(ConstLatticeRef l, ScoreId id) {
        if (!latticeAlignment_) {
            const Bliss::SpeechSegment* segment = static_cast<const Bliss::SpeechSegment*>(requestData(1));
            latticeAlignment_                   = builder_->build(l, segment);
        }
        l = extendByAcousticScore(l, latticeAlignment_, id, scale(), maxScore_, scoreEpsArcs_, rescoreMode);

        return l;
    }

public:
    ExtendByAcousticScoreNode(const std::string& name, const Core::Configuration& config)
            : Precursor(name, config) {}
    virtual ~ExtendByAcousticScoreNode() {}

    virtual void init(const std::vector<std::string>& arguments) {
        if (!connected(0) || !connected(1))
            criticalError("Incoming lattice at port 0 and speech segment at port 1 expected.");
        Core::Component::Message msg = log();
        maxScore_                    = paramMaxScore(config);
        if (maxScore_ > 0.0) {
            msg << "score flooring at " << maxScore_ << "\n";
        }
        else
            maxScore_ = Semiring::Max;
        scoreEpsArcs_ = paramScoreEps(config);
        if (scoreEpsArcs_)
            msg << "score epsilon arcs\n";
        builder_ = LatticeAlignmentBuilder::create(config, LabelMapList(), false);
    }

    virtual void sync() {
        latticeAlignment_.reset();
    }
};
const Core::ParameterFloat ExtendByAcousticScoreNode::paramMaxScore(
        "max-score",
        "max score",
        0.0);
const Core::ParameterBool ExtendByAcousticScoreNode::paramScoreEps(
        "score-eps",
        "score epsilon arcs",
        false);
NodeRef createExtendByAcousticScoreNode(const std::string& name, const Core::Configuration& config) {
    return NodeRef(new ExtendByAcousticScoreNode(name, config));
}
// -------------------------------------------------------------------------

// -------------------------------------------------------------------------
class AllophoneStatePosteriorCnNode : public Node {
private:
    u32                        n_;
    FwdBwdBuilderRef           fbBuilder_;
    LatticeAlignmentBuilderRef alignmentBuilder_;

    ConstPosteriorCnRef cn_;
    bool                isValid_;

protected:
    virtual ConstPosteriorCnRef getAllophoneStatePosteriorCn() {
        if (!isValid_) {
            ConstLatticeRefList lats(n_);
            lats[0] = requestLattice(0);
            for (u32 i = 1; i < n_; ++i)
                lats[i] = requestLattice(i + 1);
            const Bliss::SpeechSegment*                segment = static_cast<const Bliss::SpeechSegment*>(requestData(1));
            std::pair<ConstLatticeRef, ConstFwdBwdRef> result  = fbBuilder_->build(lats);
            if (result.second) {
                ConstLatticeAlignmentRef latticeAlignment     = alignmentBuilder_->build(result.first, segment);
                cn_                                           = latticeAlignment->framePosteriorCn(result.second);
                const_cast<PosteriorCn*>(cn_.get())->alphabet = Lexicon::us()->unknownAlphabet();
            }
            else
                cn_ = ConstPosteriorCnRef();
            isValid_ = true;
        }
        return cn_;
    }

public:
    AllophoneStatePosteriorCnNode(const std::string& name, const Core::Configuration& config)
            : Node(name, config) {}
    virtual ~AllophoneStatePosteriorCnNode() {}

    virtual void init(const std::vector<std::string>& arguments) {
        if (!connected(0) || !connected(1))
            criticalError("Incoming lattice at port 0 and speech segment at port 1 expected.");
        n_ = 1;
        for (u32 i = 2; connected(i); ++i, ++n_)
            ;
        fbBuilder_        = FwdBwdBuilder::create(select("fb"));
        alignmentBuilder_ = LatticeAlignmentBuilder::create(config);

        Core::Component::Message msg(log());
        msg << "Combine " << n_ << " lattice(s):\n";

        isValid_ = false;
    }

    virtual ConstLatticeRef sendLattice(Port to) {
        verify(to == 0);
        return posteriorCn2lattice(getAllophoneStatePosteriorCn());
    }

    virtual ConstPosteriorCnRef sendPosteriorCn(Port to) {
        verify(to == 1);
        return getAllophoneStatePosteriorCn();
    }

    virtual void sync() {
        cn_.reset();
        isValid_ = false;
    }
};
NodeRef createAllophoneStatePosteriorCnNode(const std::string& name, const Core::Configuration& config) {
    return NodeRef(new AllophoneStatePosteriorCnNode(name, config));
}
// -------------------------------------------------------------------------

// -------------------------------------------------------------------------
namespace {
class PosteriorCnFeatureLattice : public RescoreLattice {
    typedef RescoreLattice Precursor;

public:
    struct Configuration {
        ScoreId phonemeScoreId;
        ScoreId phonemeConfidenceId;
        Score   phonemeAlpha;
        Configuration()
                : phonemeScoreId(Semiring::InvalidId),
                  phonemeConfidenceId(Semiring::InvalidId),
                  phonemeAlpha(0.05) {}
    };

private:
    const Configuration      config_;
    ConstLatticeAlignmentRef latticeAlignment_;
    ConstPosteriorCnRef      phonemeCn_;

protected:
    std::pair<Score, Score> posteriorScoreAndConfidence(Time t_begin, Time t_end, const SubWordAlignment& alignment, const PosteriorCn& cn, Score alpha) const {
        Time        t   = t_begin;
        Probability sum = 0.0, max = 0.0, conf = Semiring::Max, length = t_end - t_begin;
        for (SubWordAlignment::const_iterator itSw = alignment.begin(), endSw = alignment.end(); itSw != endSw; ++itSw) {
            Fsa::LabelId label = itSw->label;
            for (Time swt_end = t + itSw->duration; t != swt_end; ++t) {
                Score score = cn.score(t, label);
                sum += score;
                if (score > max)
                    max = score;
            }
            if (max < conf)
                conf = max;
            max = 0.0;
        }
        verify((0.0 <= sum) && (sum <= length + 0.005));
        verify(conf != Semiring::Max);
        return std::make_pair((length - sum) / (1.0 + alpha * (length - 1.0)), conf);
    }

public:
    PosteriorCnFeatureLattice(
            ConstLatticeRef l, RescoreMode rescoreMode, const Configuration& config,
            ConstLatticeAlignmentRef latticeAlignment, ConstPosteriorCnRef phonemeCn)
            : Precursor(l, rescoreMode), config_(config), latticeAlignment_(latticeAlignment), phonemeCn_(phonemeCn) {}

    virtual void rescore(State* sp) const {
        const Boundaries& boundaries = *fsa_->getBoundaries();
        Time              t_begin    = boundaries.get(sp->id()).time();
        for (State::iterator a = sp->begin(); a != sp->end(); ++a) {
            std::pair<Score, Score> scoreAndConf;
            if ((Fsa::FirstLabelId <= a->input()) && (a->input() <= Fsa::LastLabelId)) {
                Time                     t_end            = boundaries.get(a->target()).time();
                ConstSubWordAlignmentRef phonemeAlignment = latticeAlignment_->phonemeAlignment(sp, a);
                scoreAndConf                              = (phonemeAlignment) ? posteriorScoreAndConfidence(t_begin, t_end, *phonemeAlignment, *phonemeCn_, config_.phonemeAlpha) : std::make_pair(Semiring::Invalid, Semiring::Invalid);
            }
            else
                scoreAndConf = std::make_pair(0.0, 1.0);
            if (config_.phonemeScoreId != Semiring::InvalidId)
                a->setScore(config_.phonemeScoreId, scoreAndConf.first);
            if (config_.phonemeConfidenceId != Semiring::InvalidId)
                a->setScore(config_.phonemeConfidenceId, scoreAndConf.second);
        }
    }

    std::string describe() const {
        return Core::form("phoneme-posterior-features(%s)", fsa_->describe().c_str());
    }
};
}  // namespace

class PhonemePosteriorFeatureNode : public RescoreNode {
public:
    static const Core::ParameterString paramScoreKey;
    static const Core::ParameterString paramConfidenceKey;
    static const Core::ParameterFloat  paramAlpha;

private:
    FwdBwdBuilderRef                         fbBuilder_;
    LatticeAlignmentBuilderRef               alignmentBuilder_;
    Key                                      phonemeScoreKey_;
    Key                                      phonemeConfidenceKey_;
    PosteriorCnFeatureLattice::Configuration posteriorCnConfig_;
    ConstSemiringRef                         lastSemiring_;
    ConstLatticeRef                          l_;

protected:
    virtual ConstLatticeRef rescore(ConstLatticeRef l) {
        if (!l)
            return ConstLatticeRef();
        if (!l_) {
            if (!lastSemiring_ || (l->semiring().get() != lastSemiring_.get())) {
                lastSemiring_ = l->semiring();
                if (!phonemeScoreKey_.empty()) {
                    posteriorCnConfig_.phonemeScoreId = lastSemiring_->id(phonemeScoreKey_);
                    if (posteriorCnConfig_.phonemeScoreId == Semiring::InvalidId)
                        error("No dimension labeled \"%s\" found.", phonemeScoreKey_.c_str());
                }
                if (!phonemeConfidenceKey_.empty()) {
                    posteriorCnConfig_.phonemeConfidenceId = lastSemiring_->id(phonemeConfidenceKey_);
                    if (posteriorCnConfig_.phonemeConfidenceId == Semiring::InvalidId)
                        error("No dimension labeled \"%s\" found.", phonemeConfidenceKey_.c_str());
                }
            }
            const Bliss::SpeechSegment*                segment          = static_cast<const Bliss::SpeechSegment*>(requestData(1));
            std::pair<ConstLatticeRef, ConstFwdBwdRef> result           = fbBuilder_->build(l);
            ConstLatticeAlignmentRef                   latticeAlignment = alignmentBuilder_->build(result.first, segment);
            ConstPosteriorCnRef                        phonemeCn        = latticeAlignment->phonemeFramePosteriorCn(result.second);
            l_                                                          = ConstLatticeRef(new PosteriorCnFeatureLattice(l, rescoreMode, posteriorCnConfig_, latticeAlignment, phonemeCn));
        }
        return l_;
    }

public:
    PhonemePosteriorFeatureNode(const std::string& name, const Core::Configuration& config)
            : RescoreNode(name, config) {}
    virtual ~PhonemePosteriorFeatureNode() {}

    virtual void init(const std::vector<std::string>& arguments) {
        if (!connected(1))
            criticalError("Expect speech segment at port 1.");
        fbBuilder_                      = FwdBwdBuilder::create(select("fb"));
        alignmentBuilder_               = LatticeAlignmentBuilder::create(config);
        phonemeScoreKey_                = paramScoreKey(config);
        phonemeConfidenceKey_           = paramConfidenceKey(config);
        posteriorCnConfig_.phonemeAlpha = paramAlpha(select("score"), Semiring::Invalid);
    }

    virtual void sync() {
        l_.reset();
    }
};
const Core::ParameterString PhonemePosteriorFeatureNode::paramScoreKey(
        "score-key",
        "score key",
        "");
const Core::ParameterString PhonemePosteriorFeatureNode::paramConfidenceKey(
        "confidence-key",
        "confidence key",
        "");
const Core::ParameterFloat PhonemePosteriorFeatureNode::paramAlpha(
        "alpha",
        "alpha",
        0.05);
NodeRef createPhonemePosteriorFeatureNode(const std::string& name, const Core::Configuration& config) {
    return NodeRef(new PhonemePosteriorFeatureNode(name, config));
}
// -------------------------------------------------------------------------

// -------------------------------------------------------------------------
class AcousticOrthographyAligner : public Speech::OrthographyAlignmentGenerator, public SegmentwiseSpeechProcessor {
private:
    ConstSemiringRef semiring_;
    ScoreId          amId_;

protected:
    virtual void process(const FeatureList& features) {
        Speech::SegmentwiseFeaturesRef segmentwiseFeatures = Speech::SegmentwiseFeaturesRef(new Speech::SegmentwiseFeatures);
        for (FeatureList::const_iterator itFeature = features.begin(), endFeature = features.end();
             itFeature != endFeature; ++itFeature)
            segmentwiseFeatures->feed(*itFeature);
        setFeatures(segmentwiseFeatures);
    }

    ScoresRef buildScore(Score amScore) {
        if (amId_ != Semiring::InvalidId) {
            ScoresRef scores = semiring_->clone(semiring_->one());
            scores->set(amId_, amScore);
            return scores;
        }
        else
            return semiring_->one();
    }

public:
    AcousticOrthographyAligner(const Core::Configuration& config, ModelCombinationRef mc, ConstSemiringRef semiring, ScoreId amId)
            : Speech::OrthographyAlignmentGenerator(config, mc),
              SegmentwiseSpeechProcessor(config, mc),
              semiring_(semiring),
              amId_(amId) {
    }
    virtual ~AcousticOrthographyAligner() {}

    void align(const Bliss::SpeechSegment* segment) {
        setSpeechSegment(const_cast<Bliss::SpeechSegment*>(segment));
        processSegment(segment);
    }

    const Speech::Alignment* alignment() {
        return getAlignment();
    }

    ConstLatticeRef lattice() {
        Core::Ref<const ::Lattice::WordLattice> lattice = getWordLattice();
        if (!lattice)
            return ConstLatticeRef();

        Core::Ref<const ::Lattice::WordBoundaries> boundaries = lattice->wordBoundaries();
        Fsa::ConstAutomatonRef                     amFsa      = lattice->part(::Lattice::WordLattice::acousticFsa);
        require_(Fsa::isAcyclic(amFsa));
        StaticBoundariesRef b = StaticBoundariesRef(new StaticBoundaries);
        StaticLatticeRef    s = StaticLatticeRef(new StaticLattice);
        s->setType(Fsa::TypeAcceptor);
        s->setProperties(Fsa::PropertyAcyclic | PropertyCrossWord, Fsa::PropertyAll);
        s->setInputAlphabet(Lexicon::us()->lemmaPronunciationAlphabet());
        s->setSemiring(semiring_);
        s->setDescription(Core::form("alignment(%s,dim=%zu)", segmentId_.c_str(), amId_));
        s->setBoundaries(ConstBoundariesRef(b));
        s->setInitialStateId(0);

        Fsa::Stack<Fsa::StateId>   S;
        Core::Vector<Fsa::StateId> sidMap(amFsa->initialStateId() + 1, Fsa::InvalidStateId);
        sidMap[amFsa->initialStateId()] = 0;
        S.push_back(amFsa->initialStateId());
        Fsa::StateId nextSid = 1;
        while (!S.isEmpty()) {
            Fsa::StateId sid = S.pop();
            verify(sid < sidMap.size());
            const ::Lattice::WordBoundary& boundary = (*boundaries)[sid];
            Fsa::ConstStateRef             amSr     = amFsa->getState(sid);
            State*                         sp       = new State(sidMap[sid]);
            s->setState(sp);
            b->set(sp->id(), Boundary(
                                     boundary.time(),
                                     Boundary::Transit(boundary.transit().final, boundary.transit().initial)));
            if (amSr->isFinal())
                sp->setFinal(buildScore(amSr->weight()));
            for (Fsa::State::const_iterator am_a = amSr->begin(); am_a != amSr->end(); ++am_a) {
                sidMap.grow(am_a->target(), Fsa::InvalidStateId);
                if (sidMap[am_a->target()] == Fsa::InvalidStateId) {
                    sidMap[am_a->target()] = nextSid++;
                    S.push(am_a->target());
                }
                sp->newArc(sidMap[am_a->target()], buildScore(am_a->weight()), am_a->input());
            }
        }
        return ConstLatticeRef(s);
    }
};
// -------------------------------------------------------------------------

// -------------------------------------------------------------------------
OrthographyAlignment::OrthographyAlignment(AcousticOrthographyAligner* aligner)
        : aligner_(aligner) {}

OrthographyAlignment::~OrthographyAlignment() {}

const Speech::Alignment* OrthographyAlignment::stateAlignment() const {
    return aligner_->getAlignment();
}

ConstLatticeRef OrthographyAlignment::lattice() const {
    return aligner_->lattice();
}
// -------------------------------------------------------------------------

// -------------------------------------------------------------------------
OrthographyAlignmentBuilder::OrthographyAlignmentBuilder(const Core::Configuration& config, AcousticOrthographyAligner* aligner)
        : Core::Component(config), aligner_(aligner) {}

OrthographyAlignmentBuilder::~OrthographyAlignmentBuilder() {
    delete aligner_;
}

ConstOrthographyAlignmentRef OrthographyAlignmentBuilder::build(const Bliss::SpeechSegment* segment) {
    aligner_->align(segment);
    return ConstOrthographyAlignmentRef(new OrthographyAlignment(aligner_));
}

const Core::ParameterString paramScoreKey(
        "score-key",
        "dimension to store the alignment score",
        "");
OrthographyAlignmentBuilderRef OrthographyAlignmentBuilder::create(const Core::Configuration& config) {
    ModelCombinationRef mc       = getModelCombination(config, getAm(Core::Configuration(config, "acoustic-model")));
    ConstSemiringRef    semiring = Semiring::create(Core::Configuration(config, "semiring"));
    ScoreId             amId     = Semiring::InvalidId;
    if (semiring) {
        Key amKey = paramScoreKey(config);
        if (!amKey.empty()) {
            amId = semiring->id(amKey);
            if (amId == Semiring::InvalidId)
                Core::Application::us()->error(
                        "Semiring \"%s\" has no dimension \"%s\".",
                        semiring->name().c_str(), amKey.c_str());
        }
    }
    else {
        semiring = Semiring::create(Fsa::SemiringTypeTropical, 1, ScoreList(1, 1.0), KeyList(1, "am"));
        amId     = 0;
    }
    AcousticOrthographyAligner* aligner = new AcousticOrthographyAligner(config, mc, semiring, amId);
    return OrthographyAlignmentBuilderRef(new OrthographyAlignmentBuilder(config, aligner));
}
// -------------------------------------------------------------------------

// -------------------------------------------------------------------------
class OrthographyAlignmentNode : public Node {
private:
    OrthographyAlignmentBuilderRef builder_;
    ConstOrthographyAlignmentRef   orthAlignment_;

protected:
    ConstOrthographyAlignmentRef getOrthographyAlignment() {
        if (!orthAlignment_) {
            const Bliss::SpeechSegment* segment = static_cast<const Bliss::SpeechSegment*>(requestData(1));
            orthAlignment_                      = builder_->build(segment);
        }
        return orthAlignment_;
    }

public:
    OrthographyAlignmentNode(const std::string& name, const Core::Configuration& config)
            : Node(name, config) {}
    virtual ~OrthographyAlignmentNode() {}

    virtual void init(const std::vector<std::string>& arguments) {
        if (!connected(1))
            criticalError("Incoming speech segment at port 1 expected.");
        builder_ = OrthographyAlignmentBuilder::create(config);
    }

    virtual ConstLatticeRef sendLattice(Port to) {
        verify(to == 0);
        return getOrthographyAlignment()->lattice();
    }

    virtual const void* sendData(Port to) {
        verify(to == 1);
        return getOrthographyAlignment()->stateAlignment();
    }

    virtual void sync() {
        orthAlignment_.reset();
    }
};
NodeRef createOrthographyAlignmentNode(const std::string& name, const Core::Configuration& config) {
    return NodeRef(new OrthographyAlignmentNode(name, config));
}
// -------------------------------------------------------------------------

}  // namespace Flf
