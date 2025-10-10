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
#include "LinearSearch.hh"

#include <Lattice/LatticeAdaptor.hh>
#include <Lm/FsaLm.hh>
#include <Mm/Mixture.hh>
#include <assert.h>
#include <stdio.h>

// #define SEARCH_DEBUG

namespace Search {

const Core::ParameterBool LinearSearch::paramSingleWordRecognition_(
        "single-word-recognition",
        "only recognize single words",
        true);

LinearSearch::Pronunciation::Pronunciation(const Bliss::LemmaPronunciation* const lemma, const Am::AcousticModel* acousticModel)
        : lemma_(lemma) {
    const Bliss::Pronunciation* pron      = lemma->pronunciation();
    const Am::Phonology&        phonology = *(acousticModel->phonology());
    isRegularWord_                        = isRegularWordPrivate();

    verify(acousticModel->silence() != Bliss::Phoneme::invalidId);

    for (u32 a = 0; a < pron->length(); a++) {
        const Am::Allophone* allo = 0;

        /// Get the allophone:
        {
            u16 bound = 0;
            if (a == 0)
                bound |= Am::Allophone::isInitialPhone;
            if (a == pron->length() - 1)
                bound |= Am::Allophone::isFinalPhone;

            Am::Allophone allophone(phonology(*pron, a), bound);

            verify(acousticModel->phonemeInventory()->phoneme(allophone.phoneme(0))->isContextDependent() || (allophone.history().size() == 0 && allophone.future().size() == 0));

            allo = acousticModel->allophoneAlphabet()->allophone(allophone);
        }

        verify(allo);

        const Am::ClassicHmmTopology* hmmTopology = acousticModel->hmmTopology((*pron)[a]);
        verify(hmmTopology != 0);
        int nPhoneStates = hmmTopology->nPhoneStates();
        int nReps        = hmmTopology->nSubStates();
        verify(nPhoneStates != 0);
        verify(nReps != 0);

        bool isSilence = ((*pron)[a] == acousticModel->silence());

        verify(!isSilence || (nReps == 1 && nPhoneStates == 1));

        for (int a = 0; a < nPhoneStates; a++) {
            for (int b = 0; b < nReps; b++) {
                MixtureItem        i;
                Am::AllophoneState alloState = acousticModel->allophoneStateAlphabet()->allophoneState(allo, a);
                i.mixture                    = acousticModel->emissionIndex(alloState);

                if (isSilence)
                    i.stateTransitionModel = acousticModel->stateTransition(Am::TransitionModel::silence);
                else
                    i.stateTransitionModel = acousticModel->stateTransition(Am::TransitionModel::phone0 + b);

                mixtures_.push_back(i);
            }
        }
    }
}

bool LinearSearch::Pronunciation::isRegularWordPrivate() const {
    std::pair<Bliss::Lemma::EvaluationTokenSequenceIterator, Bliss::Lemma::EvaluationTokenSequenceIterator> eRange = lemma_->lemma()->evaluationTokenSequences();
    if (eRange.first != eRange.second) {
        for (; eRange.first != eRange.second; ++eRange.first)
            if (eRange.first->isEpsilon())
                return false;
        return true;
    }
    else
        return false;
}

// check if it's really necessary to keep this reference-counted
class WordPronunciationState : public Core::ReferenceCounted {
    typedef LinearSearch::Book Book;

public:
    /// What Hyp[wrd][sta] used to be on the papers
    struct Hypo { /*This includes lmScore, so the real score is "score - lmScore" */
        Score                                     score;
        Score                                     lmScore;
        Book*                                     bkp;
        LinearSearch::Pronunciation::MixtureItem* mixture;
    };
    typedef std::vector<Hypo> HypoVector;

    WordPronunciationState(const Bliss::LemmaPronunciation* const lemma, const Am::AcousticModel* const acousticModel, Score unigramScore)
            : pron_(new LinearSearch::Pronunciation(lemma, acousticModel)), position_(0), unigramScore_(unigramScore), irregularChain_(false) {
        hyp_.resize(pron_->nMixtures() + 1);
        HypoVector::iterator it = hyp_.begin();
        it->mixture             = 0;
        ++it;
        LinearSearch::Pronunciation::MixtureVector&          mixtures = pron_->mixtures();
        LinearSearch::Pronunciation::MixtureVector::iterator it2      = mixtures.begin();
        verify(mixtures.size() == hyp_.size() - 1);

        while (it != hyp_.end()) {
            it->mixture = &(*it2);

            ++it2;
            ++it;
        }

        restart();
    }

    ~WordPronunciationState() {
        delete pron_;
    }

    LinearSearch::Pronunciation* pronunciation() const {
        return pron_;
    }

    void restart() {
        for (HypoVector::iterator it = hyp_.begin(); it != hyp_.end(); ++it) {
            it->score   = Core::Type<Score>::max;
            it->lmScore = 0;
            it->bkp     = 0;
        }
        position_ = 0;
    }

    const u32 position() const {
        return position_;
    }

    void setPosition(const u32 pos) {
        position_ = pos;
    }

    HypoVector& hyp() {
        return hyp_;
    }

    Score unigramScore() const {
        return unigramScore_;
    }

    bool irregularChain() const {
        return irregularChain_;
    }

    void setIrregularChain(bool b = true) {
        irregularChain_ = b;
    }

private:
    WordPronunciationState(const WordPronunciationState& rhs) {}
    WordPronunciationState& operator=(const WordPronunciationState& rhs) {
        return *this;
    }

    HypoVector                   hyp_;
    LinearSearch::Pronunciation* pron_;
    u32                          position_;
    Score                        unigramScore_;
    bool                         irregularChain_;  // Whether this state should be used for chains of irregular words
};

LinearSearch::LinearSearch(const Core::Configuration& conf)
        : Core::Component(conf), SearchAlgorithm(conf), singleWordRecognition_(paramSingleWordRecognition_(conf)), time_(0) {
    log("using linear search");
    if (singleWordRecognition_)
        log("using new single-word-recognition");
};

LinearSearch::~LinearSearch() {}

bool LinearSearch::setModelCombination(const Speech::ModelCombination& modelCombination) {
    lexicon_            = modelCombination.lexicon();
    silence_            = lexicon_->specialLemma("silence");
    acousticModel_      = modelCombination.acousticModel();
    lm_                 = modelCombination.languageModel();
    pronunciationScale_ = modelCombination.pronunciationScale();

    std::pair<Bliss::Lexicon::LemmaPronunciationIterator, Bliss::Lexicon::LemmaPronunciationIterator> pronunciations = lexicon_->lemmaPronunciations();
    addPronunciations(pronunciations);
    return true;
}

void LinearSearch::setGrammar(Fsa::ConstAutomatonRef g) {
    log("Set grammar");
    require(lm_);
    const Lm::FsaLm* constFsaLm = dynamic_cast<const Lm::FsaLm*>(lm_->unscaled().get());
    require(constFsaLm);
    Lm::FsaLm* fsaLm = const_cast<Lm::FsaLm*>(constFsaLm);
    fsaLm->setFsa(g);
}

void LinearSearch::restart() {
    for (WordPronunciationStateVector::iterator it = state_.begin(); it != state_.end(); ++it) {
        (*it)->restart();
    }
    book_.clear();
    irregularBook_.clear();
    time_ = 0;
}

void LinearSearch::feed(const Mm::FeatureScorer::Scorer& emissionScores) {
    require(emissionScores);
    require(emissionScores->nEmissions() >= acousticModel_->nEmissions());

    ++time_;
    WordPronunciationState::HypoVector hypTmp;

    // emissionScores represents the scorer for only one single mixture/state
    u32 wordNumber = 0;

    for (WordPronunciationStateVector::iterator it = state_.begin(); it != state_.end(); ++it, ++wordNumber) {
#if defined SEARCH_DEBUG
        std::cout << "------------------------------------------------------------" << std::endl
                  << " word " << (*it)->pronunciation()->lemma()->lemma()->preferredOrthographicForm()
                  << " at time frame " << time_ << std::endl
                  << "------------------------------------------------------------" << std::endl;
#endif
        Book* last = 0;

        if (!book_.empty())
            last = &book_.back();

        verify((*it)->pronunciation()->lemma()->lemma() != silence_ || !(*it)->pronunciation()->isRegularWord());

        if ((singleWordRecognition_ && last && last->hadRegularWord && (*it)->pronunciation()->isRegularWord()) || (*it)->irregularChain()) {
            // Only irregular words are allowed before this one
            if (irregularBook_.empty()) {
                last = 0;
            }
            else {
                last = &irregularBook_.back();
            }
        }

        WordPronunciationState::HypoVector& hyp  = (*it)->hyp();
        Pronunciation*                      pron = (*it)->pronunciation();

        hyp[0].bkp = last;

        if (last) {
            Score localLmScore = 0;  // CHANGED from f32
            if (isUnigram()) {
                localLmScore = (*it)->unigramScore();
            }
            else {
                /// Bigram
                Lm::History h(lm_->startHistory());
                Lm::extendHistoryByLemmaPronunciation(lm_, last->word->pronunciation()->lemma(), h);
                Lm::addLemmaPronunciationScore(lm_, pron->lemma(), pronunciationScale_, lm_->scale(), h, localLmScore);
            }

            hyp[0].lmScore = localLmScore + last->lmScore;
            hyp[0].score   = last->score;
        }
        else {
            f32 localLmScore = (*it)->unigramScore();

            hyp[0].lmScore = localLmScore;
            hyp[0].score   = 0;
        }
        hyp[0].score += hyp[0].lmScore;  // The complete lm-score must be added, because last is of type Book, and in Book "lmScore" is not included in "score"

        hyp[0].mixture = 0;
#if defined SEARCH_DEBUG
        std::cout << "OLD HYP FROM TIME FRAME " << time_ - 1 << std::endl;
        for (u32 i = 0; i < hyp.size(); ++i) {
            std::cout << "hyp[" << i << "] = " << hyp[i].score << std::endl;
        }
#endif
        hypTmp.resize(hyp.size());
#if defined SEARCH_DEBUG
        std::vector<u32> buffer(hyp.size());
#endif
        for (u32 sta = 1; sta < hyp.size(); ++sta) {
            hypTmp[sta].score   = Core::Type<Score>::max;
            hypTmp[sta].lmScore = 0;

            u32 pre = sta >= 2 ? sta - 2 : 0;

            for (; pre <= sta; pre++) {
                Score scoTmp = hyp[pre].score;

                const Am::StateTransitionModel* m;
                if (pre != 0) {
                    verify(hyp[pre].mixture != 0);
                    m = hyp[pre].mixture->stateTransitionModel;
                }
                else {
                    m = acousticModel_->stateTransition(Am::TransitionModel::entryM1);  // This is the first real state, so we use the StateTransitionModel entryM1
                }

                scoTmp += (*m)[sta - pre];

                //					std::cout << "state transition score from " << pre << " to state " << sta << " = " << (*m)[ sta - pre ] << std::endl;
                if (scoTmp < hypTmp[sta].score) {
                    hypTmp[sta].score   = scoTmp;
                    hypTmp[sta].bkp     = hyp[pre].bkp;
                    hypTmp[sta].lmScore = hyp[pre].lmScore;
#if defined SEARCH_DEBUG
                    // DEBUG
                    buffer[sta] = sta - pre;
#endif
                }
            }
        }
#if defined SEARCH_DEBUG
        std::cout << "NEW HYP FOR TIME " << time_ << std::endl;
#endif
        for (u32 sta = 1; sta < hyp.size(); sta++) {
            hyp[sta].bkp     = hypTmp[sta].bkp;
            hyp[sta].score   = hypTmp[sta].score + emissionScores->score(hyp[sta].mixture->mixture);
            hyp[sta].lmScore = hypTmp[sta].lmScore;
#if defined SEARCH_DEBUG
            // DEBUG
            std::cout << "tmpHyp[" << sta << "] = " << hyp[sta].score << " | transition: " << buffer[sta] << " local score: "
                      << emissionScores->score(hyp[sta].mixture->mixture)
                      << " lmScore " << hyp[sta].lmScore
                      << std::endl;
#endif
        }
    }

    book_.push_back(Book());

    Book& newBook(book_.back());
    newBook.score = Core::Type<Score>::max;

    bookKeeping(newBook);

    if (newBook.score == Core::Type<Score>::max)
        book_.pop_back();

    if (singleWordRecognition_) {
        // When single-word recognition is enabled it is also neccessary to keep track of irregular-book-keeping
        irregularBook_.push_back(Book());

        Book& newBook(irregularBook_.back());
        newBook.score = Core::Type<Score>::max;

        bookKeeping(newBook, true);

        if (newBook.score == Core::Type<Score>::max)
            irregularBook_.pop_back();
        else
            verify(!irregularBook_.back().hadRegularWord);
    }
}

void LinearSearch::bookKeeping(Book& newBook, bool irregular) {
#if defined SEARCH_DEBUG
    // DEBUG
    u32   wordNumber = 0;
    u32   chosen     = 0;
    Score exit       = 0;
#endif
    for (WordPronunciationStateVector::iterator it = state_.begin(); it != state_.end(); ++it)  //, ++wordNumber )
    {
        WordPronunciationStatePointer p = (*it);
        verify(!p->hyp().empty());
        WordPronunciationState::Hypo& hyp = p->hyp().back();
        if (irregular && p->pronunciation()->isRegularWord()) {
            continue;  // Drop regular words when only irregular are wanted
        }
        if (irregular && hyp.bkp && hyp.bkp->hadRegularWord) {
            continue;  // Drop hypotheses whose histories contain regular words when only irregular are wanted
        }

        Score tmpScore = hyp.score;
        if (hyp.mixture)
            tmpScore += (*hyp.mixture->stateTransitionModel)[Am::StateTransitionModel::exit];
        else {
            continue;
        }

        if (tmpScore < newBook.score + newBook.lmScore) {
            newBook.score   = tmpScore - hyp.lmScore;
            newBook.lmScore = hyp.lmScore;
            newBook.bkp     = hyp.bkp;
            newBook.word    = *it;

            if (newBook.word->pronunciation()->isRegularWord()) {
                verify(!singleWordRecognition_ || (!newBook.bkp || newBook.bkp->hadRegularWord == false));
                newBook.hadRegularWord = true;
            }
            else if (newBook.bkp) {
                newBook.hadRegularWord = newBook.bkp->hadRegularWord;
            }
            else {
                newBook.hadRegularWord = false;
            }

            newBook.time = time_;
#if defined SEARCH_DEBUG
            // DEBUG
            chosen = wordNumber;
            exit   = (*hyp.mixture->stateTransitionModel)[Am::StateTransitionModel::exit];
#endif
        }
    }
}

bool LinearSearch::isUnigram() const {
    return true;  // false; //Always return false, because the score-precomputation doesn't work correctly
}

void LinearSearch::getCurrentBestSentence(Traceback& result) const {
    result.clear();

    if (book_.empty())
        return;

    Lm::History h(lm_->startHistory());

    if (book_.back().bkp)
        Lm::extendHistoryByLemmaPronunciation(lm_, book_.back().bkp->word->pronunciation()->lemma(), h);

    Lm::extendHistoryByLemmaPronunciation(lm_, book_.back().word->pronunciation()->lemma(), h);

    result.push_back(TracebackItem(0, time_, ScoreVector(book_.back().score, book_.back().lmScore + lm_->sentenceEndScore(h)), TracebackItem::Transit()));

    const Book* bkp = &book_.back();
    int         cnt = 0;

    while (bkp != 0) {
        cnt++;
        TracebackItem item(bkp->word->pronunciation()->lemma(), bkp->time, ScoreVector(bkp->score, bkp->lmScore), TracebackItem::Transit());
        result.push_back(item);
        bkp = bkp->bkp;
    }
    if (cnt != 0)
        log("returning %i words", cnt);

    result.push_back(TracebackItem(0, 0, ScoreVector(0, 0), TracebackItem::Transit()));

    std::reverse(result.begin(), result.end());
}

void LinearSearch::getPartialSentence(Traceback& result) {
    getCurrentBestSentence(result);
    restart();
}

Core::Ref<const LatticeAdaptor> LinearSearch::getCurrentWordLattice() const {
    return Core::ref(new Lattice::WordLatticeAdaptor);
}

void LinearSearch::resetStatistics() {}

void LinearSearch::logStatistics() const {}

LinearSearch::Book::Book()
        : score(Core::Type<Score>::max), lmScore(0), hadRegularWord(false), bkp(0), time(0) {
}

LinearSearch::Book::~Book() {
}

template<class IteratorType>
LinearSearchHistoryData LinearSearch::addPronunciations(std::pair<IteratorType, IteratorType> pronunciations) {
    LinearSearchHistoryData ret;
    ret.success = true;

    for (; pronunciations.first != pronunciations.second; ++pronunciations.first) {
        Score       unigramScore = 0;
        Lm::History emptyHistory(lm_->startHistory());
        Lm::addLemmaPronunciationScore(lm_, (*pronunciations.first), pronunciationScale_, lm_->scale(), emptyHistory, unigramScore);

        WordPronunciationStatePointer p(new WordPronunciationState(*pronunciations.first, acousticModel_.get(), unigramScore));
        state_.push_back(p);
        ret.states.push_back(p);
        if (singleWordRecognition_ && !p->pronunciation()->isRegularWord()) {
            // Add an additional state that is necessary for tracking the probability of chains of irregular words
            WordPronunciationStatePointer p2(new WordPronunciationState(*pronunciations.first, acousticModel_.get(), unigramScore));
            p2->setIrregularChain();
            state_.push_back(p2);
            ret.states.push_back(p2);
        }
    }

    return ret;
}

bool LinearSearch::removePronunciations(const LinearSearchHistoryData& data) {
    for (std::list<WordPronunciationStatePointer>::const_iterator it = data.states.begin(); it != data.states.end(); ++it) {
        /// Make more efficient
        for (WordPronunciationStateVector::iterator it2 = state_.begin(); it2 != state_.end(); ++it2) {
            if (*it == *it2) {
                state_.erase(it2);
                break;
            }
        }
    }
    return true;
}

LinearSearchHistoryData::LinearSearchHistoryData()
        : success(false) {
}

LinearSearchHistoryData::~LinearSearchHistoryData() {
}
};  // namespace Search
