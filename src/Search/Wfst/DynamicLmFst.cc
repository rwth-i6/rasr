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
#include <Core/Application.hh>
#include <OpenFst/SymbolTable.hh>
#include <Search/Wfst/DynamicLmFst.hh>
#include <ext/numeric>

namespace Search {
namespace Wfst {

class DynamicLmFstScoreCache {
public:
    typedef std::vector<float> Cache;
    DynamicLmFstScoreCache(size_t maxElements)
            : maxElements_(maxElements),
              elements_(0) {}
    ~DynamicLmFstScoreCache();
    Cache* Get(u32 s) {
        if (s < data_.size()) {
            data_[s].second = true;
            return data_[s].first;
        }
        else {
            return 0;
        }
    }
    void Set(u32 s, Cache* cache) {
        if (s < data_.size() && data_[s].first) {
            delete data_[s].first;
            data_[s].first = cache;
        }
        else {
            if ((elements_ + 1) > maxElements_)
                CleanCache(false);
            ++elements_;
            if (s >= data_.size())
                data_.resize(s + 1, Element(0, false));
            data_[s].first = cache;
        }
        data_[s].second = true;
    }
    void Clear() {
        elements_ = 0;
        for (std::vector<Element>::iterator i = data_.begin(); i != data_.end(); ++i)
            delete i->first;
        data_.clear();
    }
    u32 Size() const {
        return elements_;
    }

private:
    typedef std::pair<Cache*, bool> Element;
    void                            CleanCache(bool);
    size_t                          maxElements_;
    size_t                          elements_;
    std::vector<Element>            data_;

    void operator=(const DynamicLmFstScoreCache&);
    DynamicLmFstScoreCache(const DynamicLmFstScoreCache&);
};

DynamicLmFstScoreCache::~DynamicLmFstScoreCache() {
    for (std::vector<Element>::iterator i = data_.begin(); i != data_.end(); ++i)
        delete i->first;
}

void DynamicLmFstScoreCache::CleanCache(bool freeRecent) {
    size_t                         targetSize = (2 * maxElements_) / 3 + 1;
    std::vector<Element>::iterator i          = data_.begin();
    while (i != data_.end() && elements_ > targetSize) {
        if (i->first) {
            if (freeRecent || !i->second) {
                delete i->first;
                i->first = 0;
                --elements_;
            }
            i->second = false;
        }
        ++i;
    }
    if (!freeRecent && elements_ > targetSize)
        CleanCache(true);
}

DynamicLmFstImpl::DynamicLmFstImpl(const DynamicLmFstOptions& opts)
        : CacheImpl(opts),
          lm_(opts.lm),
          lemmas_(lm_->lexicon()->lemmaAlphabet()),
          wpScale_(opts.pronunciationScale),
          nCalculated_(0),
          nCached_(0),
          silenceWeight_(opts.silenceWeight),
          batchRequest_(0),
          scoreCache_(new DynamicLmFstScoreCache(MaxScoreCaches)) {
    require(lm_);
    SetType("dynamic-lm");
    OpenFst::SymbolTable* symbols = 0;
    const Bliss::Lemma*   silence = lm_->lexicon()->specialLemma("silence");
    if (opts.outputType == OutputLemmaPronunciation) {
        lemmaProns_ = lm_->lexicon()->lemmaPronunciationAlphabet();
        symbols     = OpenFst::convertAlphabet(lemmaProns_, "lemma-pronunciations");
        silence_    = OpenFst::convertLabelFromFsa(silence->pronunciations().first->id());
    }
    else {
        symbols  = OpenFst::convertAlphabet(lemmas_, "lemmas");
        silence_ = OpenFst::convertLabelFromFsa(silence->id());
    }
    silenceLabel_ = silence_;
    SetInputSymbols(symbols);
    SetOutputSymbols(symbols);
    nLabels_ = nArcs_ = symbols->NumSymbols() - 1;
    delete symbols;
    SetProperties(kProperties, kProperties);
    batchRequest_ = CompileBatchRequest();
}

DynamicLmFstImpl::DynamicLmFstImpl(const DynamicLmFstImpl& impl)
        : CacheImpl(impl),
          lm_(impl.lm_),
          lemmas_(impl.lemmas_),
          lemmaProns_(impl.lemmaProns_),
          wpScale_(impl.wpScale_),
          nLabels_(impl.nLabels_),
          nArcs_(impl.nArcs_),
          nCalculated_(0),
          nCached_(0),
          silence_(impl.silence_),
          silenceLabel_(impl.silenceLabel_),
          batchRequest_(0),
          scoreCache_(new DynamicLmFstScoreCache(MaxScoreCaches)) {
    SetType("dynamic-lm");
    SetProperties(impl.Properties(), FstLib::kCopyProperties);
    SetInputSymbols(impl.InputSymbols());
    SetOutputSymbols(impl.OutputSymbols());
    batchRequest_ = CompileBatchRequest();
}

DynamicLmFstImpl::~DynamicLmFstImpl() {
    Core::Application::us()->log()
            << Core::XmlOpen("statistics") + Core::XmlAttribute("name", "dynamic LM")
            << Core::XmlFull("states", state2History_.size())
            << Core::XmlFull("cache-size", cachedArcs_.size())
            << Core::XmlFull("arcs-calculated", nCalculated_)
            << Core::XmlFull("cached-requests", nCached_)
            << Core::XmlFull("score-caches", scoreCache_->Size())
            << Core::XmlClose("statistics");
    delete batchRequest_;
    delete scoreCache_;
}

Lm::CompiledBatchRequest* DynamicLmFstImpl::CompileBatchRequest() const {
    Lm::BatchRequest batch;
    for (Label l = 1; l <= nArcs_; ++l) {
        Label                                wordLabel     = GetLabel(l);
        Lm::Score                            score         = 0;
        const Bliss::SyntacticTokenSequence& tokenSequence = SyntacticToken(wordLabel, &score);
        Lm::Request                          request(tokenSequence, l, score);
        batch.push_back(request);
    }
    return lm_->compileBatchRequest(batch, 1.0);
}

void DynamicLmFstImpl::CacheScores(StateId s) const {
    if (scoreCache_->Get(s))
        return;
    std::vector<float>* cache   = new std::vector<float>(nArcs_ + 1, Core::Type<Lm::Score>::max);
    const Lm::History&  history = state2History_[s];
    lm_->getBatch(history, batchRequest_, *cache);
    cache->at(silenceLabel_) = silenceWeight_.Value();
    scoreCache_->Set(s, cache);
}

const DynamicLmFstImpl::ScoreCache* DynamicLmFstImpl::GetScores(StateId s) const {
    return scoreCache_->Get(s);
}

inline DynamicLmFstImpl::StateId DynamicLmFstImpl::GetState(const Lm::History& history) {
    HistoryMap::const_iterator i     = history2State_.find(history);
    StateId                    state = FstLib::kNoStateId;
    if (i == history2State_.end()) {
        state = state2History_.size();
        state2History_.push_back(history);
        history2State_.insert(HistoryMap::value_type(history, state));
    }
    else {
        state = i->second;
    }
    return state;
}

inline const Lm::History& DynamicLmFstImpl::GetHistory(StateId state) const {
    require(state < state2History_.size());
    return state2History_[state];
}

DynamicLmFstImpl::StateId DynamicLmFstImpl::Start() {
    if (!HasStart()) {
        Lm::History start = lm_->startHistory();
        SetStart(GetState(start));
    }
    return CacheImpl::Start();
}

DynamicLmFstImpl::Weight DynamicLmFstImpl::Final(StateId s) {
    /**! @todo: add option for explicit sentence end tokens */
    if (!HasFinal(s)) {
        const Lm::History& history = state2History_[s];
        CacheImpl::SetFinal(s, lm_->sentenceEndScore(history));
    }
    return CacheImpl::Final(s);
}

void DynamicLmFstImpl::Expand(StateId s) {
    defect();
    CacheImpl::ReserveArcs(s, nArcs_);
    for (Label l = 1; l <= nArcs_; ++l) {
        Arc arc;
        CreateArc(s, l, &arc, false);
        CacheImpl::PushArc(s, arc);
    }
    SetArcs(s);
}

const Bliss::SyntacticTokenSequence& DynamicLmFstImpl::SyntacticToken(Label wordLabel, f32* pronScore) const {
    const Bliss::Lemma* lemma = 0;
    if (lemmaProns_) {
        const Bliss::LemmaPronunciation* lp = lemmaProns_->lemmaPronunciation(OpenFst::convertLabelToFsa(wordLabel));
        lemma                               = lp->lemma();
        *pronScore += wpScale_ * lp->pronunciationScore();
    }
    else {
        lemma = lemmas_->lemma(OpenFst::convertLabelToFsa(wordLabel));
    }
    return lemma->syntacticTokenSequence();
}

void DynamicLmFstImpl::CreateArc(StateId source, Label label, Arc* arc, bool cache) {
    if (GetCachedArc(source, label, arc)) {
        ++nCached_;
        return;
    }
    const ScoreCache* scores = GetScores(source);
    ++nCalculated_;
    Lm::History history   = state2History_[source];
    Label       wordLabel = GetLabel(label);
    verify(wordLabel != FstLib::kNoLabel);
    Lm::Score                            score         = 0;
    const Bliss::SyntacticTokenSequence& tokenSequence = SyntacticToken(wordLabel, &score);
    for (u32 ti = 0; ti < tokenSequence.length(); ++ti) {
        const Bliss::SyntacticToken* st = tokenSequence[ti];
        score += scores ? scores->at(label) : lm_->score(history, st);
        history = lm_->extendedHistory(history, st);
    }
    arc->ilabel    = label;
    arc->olabel    = wordLabel;
    arc->weight    = score;
    arc->nextstate = GetState(history);
    if (cache) {
        CacheArc(source, *arc);
    }
}

void DynamicLmFstImpl::CacheArc(StateId s, const Arc& arc) {
    cachedArcs_.insert(ArcCache::value_type(ArcCacheKey(s, arc.ilabel), arc));
}

bool DynamicLmFstImpl::GetCachedArc(StateId s, Label l, Arc* arc) const {
    ArcCache::const_iterator i = cachedArcs_.find(ArcCacheKey(s, l));
    if (i == cachedArcs_.end()) {
        return false;
    }
    else {
        *arc = i->second;
        return true;
    }
}

void DynamicLmFstImpl::SetLabelMapping(const std::vector<std::pair<Label, Label>>& map) {
    if (!map.empty()) {
        relabeling_.resize(InputSymbols()->NumSymbols(), FstLib::kNoLabel);
        __gnu_cxx::iota(relabeling_.begin(), relabeling_.end(), 0);
        typedef std::vector<std::pair<Label, Label>>::const_iterator Iter;
        for (Iter i = map.begin(); i != map.end(); ++i) {
            verify_lt(i->second, relabeling_.size());
            relabeling_[i->second] = i->first;
            if (i->first == silence_)
                silenceLabel_ = i->second;
        }
        nArcs_ = map.size();
        SetProperties(FstLib::kNotAcceptor,
                      FstLib::kAcceptor | FstLib::kNotAcceptor);
        SetProperties(FstLib::kNotOLabelSorted,
                      FstLib::kNotOLabelSorted | FstLib::kOLabelSorted);
        scoreCache_->Clear();
        delete batchRequest_;
        batchRequest_ = CompileBatchRequest();
    }
}

}  // namespace Wfst
}  // namespace Search
